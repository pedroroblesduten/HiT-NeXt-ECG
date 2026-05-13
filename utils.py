import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist


def get_relative_distances(window_size):
    """
    Compute relative distances between positions in a 1D window.

    Args:
        window_size (int): The size of the window.

    Returns:
        torch.Tensor: A tensor containing the relative position indices.
    """
    indices = torch.arange(window_size)
    relative_indices = indices[None, :] - indices[:, None]
    return relative_indices + window_size - 1


def get_1d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 1D sinusoidal positional embeddings using PyTorch, with a calculation
    method similar to the provided NumPy snippet.

    Args:
        embed_dim: Dimensionality of the embeddings. Must be even.
        grid_size: Number of positions for which embeddings are generated.

    Returns:
        A PyTorch nn.Parameter containing the positional embeddings.
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Generate position indices
    pos = torch.arange(grid_size).float().unsqueeze(1)

    # Calculate omega based on embed_dim (using the provided mathematical approach)
    omega = torch.arange(embed_dim // 2).float()
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))

    # Calculate the outer product, similar to np.einsum('m,d->md', pos, omega)
    out = pos * omega

    # Generate sinusoidal embeddings
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # Concatenate sin and cos embeddings
    emb = torch.cat([emb_sin, emb_cos], dim=1)

    # Wrap the embeddings in nn.Parameter, with requires_grad=False
    pos_embed = torch.nn.Parameter(emb.unsqueeze(0), requires_grad=False)
    return pos_embed


def off_diagonal(x):
    """
    Extract the off-diagonal elements of a square matrix.

    Args:
        x (torch.Tensor): A square matrix of shape (D, D)

    Returns:
        torch.Tensor: A 1D tensor containing all off-diagonal elements of x, flattened.
    """
    n, m = x.shape
    assert (
        n == m
    ), "Input must be of shape (D, D)"  # Ensure the input is a square matrix

    # Flatten the matrix, remove the last element to prepare for reshaping
    # The view reshapes it to (D-1, D+1) so the diagonal elements are aligned diagonally
    # in this new view.
    x_off = x.flatten()[:-1].view(n - 1, n + 1)

    # Remove the first column of each row (which corresponds to the diagonal)
    # and flatten again to get all off-diagonal elements
    return x_off[:, 1:].flatten()


class OutputHead(nn.Module):
    """
    Generic output head for an arbitrary sequence encoder.

    Parameters
    ----------
    embedding_dim  : int              –  D
    hidden_dim     : int
    out_dim      : int
    dropout        : float
    linear_only    : bool             –  use single linear layer if True
    norm           : bool             –  add LayerNorm before the head if True
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.7,
        linear_only: bool = False,
        norm: bool = True,
    ):
        super().__init__()

        # Build head layers ---------------------------------------------------
        layers = nn.ModuleList()
        if norm:
            layers.append(nn.LayerNorm(embedding_dim))

        if linear_only:
            layers.append(nn.Linear(embedding_dim, out_dim))
        else:
            layers.extend(
                [
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                ]
            )

        self.head = nn.Sequential(*layers)

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=1)  # global average over sequence (B, L, D)
        return self.head(out)


class GRN(nn.Module):
    """
    Gated Residual Normalization (GRN) module.

    This module normalizes input features and applies learnable scaling and shifting.

    Args:
        dim (int): Dimensionality of input features.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        """
        Forward pass of the GRN module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim, sequence_length).

        Returns:
            torch.Tensor: Transformed tensor with normalized scaling and bias applied.
        """
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm(nn.Module):
    """
    LayerNorm that supports channels_last (N, L, C) or channels_first (N, C, L).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ):
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise NotImplementedError(f"Unsupported data_format={data_format}")
        self.data_format = data_format
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":  # (N, L, C)
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:  # (N, C, L)
            mean = x.mean(dim=2, keepdim=True)
            var = (x - mean).pow(2).mean(dim=2, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None] * x + self.bias[:, None]


class SEM(nn.Module):
    def __init__(self, L, V, tau):
        super().__init__()
        self.L = L
        self.V = V
        self.tau = tau

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        assert dim == self.L * self.V, "Invalid dim for SEM."

        logits = x.view(batch_size * seq_len, self.L, self.V)

        return F.softmax(logits / self.tau, -1).view(batch_size, seq_len, dim)


# OBS: FullGatherLayer sometimes does not work for small batch sizes due to all_gather inconsistencies.
# Batch size > 128 works fine. If it becomes a problem, look into padding for making batches the same shape.
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all ranks and support backward propagation—even when
    each rank has a different local batch size.

    * Forward  :
        1.  all_gather local batch-sizes (int32)
        2.  pad the local tensor to max(batch_sizes)
        3.  all_gather the padded tensors
        4.  unpad each slice back to its true length
    * Backward :
        1.  pad each incoming grad to max(batch_sizes)
        2.  sum them (preserves original behaviour)
        3.  remove the padding that belongs to *this* rank
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        if not dist.is_available() or not dist.is_initialized():
            return (x,)

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = x.device

        # --- 1. Exchange local batch-sizes -------------------------------------------------
        local_bs = torch.tensor([x.size(0)], device=device, dtype=torch.int32)
        bs_list = [torch.zeros_like(local_bs) for _ in range(world_size)]
        dist.all_gather(bs_list, local_bs)
        bs_list = [int(bs.item()) for bs in bs_list]
        max_bs = max(bs_list)

        # Save context for backward
        ctx.bs_list = bs_list
        ctx.max_bs = max_bs
        ctx.rank = rank

        # --- 2. Pad -----------------------------------------------------------------------
        if local_bs < max_bs:
            pad_shape = [max_bs - local_bs] + list(x.shape[1:])
            pad = x.new_zeros(pad_shape)  # zeros do NOT affect contrastive losses
            x_padded = torch.cat([x, pad], dim=0)
        else:
            x_padded = x

        # --- 3. All-gather ----------------------------------------------------------------
        gather_list = [torch.empty_like(x_padded) for _ in range(world_size)]
        dist.all_gather(gather_list, x_padded)

        # --- 4. Un-pad --------------------------------------------------------------------
        out = tuple(t[:bs] for t, bs in zip(gather_list, bs_list))
        return out

    @staticmethod
    def backward(ctx, *grads):
        """
        Preserve the original “sum-all-grads” semantics—even with uneven shapes.
        """
        device = grads[0].device
        max_bs = ctx.max_bs

        # 1. Pad every incoming grad so they align for summation
        padded = []
        for g, bs in zip(grads, ctx.bs_list):
            if bs < max_bs:
                pad_shape = [max_bs - bs] + list(g.shape[1:])
                g = torch.cat([g, g.new_zeros(pad_shape)], dim=0)
            padded.append(g)

        # 2. Sum across ranks (matches original implementation)
        grad_sum = torch.stack(padded).sum(dim=0)

        # 3. Return only the slice that corresponds to *this* rank
        grad_input = grad_sum[: ctx.bs_list[ctx.rank]]
        return grad_input


class CoPE(nn.Module):
    """
    Contextual Positional Encoding (CoPE) module.

    This module computes position embeddings dynamically based on attention logits,
    allowing for flexible and adaptive positional encoding.

    Args:
        npos_max (int): Maximum number of discrete positions.
        head_dim (int): Dimensionality of the attention head.
    """

    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        """
        Forward pass for the CoPE module.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_length, head_dim).
            attn_logits (torch.Tensor): Attention logits of shape (batch_size, num_heads, seq_length, seq_length).

        Returns:
            torch.Tensor: Contextual positional encodings of shape (batch_size, num_heads, seq_length, head_dim).
        """
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)

        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor

        return logits_ceil * w + logits_floor * (1 - w)


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with GELU activation and dropout.

    This module consists of two linear layers with an intermediate GELU activation
    and optional dropout.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden units. Defaults to `in_features`.
        out_features (int, optional): Number of output features. Defaults to `in_features`.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        p_dropout=0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, out_features).
        """
        out = self.linear1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)

        return out
