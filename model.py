import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import einsum
from einops import rearrange
from timm.layers import trunc_normal_
from .utils import (
    get_1d_sincos_pos_embed,
    get_relative_distances,
    CoPE,
    MLP,
    GRN,
    OutputHead,
    SEM,
)


class ECGWindowAttention(nn.Module):
    """
    Window-based Self-Attention module for ECG signals.

    This module performs self-attention within fixed-size windows and optionally applies
    relative positional encoding (RPE), cyclic shifts, and cosine similarity-based attention.

    Args:
        dim (int): Feature dimension.
        heads (int): Number of attention heads.
        window_size (int): Size of each attention window.
        use_rpe (bool): Whether to use relative positional encoding (RPE).
        rpe_type (str): Type of RPE to use. Options are "cope", "default", or "combined".
        shift (bool): Whether to apply cyclic shift for shifted window attention.
        shift_size (int): Cyclic shift size. Defaults to 'window_size//2'.
        cosine_sim (bool): Whether to use cosine similarity for attention computation.
        compute_attn_ent (bool): Whether to compute the attention entropy.
    """

    def __init__(
        self,
        dim,
        heads,
        window_size,
        use_rpe,
        rpe_type="cope",
        shift=False,
        shift_size=None,
        cosine_sim=True,
        compute_attn_ent=False,
    ):
        super().__init__()
        head_dim = dim // heads
        self.heads = heads
        self.scale = head_dim**-0.5
        self.window_size = window_size
        self.shift = shift
        self.shift_size = shift_size
        self.use_rpe = use_rpe
        self.rpe_type = rpe_type
        self.cosine_sim = cosine_sim
        self.compute_attn_ent = compute_attn_ent

        # Current batch attention entropy
        self.curr_attn_ent = None

        # Linear projection for queries, keys, and values
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        if self.cosine_sim:
            # Learnable temperature parameter for cosine similarity attention
            self.tau = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        # Relative positional encoding setup
        if self.use_rpe:
            if self.rpe_type == "cope":  # CoPE-based relative position encoding
                self.cope = CoPE(window_size, head_dim)

            elif self.rpe_type == "combined":  # Combination of CoPE and standard RPE
                self.cope = CoPE(window_size, head_dim)
                self.rpe_alpha = nn.Parameter(torch.ones(2), requires_grad=True)
                self.relative_indices = get_relative_distances(window_size)
                self.pos_embedding = nn.Parameter(torch.zeros(2 * window_size - 1))

            elif self.rpe_type == "default":  # Standard relative position encoding
                self.relative_indices = get_relative_distances(window_size)
                self.pos_embedding = nn.Parameter(torch.zeros(2 * window_size - 1))

            else:
                raise ValueError("Invalid value for 'rpe_type'.")

        self.to_out = nn.Linear(dim, dim)

    def _create_mask_1d(self, window_size, displacement):
        mask = torch.zeros(window_size, window_size)
        if displacement > 0:
            mask[:-displacement, -displacement:] = float("-inf")
            mask[-displacement:, :-displacement] = float("-inf")

        return mask

    def _get_pos_embedding(self, query, attn_logits, mask=None):
        """
        Compute relative positional encodings based on the selected RPE type.

        Args:
            query (torch.Tensor): Query tensor (batch_size, num_heads, num_windows, seq_length, head_dim).
            attn_logits (torch.Tensor): Attention logits (batch_size, num_heads, num_windows, seq_length, seq_length).
            mask (torch.Tensor, optional): Mask indicating the valid positions in the sequence.

        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        if self.rpe_type == "cope":
            return self.cope(query, attn_logits)

        else:
            if mask is None:
                relative_indices = self.relative_indices
            else:
                context_index = mask.to(self.relative_indices.device)
                relative_indices = self.relative_indices[context_index][
                    :, context_index
                ]

            if self.rpe_type == "combined":
                cope_embeddings = self.cope(query, attn_logits)
                rpe_embeddings = self.pos_embedding[relative_indices]

                norm_rpe_alpha = self.rpe_alpha / torch.norm(self.rpe_alpha)
                embeddings = (
                    norm_rpe_alpha[0] * cope_embeddings
                    + norm_rpe_alpha[1] * rpe_embeddings
                )

                return embeddings

            else:
                return self.pos_embedding[relative_indices]

    def _entropy(self, attn_probs: torch.Tensor, eps: float = 1e-9):
        """
        Compute token-wise attention entropy H_{lhi}
        (paper eq. (5)). Expects *probabilities* after softmax.

        Args
        ----
        attn_probs : Tensor
            Shape (B, heads, n_windows, L, L) with ∑_j A_ij = 1
        Returns
        -------
        Tensor
            Shape (B, heads, n_windows, L) with
            H_ij = -Σ_j A_ij log A_ij
        """
        self.curr_attn_ent = (-attn_probs * (attn_probs + eps).log()).sum(dim=-1)

    def forward(self, x, mask=None):
        """
        Forward pass with optional context mask, restricting attention to specific positions.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_length, feature_dim).
            mask (torch.Tensor): List indicating valid positions within a window.

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        b, l, _ = x.shape

        if mask is None:
            window_size = self.window_size
        else:
            window_size = mask.shape[0]

        shift_size = (
            self.shift_size or window_size // 2
        )  # Shift size defaults to window_size//2 if not specified
        nw_l = l // window_size  # Number of windows along the sequence length

        if self.shift:
            x_shift = torch.roll(x, shifts=-shift_size, dims=1)
        else:
            x_shift = x

        qkv = self.to_qkv(x_shift)
        q, k, v = (
            qkv.view(b, nw_l, window_size, self.heads, -1)
            .permute(0, 3, 1, 2, 4)
            .chunk(3, dim=4)
        )

        if self.cosine_sim:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            attn = einsum("b h w i d, b h w j d -> b h w i j", q, k) / self.tau
        else:
            attn = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        # Apply attention mask if cyclic shift is used
        if self.shift:
            attn_mask = self._create_mask_1d(window_size, shift_size).to(attn.device)
            attn[:, :, -1] += attn_mask

        if self.use_rpe:
            attn += self._get_pos_embedding(query=q, attn_logits=attn, mask=mask)

        attn = attn.softmax(dim=-1)
        if self.compute_attn_ent:
            self._entropy(attn)

        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(out, "b h w l d -> b (w l) (h d)")
        out = self.to_out(out)

        if self.shift:
            out = torch.roll(out, shifts=shift_size, dims=1)

        return out


class ECGSwinBlock(nn.Module):
    """
    Swin Transformer Block for ECG signal processing.

    This block applies window-based self-attention followed by a feed-forward network.
    It supports optional shifted windows and relative positional encoding (RPE).

    Args:
        dim (int): Feature dimension.
        heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension for the feed-forward network.
        shift (bool): Whether to apply cyclic shift for shifted window attention.
        shift_size (int): Cyclic shift size. Defaults to 'window_size//2'.
        window_size (int): Size of the attention window.
        use_rpe (bool): Whether to use relative positional encoding (RPE).
        rpe_type (str): Type of RPE to use ("cope", "default", "combined").
        cosine_sim (bool): Whether to use cosine similarity for attention computation.
        compute_attn_ent (bool): Whether to compute the attention entropy.
    """

    def __init__(
        self,
        dim,
        heads,
        mlp_dim,
        shift,
        shift_size,
        window_size,
        use_rpe,
        rpe_type,
        cosine_sim,
        compute_attn_ent,
    ):
        super().__init__()
        self.attention = ECGWindowAttention(
            dim=dim,
            heads=heads,
            window_size=window_size,
            use_rpe=use_rpe,
            rpe_type=rpe_type,
            shift=shift,
            shift_size=shift_size,
            cosine_sim=cosine_sim,
            compute_attn_ent=compute_attn_ent,
        )
        self.layer_norm1 = nn.LayerNorm(dim)
        self.feed_forward = MLP(
            in_features=dim,
            hidden_features=mlp_dim,
            out_features=dim,
            p_dropout=0.1,
        )
        self.layer_norm2 = nn.LayerNorm(dim)

        self.dim = dim
        self.window_size = window_size

    def forward(self, x, mask=None):
        """
        Forward pass with optional mask.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        # Apply first normalization and self-attention
        residual = x
        out = self.layer_norm1(x)
        out = self.attention(out, mask)
        out = out + residual  # Add residual connection

        # Apply second normalization and feed-forward network
        residual = out
        out = self.layer_norm2(out)
        out = self.feed_forward(out)
        out = out + residual  # Add residual connection

        return out


class ConvMerge(nn.Module):
    """
    Convolutional Module for Merging Patches in ECG Signals.

    This module applies multiple convolutional layers to merge patches from ECG signals.
    It includes two sequential convolutional branches with skip connections to enhance
    information flow and gradient propagation.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        seq_len (int): Length of the input sequence.
        config (dict): Dictionary containing configurations for convolutional layers,
                       dropout rate, and max pooling.
    """

    def __init__(self, in_c, out_c, seq_len, config):
        super().__init__()

        self.seq_len = seq_len
        self.scale_factor = config["reduction"]

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Conv1d(in_c, out_c, **config["conv_reduc"]),  # Reduce channel size
            nn.LayerNorm([out_c, seq_len]),  # Apply LayerNorm
            nn.Conv1d(out_c, out_c * 4, **config["conv_same"]),  # Expand channel size
            nn.GELU(),  # Non-linearity
            nn.Dropout(config["dropout_rate"]),  # Apply dropout
            GRN(out_c * 4),  # Gated Residual Normalization for feature scaling
            nn.Conv1d(out_c * 4, out_c, **config["conv_same"]),  # Restore channel size
        )

        # Skip connection for residual learning
        self.skip_connection = nn.Sequential(
            nn.MaxPool1d(**config["max_pool_sc"]),  # Downsample input
            nn.Conv1d(in_c, out_c, **config["conv_sc"]),  # Adjust channel size
        )

        # Final convolutional branch
        self.final_layer = nn.Sequential(
            nn.LayerNorm([out_c, seq_len]),  # Apply LayerNorm
            nn.Conv1d(out_c, out_c * 4, **config["conv_same"]),  # Expand channel size
            nn.GELU(),  # Non-linearity
            GRN(out_c * 4),  # Gated Residual Normalization
            nn.Conv1d(out_c * 4, out_c, **config["conv_same"]),  # Restore channel size
        )

    def _upsample_mask(self, mask):
        """
        mask:       Tensor of shape (n, ) indicating visible indices.
        Returns:    The upsampled binary mask of shape (L, ).
        """

        # Initialize binary mask
        b_mask = torch.zeros(self.seq_len, dtype=torch.float32, device=mask.device)

        # Set indices to 1
        b_mask[mask] = 1.0

        # Upsample using nearest neighbor
        return (
            F.interpolate(
                b_mask.unsqueeze(0).unsqueeze(0),
                scale_factor=self.scale_factor,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        )

    def _apply_mask(self, x, mask):
        """
        x:      Tensor of shape (B, C, L)
        mask:   Tensor of shape (n, ) indicating visible indices.

        Returns:
            x with the mask applied, where L positions not in `mask` are zeroed out.
        """
        up_mask = self._upsample_mask(mask)  # Shape (L,)

        # Expand mask to match x's shape (B, C, L)
        up_mask = up_mask.view(1, 1, -1)  # Shape (1, 1, L)

        return x * up_mask  # Element-wise multiplication

    def forward(self, x, mask=None):
        """
        Forward pass of ConvMerge.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, out_channels).
        """
        # Convert from (B, L, C) to (B, C, L)
        out = x.permute(0, 2, 1).contiguous()

        # Apply mask
        if mask is not None:
            out = self._apply_mask(out, mask)

        # Apply skip connection
        residual = self.skip_connection(out)
        out = self.input_layer(out)
        out = residual + out  # Residual connection

        # Apply second convolutional block
        residual = out
        out = self.final_layer(out)
        out = residual + out  # Residual connection

        # Convert to (B, L, C) always
        out = out.permute(0, 2, 1).contiguous()

        return out


class ECGStageModule(nn.Module):
    """
    ECG Stage Module using Swin Transformer Blocks.

    This module applies a sequence of ECG-specific Swin Transformer blocks and an optional
    patch merging step for hierarchical feature extraction.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        layers (int): Number of Swin Transformer blocks (must be even for regular/shifted pairing).
        merge (bool): Whether to apply patch merging.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        use_rpe (bool): Whether to use relative positional encoding (RPE).
        rpe_type (str): Type of RPE to use ("cope", "default", "combined").
        stage (int): Current stage of the model (used for absolute position embeddings).
        merge_config (dict): Configuration for the patch merging module.
        cosine_sim (bool, optional): Whether to use cosine similarity for attention computation. Defaults to True.
        compute_attn_ent (bool): Whether to compute the attention entropy.
        shift (bool, optional): Whether to apply cyclic shift for shifted window attention. Defaults to True.
        shift_size (int, optional): Cyclic shift size. Defaults to 'window_size//2'.
        use_ape (bool, optional): Whether to use absolute positional embeddings (APE). Defaults to True.
        merged_seq_len (int, optional): Length of the ECG signal after merging. Defaults to 160.
    """

    def __init__(
        self,
        in_c,
        out_c,
        layers,
        num_heads,
        window_size,
        use_rpe,
        rpe_type,
        stage,
        merge_config,
        cosine_sim=True,
        compute_attn_ent=False,
        shift=True,
        shift_size=None,
        use_ape=True,
        merged_seq_len=160,
    ):
        super().__init__()
        assert (
            layers % 2 == 0
        ), "Stage layers need to be divisible by 2 for regular and shifted blocks."

        self.use_ape = use_ape
        self.rpe_type = rpe_type
        self.stage = stage
        self.window_size = window_size
        self.merged_seq_len = merged_seq_len

        # Offset for mask
        self.n_windows = self.merged_seq_len // self.window_size
        self.mask_offset = torch.tensor(
            [(i * self.window_size) for i in range(self.n_windows)]
        )

        # Patch merging for ECG signals
        self.patch_merge = ConvMerge(
            in_c=in_c,
            out_c=out_c,
            seq_len=merged_seq_len,
            config=merge_config,
        )

        # Absolute positional embeddings (APE) for later stages
        if self.use_ape:
            self.abs_pos_emb = get_1d_sincos_pos_embed(out_c, merged_seq_len)

        # Swin Transformer Blocks for ECG signals
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(
                nn.ModuleList(
                    [
                        ECGSwinBlock(
                            dim=out_c,
                            heads=num_heads,
                            mlp_dim=out_c * 4,
                            cosine_sim=cosine_sim,
                            compute_attn_ent=compute_attn_ent,
                            shift=False,  # Regular window-based attention
                            shift_size=shift_size,
                            window_size=window_size,
                            use_rpe=use_rpe,
                            rpe_type=rpe_type,
                        ),
                        ECGSwinBlock(
                            dim=out_c,
                            heads=num_heads,
                            mlp_dim=out_c * 4,
                            cosine_sim=cosine_sim,
                            compute_attn_ent=compute_attn_ent,
                            shift=shift,  # Shifted window-based attention
                            shift_size=shift_size,
                            window_size=window_size,
                            use_rpe=use_rpe,
                            rpe_type=rpe_type,
                        ),
                    ]
                )
            )

    def _apply_mask(self, x, mask):
        l_mask = len(mask)
        expanded_mask = torch.cat(
            [mask] * self.n_windows
        ) + self.mask_offset.repeat_interleave(l_mask).to(
            mask.device
        )  # Expand mask for all windows and sum offset
        x = x[:, expanded_mask, :]  # Apply mask

        return x

    def forward(self, x, mask=None):
        """
        Forward pass for the ECG stage module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).
            mask (torch.Tensor, optional): Mask for attention (default: None).

        Returns:
            torch.Tensor: Processed tensor after patch merging (if applicable) and Swin Transformer blocks.
        """
        # Patch Merge
        out = self.patch_merge(x, mask)

        # Add absolute positional embeddings if enabled and in later stages
        if self.use_ape:
            out += self.abs_pos_emb

        # Apply mask
        if mask is not None:
            out = self._apply_mask(out, mask)

        # Apply all Swin Transformer blocks in sequence
        for block1, block2 in self.layers:
            out = block1(out, mask)
            out = block2(out, mask)

        return out


class ECGHiTNeXt(nn.Module):
    """
    Hierarchical Transformer Network for ECG Processing (ECGHiTNeXt).

    This model processes ECG signals using a hierarchical transformer architecture with multiple stages.
    Each stage applies a Swin Transformer-based ECGStageModule, optionally using absolute and relative
    positional embeddings, and progressively reduces the sequence length.

    Args:
        ecg_size (tuple): Shape of the input ECG data (seq_len, channels).
        hidden_dim (list): List of hidden dimensions for each stage.
        layers (list): Number of Swin Transformer layers in each stage.
        heads (list): Number of attention heads in each stage.
        window_size (int): Size of the attention window.
        merge_config (dict): Configuration for the patch merging modules.
        apply_output_head (bool): Whether to apply the output head.
        output_head_hidden_dim (int): Hidden dimension of the output classification head.
        output_head_out_dim (int): Output dimension (e.g., number of classes) of the classification head.
        output_head_dropout (float, optional): Dropout rate in the output head. Defaults to 0.7.
        output_head_linear_only (bool, optional): Use a single linear layer in the output head if True. Defaults to False.
        output_head_norm (bool, optional): Apply LayerNorm before the output head if True. Defaults to True.
        sem_config (list[dict], optional): Config for simplicial embeddings normalization for each stage. Defaults to None.
        fwd_mode (str, optional): Forward function to use ("default", "staged"). Defaults to "default".
        use_rpe (bool, optional): Whether to use relative positional encoding (RPE). Defaults to True.
        rpe_type (str, optional): Type of RPE to use ("cope", "default", "combined"). Defaults to "cope".
        use_ape (bool, optional): Whether to use absolute positional encoding (APE). Defaults to True.
        num_stages (int, optional): Number of hierarchical stages. Defaults to 4.
        cosine_sim (bool, optional): Whether to use cosine similarity for attention computation. Defaults to False.
        compute_attn_ent (bool): Whether to compute the attention entropy.
        propagate_stage_losses (bool, optional): Whether to propagate losses at each stage for JEPA training. Defaults to False.
        shift (bool, optional): Whether to apply cyclic shift for shifted window attention. Defaults to True.
        shift_size (int, optional): Cyclic shift size. Defaults to 'window_size // 2' if None.
    """

    def __init__(
        self,
        ecg_size,
        hidden_dim,
        layers,
        heads,
        window_size,
        merge_config,
        apply_output_head,
        output_head_hidden_dim,
        output_head_out_dim,
        output_head_dropout=0.7,
        output_head_linear_only=False,
        output_head_norm=True,
        sem_config=None,
        fwd_mode="default",
        use_rpe=True,
        rpe_type="default",
        use_ape=True,
        num_stages=4,
        cosine_sim=False,
        compute_attn_ent=False,
        propagate_stage_losses=False,
        shift=True,
        shift_size=None,
    ):
        super().__init__()

        assert (
            len(layers) == num_stages
        ), "Length of 'layers' should be equal to 'num_stages'."
        assert (
            len(heads) == num_stages
        ), "Length of 'heads' should be equal to 'num_stages'."

        self.propagate_stage_losses = propagate_stage_losses
        self.num_stages = num_stages
        self.apply_output_head = apply_output_head

        self.fwd_mode = None
        self.fwd_f = None
        self.set_fwd_mode(fwd_mode)

        # Define the hierarchical stages
        reduction_factor = 1
        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            reduction_factor *= merge_config[i_stage]["reduction"]
            # Set input and output channel sizes for each stage
            if i_stage == 0:
                in_c = ecg_size[1]  # Input channels from raw ECG data
            else:
                in_c = hidden_dim[i_stage - 1]  # Output channels from previous stage

            out_c = hidden_dim[i_stage]

            # Define ECGStageModule for each stage
            stage = ECGStageModule(
                in_c=in_c,
                out_c=out_c,
                layers=layers[i_stage],
                num_heads=heads[i_stage],
                window_size=window_size,
                use_rpe=use_rpe,
                rpe_type=rpe_type,
                stage=i_stage,
                cosine_sim=cosine_sim,
                compute_attn_ent=compute_attn_ent,
                use_ape=use_ape,
                merged_seq_len=(ecg_size[0] // reduction_factor),
                merge_config=merge_config[i_stage],
                shift=shift,
                shift_size=shift_size,
            )
            self.stages.append(stage)

        if self.apply_output_head:
            self.output_head = OutputHead(
                embedding_dim=hidden_dim[-1],
                hidden_dim=output_head_hidden_dim,
                out_dim=output_head_out_dim,
                dropout=output_head_dropout,
                linear_only=output_head_linear_only,
                norm=output_head_norm,
            )

        if sem_config is not None:
            assert (
                len(sem_config) == num_stages
            ), "Length of 'sem_config' should be equal to 'num_stages'."

            self.use_sem = [False if config is None else True for config in sem_config]
            self.sem = [
                None if config is None else SEM(**config) for config in sem_config
            ]

        else:
            self.use_sem = [False] * num_stages

        self.apply(self._init_weights)

    def set_fwd_mode(self, fwd_mode):
        if fwd_mode in ["default", "staged"]:
            self.fwd_mode = fwd_mode

            if fwd_mode == "default":
                self.fwd_f = self._default_forward

            else:
                self.fwd_f = self._staged_forward

        else:
            raise ValueError(
                f"'{fwd_mode}' is not a valid value for 'fwd_mode' ('default', 'staged')."
            )

    def _default_forward(self, x, mask=None):
        """
        Standard forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).
            mask (Any, optional): Unused. Included only for compatibility with other forward methods.

        Returns:
            torch.Tensor: Output from the last stage.
        """
        curr_output = x
        for i in range(self.num_stages):
            curr_output = self.stages[i](
                curr_output, None
            )  # Process input through each stage

        if self.use_sem[-1]:
            curr_output = self.sem[-1](curr_output)

        if self.apply_output_head:
            curr_output = self.output_head(curr_output)

        return curr_output

    def _staged_forward(self, x, mask):
        """
        Forward pass with optional mask and returning outputs per stage.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).
            mask (list of torch.Tensor): List of masks for each stage.

        Returns:
            list of torch.Tensor: Outputs from each stage.
        """

        curr_output, stage_outputs = x, []
        for i in range(len(self.stages)):
            if self.propagate_stage_losses:
                stage_input = curr_output.clone()
            else:
                stage_input = curr_output.clone().detach().requires_grad_(True)

            stage_output = self.stages[i](stage_input, mask[i])
            if self.use_sem[i]:
                stage_output = self.sem[i](stage_output)
            stage_outputs.append(stage_output)

            curr_output = self.stages[i](curr_output, None)

        if self.use_sem[-1]:
            curr_output = self.sem[-1](curr_output)

        if self.apply_output_head:
            curr_output = self.output_head(curr_output)

        return curr_output, stage_outputs

    def forward(self, x, mask=None):
        """
        Chosen method for forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim).
            mask (list of torch.Tensor, optional): List of masks for training. Defaults to None.

        Returns:
            Output of the chosen method.
        """
        return self.fwd_f(x, mask)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ────────────────────────────────────────────────────────────────────────
    # 1.  Cache the attention blocks once
    # ────────────────────────────────────────────────────────────────────────
    def _attach_attn_index(self):
        if hasattr(self, "_attn_blocks"):
            return  # already done

        blocks = []
        for s, stage in enumerate(self.stages):
            for l, pair in enumerate(stage.layers):
                for b, block in enumerate(pair):
                    k = f"s{s}_l{l}_b{b}"
                    a = block.attention
                    if not a.compute_attn_ent:
                        raise ValueError(f"{k} built with compute_attn_ent=False")
                    blocks.append((k, a))
        self._attn_blocks = blocks

    # ----------------------------------------------------------------------
    # 2.  Stats for ONE batch (mean & central moments over *batch only*)
    # ----------------------------------------------------------------------
    def attn_stats_batch(
        self,
        batch,
        max_order: int = 5,
        detach: bool = False,
    ):
        assert max_order >= 2
        if not hasattr(self, "_attn_blocks"):
            self._attach_attn_index()

        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        _ = self(batch)

        stats = {}
        for key, attn in self._attn_blocks:
            ent = attn.curr_attn_ent  # (B,H,W,L)
            attn.curr_attn_ent = None
            if detach:
                ent = ent.detach()

            B = ent.shape[0]
            mu = ent.mean(dim=0)  # (H,W,L)  <-- batch only
            centred = ent - mu[None]  # broadcast over B

            entry = {"count": B, "mean": mu}
            for k in range(2, max_order + 1):
                entry[f"mom{k}"] = centred.pow(k).mean(dim=0)  # (H,W,L)
            stats[key] = entry
        return stats

    # ------------------------------------------------------------------
    # 3.  Stats for an entire loader  (weighted avg. of batch stats)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def attn_stats_loader(
        self,
        loader,
        max_order: int = 5,
        device=None,
    ):
        assert max_order >= 2
        if not hasattr(self, "_attn_blocks"):
            self._attach_attn_index()

        tot_cnt = {k: 0 for k, _ in self._attn_blocks}
        tot_mean = {k: None for k in tot_cnt}
        tot_mom = {k: {j: None for j in range(2, max_order + 1)} for k in tot_cnt}

        # tqdm wrapper --------------------------------------------------
        with tqdm(
            total=len(loader), desc="Collecting attention statistics for ECGHiTNeXt"
        ) as bar:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                if device is not None:
                    batch = batch.to(device, non_blocking=True)

                b = self.attn_stats_batch(batch, max_order=max_order, detach=True)

                for key in tot_cnt:
                    c = b[key]["count"]
                    m = b[key]["mean"]
                    if tot_mean[key] is None:
                        tot_mean[key] = m * c
                    else:
                        tot_mean[key] += m * c
                    for j in range(2, max_order + 1):
                        mm = b[key][f"mom{j}"]
                        if tot_mom[key][j] is None:
                            tot_mom[key][j] = mm * c
                        else:
                            tot_mom[key][j] += mm * c
                    tot_cnt[key] += c

                bar.update(1)  # ← advance progress bar

        # normalise -----------------------------------------------------
        out = {}
        for key in tot_cnt:
            N = tot_cnt[key]
            mu = tot_mean[key] / N
            entry = {"count": N, "mean": mu}
            for j in range(2, max_order + 1):
                entry[f"mom{j}"] = tot_mom[key][j] / N
            out[key] = entry
        return out


if __name__ == "__main__":
    import h5py
    import numpy as np
    import torch

    # --------------------------------------------------------------
    # 1.  Load 128 ECGs straight from the HDF5 file
    # --------------------------------------------------------------
    h5_path = "/home/ubuntu/DATASETS/code15.h5"  # ← change
    split = "train"  # "train", "val", or "test"
    batch_size = 8
    seq_len = 2560  # model expects 2560 samples
    device = "cpu"  # or "cuda"

    with h5py.File(h5_path, "r") as f:
        tracings_ds = f[f"{split}/tracings"]  # (N, 4096, 12)
        num_records = tracings_ds.shape[0]

        idx_random = np.random.choice(num_records, batch_size, replace=False)
        idx_sorted = np.sort(idx_random)  #  ← make them increasing

        # read in sorted order (h5py requirement)
        tracings_sorted = tracings_ds[idx_sorted]  # (128, 4096, 12)

    # If you don’t care about order, just keep tracings_sorted.
    # If you want the original random order, shuffle back:
    reorder = np.argsort(idx_random)  # positions to restore
    tracings = tracings_sorted[reorder]  # now matches idx_random

    # centre‑crop and convert as before
    start = (tracings.shape[1] - seq_len) // 2
    tracings = tracings[:, start : start + seq_len].astype(np.float32)
    batch = torch.from_numpy(tracings).to(device)

    model = ECGHiTNeXt(
        ecg_size=(seq_len, 12),
        hidden_dim=[96, 192, 384, 768],
        layers=[4, 4, 12, 4],
        heads=[4, 8, 16, 32],
        window_size=10,
        output_head_hidden_dim=512,
        output_head_out_dim=6,
        use_rpe=True,
        use_ape=False,
        num_stages=4,
        rpe_type="combined",
        cosine_sim=False,
        propagate_stage_losses=False,
        shift=False,
        merge_config=[
            {
                "conv_same": {"kernel_size": 1, "stride": 1, "padding": 0},
                "conv_reduc": {"kernel_size": 10, "stride": 4, "padding": 4},
                "max_pool_sc": {"kernel_size": 4, "stride": 4, "padding": 0},
                "conv_sc": {"kernel_size": 1, "stride": 1, "padding": 0},
                "dropout_rate": 0.5,
                "reduction": 4,
            }
        ]
        * 4,
        shift_size=2,
        apply_output_head=True,
    ).to(device)

    # --------------------------------------------------------------
    # 3.  Forward passes
    # --------------------------------------------------------------
    print("fwd w/o mask")
    out = model(batch)
    print(out.shape)
    print()

    model.set_fwd_mode("staged")

    print("staged fwd w/o mask")
    mask = [None] * 4
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    print("staged fwd w/ full mask")
    mask = [torch.tensor([0, 2, 4, 6, 8], device=device)] * 4
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    print("staged fwd w/ partial mask position 1")
    mask = [torch.tensor([0, 2, 4, 6, 8], device=device), None, None, None]
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    print("staged fwd w/ partial mask position 2")
    mask = [None, torch.tensor([0, 2, 4, 6, 8], device=device), None, None]
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    print("staged fwd w/ partial mask position 3")
    mask = [None, None, torch.tensor([0, 2, 4, 6, 8], device=device), None]
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    print("staged fwd w/ partial mask position 4")
    mask = [None, None, None, torch.tensor([0, 2, 4, 6, 8], device=device)]
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    print("staged fwd w/ partial mask position 1, 4")
    mask = [
        torch.tensor([0, 2, 4, 6, 8], device=device),
        None,
        None,
        torch.tensor([0, 2, 4, 6, 8], device=device),
    ]
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    print("staged fwd w/ partial mask position 2, 3")
    mask = [
        None,
        torch.tensor([0, 2, 4, 6, 8], device=device),
        torch.tensor([0, 2, 4, 6, 8], device=device),
        None,
    ]
    out, stage_out = model(batch, mask)
    for stage in stage_out:
        print(stage.shape)
    print(out.shape)
    print()

    # --------------------------------------------------------------
    # 2.  Build model  • attention entropy ON
    # --------------------------------------------------------------
    model = ECGHiTNeXt(
        ecg_size=(seq_len, 12),
        hidden_dim=[96, 192, 384, 768],
        layers=[4, 4, 12, 4],
        heads=[4, 8, 16, 32],
        window_size=10,
        output_head_hidden_dim=512,
        output_head_out_dim=6,
        use_rpe=True,
        use_ape=False,
        num_stages=4,
        rpe_type="combined",
        cosine_sim=False,
        propagate_stage_losses=False,
        shift=False,
        merge_config=[
            {  # repeated 4×
                "conv_same": {"kernel_size": 1, "stride": 1, "padding": 0},
                "conv_reduc": {"kernel_size": 10, "stride": 4, "padding": 4},
                "max_pool_sc": {"kernel_size": 4, "stride": 4, "padding": 0},
                "conv_sc": {"kernel_size": 1, "stride": 1, "padding": 0},
                "dropout_rate": 0.5,
                "reduction": 4,
            }
        ]
        * 4,
        shift_size=2,
        compute_attn_ent=True,  # ← NEW
        apply_output_head=True,
    ).to(device)

    # create and cache the flat list of attention blocks once
    model._attach_attn_index()

    # --------------------------------------------------------------
    # 3-A.  Stats for ONE arbitrary batch --------------------------
    # --------------------------------------------------------------
    batch_stats = model.attn_stats_batch(
        batch, max_order=4, detach=True  # mean + moments 2-4
    )  # no grads in this test

    print("\n► Batch-level stats:")
    ex_key = "s0_l0_b0"
    print(
        ex_key,
        {
            k: v.shape if torch.is_tensor(v) else v  # prettier printing
            for k, v in batch_stats[ex_key].items()
        },
    )

    ex_key = "s1_l0_b0"
    print(
        ex_key,
        {
            k: v.shape if torch.is_tensor(v) else v  # prettier printing
            for k, v in batch_stats[ex_key].items()
        },
    )

    ex_key = "s2_l0_b0"
    print(
        ex_key,
        {
            k: v.shape if torch.is_tensor(v) else v  # prettier printing
            for k, v in batch_stats[ex_key].items()
        },
    )

    ex_key = "s3_l0_b0"
    print(
        ex_key,
        {
            k: v.shape if torch.is_tensor(v) else v  # prettier printing
            for k, v in batch_stats[ex_key].items()
        },
    )

    # --------------------------------------------------------------
    # 3-B.  Stats over a DataLoader  (small demo loader) -----------
    # --------------------------------------------------------------
    import torch.utils.data as td

    # reuse the 128 traces we already loaded into RAM
    dataset = td.TensorDataset(torch.from_numpy(tracings))
    loader = td.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False
    )

    loader_stats = model.attn_stats_loader(loader, max_order=4, device=device)

    print("\n► Loader-level stats:")
    print("  blocks collected :", len(loader_stats))
    ex_key = "s0_l0_b0"
    if ex_key in loader_stats:
        moms = loader_stats[ex_key]
        print(
            f"  {ex_key} – mean shape {moms['mean'].shape}, "
            f"var shape {moms['mom2'].shape}"
        )
    else:
        print("  example key not found (different depth configuration).")

    ex_key = "s1_l0_b0"
    if ex_key in loader_stats:
        moms = loader_stats[ex_key]
        print(
            f"  {ex_key} – mean shape {moms['mean'].shape}, "
            f"var shape {moms['mom2'].shape}"
        )
    else:
        print("  example key not found (different depth configuration).")

    ex_key = "s2_l0_b0"
    if ex_key in loader_stats:
        moms = loader_stats[ex_key]
        print(
            f"  {ex_key} – mean shape {moms['mean'].shape}, "
            f"var shape {moms['mom2'].shape}"
        )
    else:
        print("  example key not found (different depth configuration).")

    ex_key = "s3_l0_b0"
    if ex_key in loader_stats:
        moms = loader_stats[ex_key]
        print(
            f"  {ex_key} – mean shape {moms['mean'].shape}, "
            f"var shape {moms['mom2'].shape}"
        )
    else:
        print("  example key not found (different depth configuration).")
