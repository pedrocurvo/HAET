"""
Physics-informed attention mechanism for 2D structured meshes.

This module implements a specialized attention mechanism optimized for
data arranged in regular 2D grids. It leverages 2D convolutions for
local feature extraction and a slicing-and-deslicing approach combined
with an ErwinTransformer for global feature interactions.
"""

import torch
import torch.nn as nn
from einops import rearrange

from ..components import ErwinFlashTransformer as ErwinTransformer


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    """Physics-informed attention for 2D structured mesh data with Transolver++.

    This attention mechanism processes 2D structured mesh data through three main steps:
    1. Slicing: Projects input features into a reduced set of eidetic states using 2D convolutions and Rep-Slice
    2. Transformation: Processes eidetic states using the ErwinTransformer
    3. De-slicing: Projects transformed eidetic states back to the original mesh points

    The use of 2D convolutions allows efficient local feature extraction that
    respects the spatial structure of the data, while Transolver++ with adaptive
    temperature and eidetic states enhances memory efficiency and performance.

    Attributes:
        dim_head (int): Dimension of each attention head
        heads (int): Number of attention heads
        H (int): Height of the 2D mesh
        W (int): Width of the 2D mesh
        dimensionality (int): Spatial dimensionality (2 for 2D meshes)
        epsilon (float): Small constant for Rep-Slice computation
        base_temp (float): Base temperature for adaptive temperature scaling
        in_project_x (nn.Conv2d): 2D convolution for input features
        in_project_slice (nn.Linear): Linear projection for slice weights
        ada_temp_linear (nn.Linear): Linear projection for adaptive temperature adjustment
        erwin (ErwinTransformer): Transformer for processing eidetic states
        to_out (nn.Sequential): Output projection
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        H=101,
        W=31,
        kernel=3,
        base_temp=0.5,
        epsilon=1e-6,
        # ErwinTransformer parameters
        c_hidden=None,
        ball_sizes=None,
        enc_num_heads=None,
        enc_depths=None,
        dec_num_heads=None,
        dec_depths=None,
        strides=None,
        rotate=1,
        decode=True,
        mlp_ratio=4,
        mp_steps=0,
        embed=False
    ):
        """Initialize the Physics_Attention_Structured_Mesh_2D module with Transolver++.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            slice_num (int): Number of slice tokens to use
            H (int): Height of the 2D mesh
            W (int): Width of the 2D mesh
            kernel (int): Size of convolution kernel for local feature extraction
            base_temp (float): Base temperature for adaptive temperature scaling
            epsilon (float): Small constant for the log(-log(ε)) term in Rep-Slice
            c_hidden (list): Hidden channel dimensions for each hierarchical level
            ball_sizes (list): Ball sizes for each hierarchical level
            enc_num_heads (list): Number of attention heads for each encoder level
            enc_depths (list): Depth of each encoder level
            dec_num_heads (list): Number of attention heads for each decoder level
            dec_depths (list): Depth of each decoder level
            strides (list): Stride values for each level
            rotate (int): Rotate flag for geometric awareness
            decode (bool): Whether to decode/upsample back to original resolution
            mlp_ratio (int): Expansion ratio in MLP blocks
            mp_steps (int): Number of message passing steps
            embed (bool): Whether to use ErwinEmbedding (True) or direct projection (False)
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.H = H
        self.W = W
        self.dimensionality = 2  # 2D space
        self.epsilon = epsilon
        self.base_temp = base_temp

        # For Transolver++, we only need one projection to save memory
        # 2D Convolutional layer for spatial feature extraction
        # kernel_size=kernel, stride=1, padding=kernel//2 to maintain spatial dimensions
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)

        # Linear projection for computing slice weights
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        
        # Ada-Temp: Base temperature + adaptive component
        self.ada_temp_linear = nn.Linear(dim_head, 1)  # Adaptive temperature adjustment

        # Orthogonal initialization for better training stability
        torch.nn.init.orthogonal_(self.in_project_slice.weight)  # use a principled initialization

        # Set default ErwinTransformer parameters if not provided
        if c_hidden is None:
            c_hidden = [dim_head, dim_head * 2]
        if ball_sizes is None:
            ball_sizes = [min(32, slice_num), min(16, slice_num // 2)]
        if enc_num_heads is None:
            enc_num_heads = [heads // 2, heads]
        if enc_depths is None:
            enc_depths = [2, 2]
        if dec_num_heads is None:
            dec_num_heads = [heads // 2]
        if dec_depths is None:
            dec_depths = [2]
        if strides is None:
            strides = [2]

        # Hierarchical transformer for processing sliced tokens
        # Specifically configured for 2D structured mesh data
        self.erwin = ErwinTransformer(
            c_in=dim_head,          # Input channel dimension matches head dimension
            c_hidden=c_hidden,      # Hidden channel dimensions for each level
            ball_sizes=ball_sizes,  # Ball sizes for each level
            enc_num_heads=enc_num_heads,  # Attention heads for each encoder level
            enc_depths=enc_depths,  # Depth of each encoder level
            dec_num_heads=dec_num_heads,  # Attention heads for each decoder level
            dec_depths=dec_depths,  # Depth of each decoder level
            strides=strides,        # Stride values for each level
            rotate=rotate,          # Enable rotation for better geometric awareness
            decode=decode,          # Enable upsampling back to original resolution
            mlp_ratio=mlp_ratio,    # Standard expansion ratio in MLP blocks
            dimensionality=self.dimensionality,  # Dimensionality of the space (2D)
            mp_steps=mp_steps,      # Number of message passing steps
            embed=embed,            # Use parameter value for ErwinEmbedding
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """Forward pass of the physics-informed attention module for 2D structured meshes with Transolver++.
        
        Implements Transolver++ Algorithm 1: Parallel Physics-Attention with Eidetic States

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, H*W, channels]

        Returns:
            torch.Tensor: Output feature tensor of shape [batch_size, H*W, channels]
        """
        # Extract batch size, number of points, and channels
        B, N, C = x.shape

        # Reshape from flattened representation to 2D structured grid
        # [B, N, C] -> [B, H, W, C] -> [B, C, H, W] (for Conv2D input)
        x = (
            x.reshape(B, self.H, self.W, C)
            .contiguous()
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        ### (1) Slice operation: Project input features to a reduced set of tokens
        # Project features using 2D convolution - drop fx to save 50% memory as per algorithm
        x_proj = (
            self.in_project_x(x)  # [B, inner_dim, H, W]
            .permute(0, 2, 3, 1)  # [B, H, W, inner_dim]
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)  # [B, H*W, heads, dim_head]
            .permute(0, 2, 1, 3)  # [B, heads, H*W, dim_head]
            .contiguous()
        )

        # Compute adaptive temperature (Ada-Temp): τ = τ0 + Linear(xi)
        # Implementation: τ(k) ←τ0 + Ada-Temp(x(k))
        adaptive_temp = self.base_temp + self.ada_temp_linear(x_proj).clamp(min=-0.4, max=0.4)
        
        # Compute Rep-Slice: Softmax(Linear(x) - log(-log(ε))) / τ
        # Implementation: w(k) ← Rep-Slice(x(k),τ(k))
        log_neg_log_epsilon = torch.log(-torch.log(torch.tensor(self.epsilon, device=x.device)))
        slice_logits = self.in_project_slice(x_proj) - log_neg_log_epsilon
        slice_weights = torch.softmax(slice_logits / adaptive_temp, dim=2)  # [B, heads, H*W, slice_num]

        # Compute weights norm: w(k)_norm ← sum_i(w(k)_i)
        slice_norm = slice_weights.sum(2)  # [B, heads, slice_num]

        # Compute eidetic states: s(k) ← w(k)T x(k) / w_norm
        # We use x_proj both as x and f to save memory
        eidetic_states = torch.einsum(
            "bhnc,bhng->bhgc", x_proj, slice_weights
        )  # [B, heads, slice_num, dim_head]

        # Normalize eidetic states by the sum of weights to maintain scale
        eidetic_states = eidetic_states / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Transform eidetic states with ErwinTransformer
        # Extract dimensions of eidetic states tensor
        B, H, G, C = eidetic_states.shape

        # Reshape eidetic states for ErwinTransformer input format
        # ErwinTransformer expects a flattened representation [num_points, channels]
        eidetic_states_flat = eidetic_states.reshape(B * H * G, C)

        # Use center of mass positions for eidetic states instead of random positions
        # Since eidetic states are already weighted averages (center of mass) of the input features,
        # we compute their spatial representation in the unit cube based on their relative positions
        # in the feature space, normalized across each batch and head
        
        # Compute center of mass positions by normalizing features to unit cube
        feat_min = eidetic_states_flat.min(dim=0, keepdim=True)[0]
        feat_max = eidetic_states_flat.max(dim=0, keepdim=True)[0]
        feat_range = feat_max - feat_min + 1e-8  # Add epsilon to avoid division by zero
        
        # Use first dimensionality components as spatial positions, normalized to [0,1]
        pos = (eidetic_states_flat[:, :self.dimensionality] - feat_min[:, :self.dimensionality]) / feat_range[:, :self.dimensionality]

        # Create batch indices to separate different batch and head combinations
        # Each batch-head pair is treated as a separate point cloud by the transformer
        batch_idx = torch.arange(B * H, device=eidetic_states.device).repeat_interleave(G)

        # Process through ErwinTransformer to allow interactions between eidetic states
        # This enables global information exchange and feature refinement
        # This corresponds to: Update eidetic states s′← Attention(s)
        processed_states = self.erwin(eidetic_states_flat, pos, batch_idx)

        # Reshape the processed states back to the multi-head format
        out_eidetic_states = processed_states.reshape(B, H, G, C)

        ### (3) Deslice operation: Project processed states back to original grid points
        # Distribute processed eidetic states information back to the original 2D grid
        # Using the same slice weights ensures information flows properly between
        # the slicing and de-slicing operations
        # Implementation: x′(k) ← Deslice(s′, w(k))
        out_x = torch.einsum(
            "bhgc,bhng->bhnc", out_eidetic_states, slice_weights
        )  # [B, heads, H*W, dim_head]

        # Rearrange from multi-head format to batch-first format
        # Concatenating all heads into a single feature dimension
        out_x = rearrange(out_x, "b h n d -> b n (h d)")  # [B, H*W, heads*dim_head]

        # Project concatenated features back to the original dimension
        return self.to_out(out_x)  # [B, H*W, dim]
