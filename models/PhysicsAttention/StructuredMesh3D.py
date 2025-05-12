"""
Physics-informed attention mechanism for 3D structured meshes.

This module implements a specialized attention mechanism optimized for
volumetric data arranged in regular 3D grids. It leverages 3D convolutions
for local feature extraction and a slicing-and-deslicing approach combined
with an ErwinTransformer for global feature interactions.
"""

import torch
import torch.nn as nn
from einops import rearrange

from ..components import ErwinTransformer


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    """Physics-informed attention for 3D structured mesh data.

    This attention mechanism processes 3D structured mesh data through three main steps:
    1. Slicing: Projects input features into a reduced set of slice tokens using 3D convolutions
    2. Transformation: Processes slice tokens using the ErwinTransformer
    3. De-slicing: Projects transformed slice tokens back to the original volumetric points

    The use of 3D convolutions allows efficient local feature extraction that
    respects the spatial structure of volumetric data.

    Attributes:
        dim_head (int): Dimension of each attention head
        heads (int): Number of attention heads
        softmax (nn.Softmax): Softmax operation for attention weights
        dropout (nn.Dropout): Dropout layer for regularization
        temperature (nn.Parameter): Learnable temperature parameter for slice weight scaling
        H (int): Height of the 3D mesh
        W (int): Width of the 3D mesh
        D (int): Depth of the 3D mesh
        dimensionality (int): Spatial dimensionality (3 for 3D meshes)
        in_project_x (nn.Conv3d): 3D convolution for query features
        in_project_fx (nn.Conv3d): 3D convolution for value features
        in_project_slice (nn.Linear): Linear projection for slice weights
        erwin (ErwinTransformer): Transformer for processing slice tokens
        to_out (nn.Sequential): Output projection
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=32,
        H=32,
        W=32,
        D=32,
        kernel=3,
    ):
        """Initialize the Physics_Attention_Structured_Mesh_3D module.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            slice_num (int): Number of slice tokens to use
            H (int): Height of the 3D mesh
            W (int): Width of the 3D mesh
            D (int): Depth of the 3D mesh
            kernel (int): Size of convolution kernel for local feature extraction
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D
        self.dimensionality = 3  # 3D space

        # 3D Convolutional layers for volumetric feature extraction
        # kernel_size=kernel, stride=1, padding=kernel//2 to maintain spatial dimensions
        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)

        # Linear projection for computing slice weights
        self.in_project_slice = nn.Linear(dim_head, slice_num)

        # Orthogonal initialization for better training stability
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        # Hierarchical transformer for processing sliced tokens
        # Specifically configured for 3D structured mesh data
        self.erwin = ErwinTransformer(
            c_in=dim_head,  # Input channel dimension matches head dimension
            c_hidden=[
                dim_head,  # First level maintains dimension
                dim_head * 2,  # Second level expands channels for higher expressivity
            ],
            ball_sizes=[
                min(32, slice_num),  # First level considers more neighbors
                min(16, slice_num // 2),  # Second level reduces neighborhood size
            ],
            enc_num_heads=[heads // 2, heads],  # More attention heads at deeper levels
            enc_depths=[2, 2],  # Same depth at each hierarchical level
            dec_num_heads=[
                heads // 2
            ],  # Decoder head count matches encoder at same level
            dec_depths=[2],  # Decoder depth matches encoder
            strides=[2],  # Coarsens the point cloud by factor of 2
            rotate=1,  # Enable rotation for better geometric awareness
            decode=True,  # Enable upsampling back to original resolution
            mlp_ratio=4,  # Standard expansion ratio in MLP blocks
            dimensionality=self.dimensionality,  # Dimensionality of the space (3D)
            mp_steps=0,  # No message passing steps needed
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """Forward pass of the physics-informed attention module for 3D structured meshes.

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, H*W*D, channels]

        Returns:
            torch.Tensor: Output feature tensor of shape [batch_size, H*W*D, channels]
        """
        # Extract batch size, number of points, and channels
        B, N, C = x.shape

        # Reshape from flattened representation to 3D structured grid
        # [B, N, C] -> [B, H, W, D, C] -> [B, C, H, W, D] (for Conv3D input)
        x = (
            x.reshape(B, self.H, self.W, self.D, C)
            .contiguous()
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

        ### (1) Slice operation: Project input features to a reduced set of tokens
        # Project value features using 3D convolution (captures local volumetric patterns)
        fx_mid = (
            self.in_project_fx(x)  # [B, inner_dim, H, W, D]
            .permute(0, 2, 3, 4, 1)  # [B, H, W, D, inner_dim]
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)  # [B, H*W*D, heads, dim_head]
            .permute(0, 2, 1, 3)  # [B, heads, H*W*D, dim_head]
            .contiguous()
        )

        # Project query features using 3D convolution
        x_mid = (
            self.in_project_x(x)  # [B, inner_dim, H, W, D]
            .permute(0, 2, 3, 4, 1)  # [B, H, W, D, inner_dim]
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)  # [B, H*W*D, heads, dim_head]
            .permute(0, 2, 1, 3)  # [B, heads, H*W*D, dim_head]
            .contiguous()
        )

        # Compute slicing weights with temperature-scaled softmax
        # Temperature is clamped for stability during training
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )  # [B, heads, H*W*D, slice_num]

        # Compute normalization factor for each slice token
        slice_norm = slice_weights.sum(2)  # [B, heads, slice_num]

        # Aggregate features into slice tokens using weighted sum
        slice_token = torch.einsum(
            "bhnc,bhng->bhgc", fx_mid, slice_weights
        )  # [B, heads, slice_num, dim_head]

        # Normalize slice tokens by the sum of weights to maintain scale
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Transform slice tokens with ErwinTransformer
        # Extract dimensions of slice token tensor
        B, H, G, C = slice_token.shape

        # Reshape slice tokens for ErwinTransformer input format
        # ErwinTransformer expects a flattened representation [num_points, channels]
        slice_token_flat = slice_token.reshape(B * H * G, C)

        # Create artificial positions for slice tokens in 3D space
        # This is needed because ErwinTransformer is position-aware
        # Using random positions in 3D space as slice tokens don't have inherent positions
        pos = torch.rand(B * H * G, self.dimensionality, device=slice_token.device)

        # Create batch indices to separate different batch and head combinations
        # Each batch-head pair is treated as a separate point cloud by the transformer
        batch_idx = torch.arange(B * H, device=slice_token.device).repeat_interleave(G)

        # Process through ErwinTransformer to allow interactions between slice tokens
        # This enables global information exchange and feature refinement across the volume
        processed_tokens = self.erwin(slice_token_flat, pos, batch_idx)

        # Reshape the processed tokens back to the multi-head format
        out_slice_token = processed_tokens.reshape(B, H, G, C)

        ### (3) Deslice operation: Project processed tokens back to original volumetric points
        # Distribute processed slice token information back to the original 3D grid
        # Using the same slice weights ensures information flows properly between
        # the slicing and de-slicing operations
        out_x = torch.einsum(
            "bhgc,bhng->bhnc", out_slice_token, slice_weights
        )  # [B, heads, H*W*D, dim_head]

        # Rearrange from multi-head format to batch-first format
        # Concatenating all heads into a single feature dimension
        out_x = rearrange(out_x, "b h n d -> b n (h d)")  # [B, H*W*D, heads*dim_head]

        # Project concatenated features back to the original dimension
        return self.to_out(out_x)  # [B, H*W*D, dim]
