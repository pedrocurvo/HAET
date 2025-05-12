"""
Physics-informed attention mechanism for irregular meshes.

This module implements a specialized attention mechanism that can process
point data in irregular meshes using physics-informed principles. It supports
data in 1D, 2D or 3D space through a slicing-and-deslicing approach combined
with an ErwinTransformer for enhanced feature interactions.
"""

import torch
import torch.nn as nn
from einops import rearrange

from ..components import ErwinTransformer


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics-informed attention for irregular mesh data.

    This attention mechanism processes irregular mesh data through three main steps:
    1. Slicing: Projects input features into a reduced set of slice tokens
    2. Transformation: Processes slice tokens using the ErwinTransformer
    3. De-slicing: Projects transformed slice tokens back to the original points

    This approach allows efficient processing of irregular point clouds while
    maintaining awareness of spatial relationships.

    Attributes:
        dim_head (int): Dimension of each attention head
        heads (int): Number of attention heads
        softmax (nn.Softmax): Softmax operation for attention weights
        dropout (nn.Dropout): Dropout layer for regularization
        temperature (nn.Parameter): Learnable temperature parameter for slice weight scaling
        dimensionality (int): Spatial dimensionality (3 for irregular meshes)
        in_project_x (nn.Linear): Linear projection for query features
        in_project_fx (nn.Linear): Linear projection for value features
        in_project_slice (nn.Linear): Linear projection for slice weights
        erwin (ErwinTransformer): Transformer for processing slice tokens
        to_out (nn.Sequential): Output projection
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        """Initialize the Physics_Attention_Irregular_Mesh module.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            slice_num (int): Number of slice tokens to use
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.dimensionality = 3  # Spatial dimensionality for the irregular mesh

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        # Hierarchical transformer for processing sliced tokens
        # This design uses a multi-level approach to capture features at different scales
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
            dimensionality=self.dimensionality,  # Dimensionality of the space (3D for irregular mesh)
            mp_steps=0,  # No message passing steps needed
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """Forward pass of the physics-informed attention module.

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, num_points, channels]

        Returns:
            torch.Tensor: Output feature tensor of shape [batch_size, num_points, channels]
        """
        # Extract batch size, number of points, and channels
        B, N, C = x.shape

        ### (1) Slice operation: Project input features to a reduced set of tokens
        # Project value features (for content representation)
        fx_mid = (
            self.in_project_fx(x)  # Linear projection [B, N, inner_dim]
            .reshape(
                B, N, self.heads, self.dim_head
            )  # Reshape for multi-head [B, N, H, C]
            .permute(0, 2, 1, 3)  # Reorder to [B, H, N, C]
            .contiguous()
        )

        # Project query features (for slice weight computation)
        x_mid = (
            self.in_project_x(x)  # Linear projection [B, N, inner_dim]
            .reshape(
                B, N, self.heads, self.dim_head
            )  # Reshape for multi-head [B, N, H, C]
            .permute(0, 2, 1, 3)  # Reorder to [B, H, N, C]
            .contiguous()
        )

        # Compute slicing weights with temperature-scaled softmax
        slice_weights = self.softmax(
            self.in_project_slice(x_mid)
            / self.temperature  # Project and scale [B, H, N, G]
        )

        # Compute normalization factor for each slice token
        slice_norm = slice_weights.sum(2)  # [B, H, G]

        # Aggregate features into slice tokens using weighted sum
        slice_token = torch.einsum(
            "bhnc,bhng->bhgc", fx_mid, slice_weights
        )  # [B, H, G, C]

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
        # Using random positions as slice tokens don't have inherent positions
        pos = torch.rand(B * H * G, self.dimensionality, device=slice_token.device)

        # Create batch indices to separate different batch and head combinations
        # Each batch-head pair is treated as a separate point cloud by the transformer
        batch_idx = torch.arange(B * H, device=slice_token.device).repeat_interleave(G)

        # Process through ErwinTransformer to allow interactions between slice tokens
        # This enables global information exchange and feature refinement
        processed_tokens = self.erwin(slice_token_flat, pos, batch_idx)

        # Reshape the processed tokens back to the multi-head format
        out_slice_token = processed_tokens.reshape(B, H, G, C)

        ### (3) Deslice operation: Project processed tokens back to original points
        # Distribute processed slice token information back to original points
        # Using the same slice weights ensures information flows properly between
        # the slicing and de-slicing operations
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)

        # Rearrange from multi-head format to batch-first format
        # Concatenating all heads into a single feature dimension
        out_x = rearrange(out_x, "b h n d -> b n (h d)")

        # Project concatenated features back to the original dimension
        return self.to_out(out_x)
