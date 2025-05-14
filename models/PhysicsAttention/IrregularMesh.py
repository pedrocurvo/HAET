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
    """Physics-informed attention for irregular mesh data with Transolver++.

    This attention mechanism processes irregular mesh data through three main steps:
    1. Slicing: Projects input features into a reduced set of eidetic states using Rep-Slice
    2. Transformation: Processes eidetic states using the ErwinTransformer
    3. De-slicing: Projects transformed eidetic states back to the original points

    This approach allows efficient processing of irregular point clouds while
    maintaining awareness of spatial relationships.

    Attributes:
        dim_head (int): Dimension of each attention head
        heads (int): Number of attention heads
        dimensionality (int): Spatial dimensionality (3 for irregular meshes)
        epsilon (float): Small constant for Rep-Slice computation
        base_temp (float): Base temperature for adaptive temperature scaling
        in_project_x (nn.Linear): Linear projection for input features
        in_project_slice (nn.Linear): Linear projection for slice weights
        ada_temp_linear (nn.Linear): Linear projection for adaptive temperature
        erwin (ErwinTransformer): Transformer for processing eidetic states
        to_out (nn.Sequential): Output projection
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, base_temp=0.5, epsilon=1e-6):
        """Initialize the Physics_Attention_Irregular_Mesh module with Transolver++.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            slice_num (int): Number of slice tokens to use
            base_temp (float): Base temperature for adaptive temperature scaling
            epsilon (float): Small constant for the log(-log(ε)) term in Rep-Slice
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dimensionality = 3  # Spatial dimensionality for the irregular mesh
        self.epsilon = epsilon
        
        # For Transolver++, we only need one projection to save memory
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        
        # Ada-Temp: Base temperature + adaptive component
        self.base_temp = base_temp
        self.ada_temp_linear = nn.Linear(dim_head, 1)  # Adaptive temperature adjustment
        
        torch.nn.init.orthogonal_(self.in_project_slice.weight)  # use a principled initialization

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
        """Forward pass of the physics-informed attention module with Transolver++.
        
        Implements Transolver++ Algorithm 1: Parallel Physics-Attention with Eidetic States

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, num_points, channels]

        Returns:
            torch.Tensor: Output feature tensor of shape [batch_size, num_points, channels]
        """
        # Extract batch size, number of points, and channels
        B, N, C = x.shape

        ### (1) Slice operation: Project input features to a reduced set of tokens
        # Project input - drop fx to save 50% memory as per algorithm
        x_proj = (
            self.in_project_x(x)  # Linear projection [B, N, inner_dim]
            .reshape(
                B, N, self.heads, self.dim_head
            )  # Reshape for multi-head [B, N, H, C]
            .permute(0, 2, 1, 3)  # Reorder to [B, H, N, C]
            .contiguous()
        )
        
        # Compute adaptive temperature (Ada-Temp): τ = τ0 + Linear(xi)
        # Implementation: τ(k) ←τ0 + Ada-Temp(x(k))
        adaptive_temp = self.base_temp + self.ada_temp_linear(x_proj).clamp(min=-0.4, max=0.4)
        
        # Compute Rep-Slice: Softmax(Linear(x) - log(-log(ε))) / τ
        # Implementation: w(k) ← Rep-Slice(x(k),τ(k))
        log_neg_log_epsilon = torch.log(-torch.log(torch.tensor(self.epsilon, device=x.device)))
        slice_logits = self.in_project_slice(x_proj) - log_neg_log_epsilon
        slice_weights = torch.softmax(slice_logits / adaptive_temp, dim=2)  # [B, H, N, G]

        # Compute weights norm: w(k)_norm ← sum_i(w(k)_i)
        slice_norm = slice_weights.sum(2)  # [B, H, G]

        # Compute eidetic states: s(k) ← w(k)T x(k) / w_norm
        # We use x_proj both as x and f to save memory
        eidetic_states = torch.einsum(
            "bhnc,bhng->bhgc", x_proj, slice_weights
        )  # [B, H, G, C]

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

        # Create artificial positions for eidetic states in 3D space
        # This is needed because ErwinTransformer is position-aware
        # Using random positions as eidetic states don't have inherent positions
        pos = torch.rand(B * H * G, self.dimensionality, device=eidetic_states.device)

        # Create batch indices to separate different batch and head combinations
        # Each batch-head pair is treated as a separate point cloud by the transformer
        batch_idx = torch.arange(B * H, device=eidetic_states.device).repeat_interleave(G)

        # Process through ErwinTransformer to allow interactions between eidetic states
        # This enables global information exchange and feature refinement
        # This corresponds to: Update eidetic states s′← Attention(s)
        processed_states = self.erwin(eidetic_states_flat, pos, batch_idx)

        # Reshape the processed states back to the multi-head format
        out_eidetic_states = processed_states.reshape(B, H, G, C)

        ### (3) Deslice operation: Project processed states back to original points
        # Distribute processed eidetic states information back to original points
        # Using the same slice weights ensures information flows properly between
        # the slicing and de-slicing operations
        # Implementation: x′(k) ← Deslice(s′, w(k))
        out_x = torch.einsum("bhgc,bhng->bhnc", out_eidetic_states, slice_weights)

        # Rearrange from multi-head format to batch-first format
        # Concatenating all heads into a single feature dimension
        out_x = rearrange(out_x, "b h n d -> b n (h d)")

        # Project concatenated features back to the original dimension
        return self.to_out(out_x)
