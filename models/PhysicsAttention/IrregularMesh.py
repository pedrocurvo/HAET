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

from ..components import ErwinFlashTransformer as ErwinTransformer


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

    def __init__(
        self, 
        dim, 
        heads=8, 
        dim_head=64, 
        dropout=0.0, 
        slice_num=64, 
        base_temp=0.5, 
        epsilon=1e-6,
        radius: float = 1.0,     # Add radius parameter with default value
        dimensionality: int = 3,
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
        embed=False,
        memory_tokens=32,
    ):
        """Initialize the Physics_Attention_Irregular_Mesh module with Transolver++.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            slice_num (int): Number of slice tokens to use
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
        self.slice_num = slice_num
        self.dimensionality = dimensionality  # Spatial dimensionality for the irregular mesh
        self.epsilon = epsilon
        self.radius = radius    # Store the radius parameter
        self.memory_tokens = self.slice_num // 32

        # Memory tokens - learnable parameters
        self.memory_states = nn.Parameter(torch.randn(1, heads, self.memory_tokens, dim_head))
        init_pos = self.uniform_memory_positions(heads, self.memory_tokens, dimensionality).clone()
        self.memory_positions = nn.Parameter(init_pos)
        
        # Initialize memory tokens
        nn.init.xavier_uniform_(self.memory_states)
       #  nn.init.xavier_uniform_(self.memory_positions)
        
        # For Transolver++, we only need one projection to save memory
        self.in_project_x = nn.Linear(dim, inner_dim)

        # # Input positions for slicing - learn position representations
        # self.pos_projector = nn.Sequential(
        #     nn.Linear(dimensionality, dimensionality * heads),
        #     nn.GELU(),
        #     nn.Linear(dimensionality * heads, heads * self.dimensionality),
        # )

        # Rep-Slice projection
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.in_project_slice._is_rep_slice = True  # Tag for custom init
        nn.init.orthogonal_(self.in_project_slice.weight)
        
        # Ada-Temp: Base temperature + adaptive component
        self.base_temp = base_temp
        self.ada_temp_norm = nn.LayerNorm(dim_head)
        self.ada_temp_linear = nn.Linear(dim_head, 1)  # Adaptive temperature adjustment
        self.ada_temp_linear._is_ada_temp = True

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
        # This design uses a multi-level approach to capture features at different scales
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
            dimensionality=self.dimensionality,  # Dimensionality of the space
            mp_steps=mp_steps,      # Number of message passing steps
            embed=embed,            # Use parameter value for ErwinEmbedding
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Initialize slice weights attribute
        self.slice_weights = None
    
    def uniform_memory_positions(self, heads, memory_tokens, dimensionality):
        grid = torch.linspace(0, 1, steps=memory_tokens)
        if dimensionality == 1:
            base = grid
        elif dimensionality == 2:
            base = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).view(-1, 2)
        elif dimensionality == 3:
            base = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).view(-1, 3)
        else:
            raise ValueError("Unsupported dimensionality")

        base = base[:memory_tokens]  # Ensure size
        base = base.unsqueeze(0).unsqueeze(0).expand(1, heads, -1, -1)  # [1, H, M, D]
        return base
    
    def gumbel_softmax_sample(self, logits, tau, training=True):
        """Gumbel-Softmax sampling as in Eq. (4)"""
        if training:
            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            return torch.softmax((logits + gumbel_noise) / tau, dim=-1)
        else:
            return torch.softmax(logits / tau, dim=-1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
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

        # Project positions for slicing - learn position representations
        pos_proj = pos.view(B, N, 1, self.dimensionality).expand(B, N, self.heads, self.dimensionality).transpose(1, 2)
        
        # Adaptive temperature
        tau = torch.clamp(self.base_temp + self.ada_temp_linear(self.ada_temp_norm(x_proj)), min=0.1, max=2.0)

        # Rep-Slice: Get raw logits and apply Gumbel-Softmax
        raw_logits = self.in_project_slice(x_proj)  # [B, H, N, G]
        slice_weights = self.gumbel_softmax_sample(raw_logits, tau, self.training)  # [B, H, N, G]

        if not self.training:
            self.slice_weights = slice_weights  # Save for inspection

        # Normalize weights (for eidetic state computation)
        norm = slice_weights.sum(2, keepdim=True)  # [B, H, 1, G]

        # Eidetic state computation
        eidetic_states = torch.matmul(slice_weights.transpose(-2, -1), x_proj) / (norm.transpose(-1, -2) + 1e-3)
        eidetic_pos = torch.matmul(slice_weights.transpose(-2, -1), pos_proj) / (norm.transpose(-1, -2) + 1e-3)

        ### Add memory tokens to eidetic states
        # Expand memory tokens to match batch size
        memory_states_expanded = self.memory_states.expand(B, -1, -1, -1)  # [B, H, M, D]
        memory_pos_expanded = self.memory_positions.expand(B, -1, -1, -1)  # [B, H, M, D_pos]
        
        # Concatenate memory tokens with eidetic states
        eidetic_states = torch.cat([eidetic_states, memory_states_expanded], dim=2)  # [B, H, G+M, D]
        eidetic_pos = torch.cat([eidetic_pos, memory_pos_expanded], dim=2)  # [B, H, G+M, D_pos]

        ### (2) Transform eidetic states with ErwinTransformer
        # Prepare for Erwin - now includes memory tokens
        B, H, G_plus_M, D = eidetic_states.shape
        states_flat = eidetic_states.reshape(B * H * G_plus_M, D)
        pos_flat = eidetic_pos.reshape(B * H * G_plus_M, self.dimensionality)
        batch_idx = torch.arange(B * H, device=x.device).repeat_interleave(G_plus_M)

        # Erwin attention
        updated = self.erwin(states_flat, pos_flat, batch_idx, radius=self.radius)
        updated = updated.view(B, H, G_plus_M, D)
        
        # Split back into slice tokens and memory tokens
        updated_slices = updated[:, :, :self.slice_num, :]  # [B, H, G, D]
        updated_memory = updated[:, :, self.slice_num:, :]  # [B, H, M, D]

        # Deslice - only use the slice tokens for output
        out = torch.matmul(slice_weights, updated_slices)  # [B, H, N, G] @ [B, H, G, D] = [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)  # [B, N, C]
        return self.to_out(out)
