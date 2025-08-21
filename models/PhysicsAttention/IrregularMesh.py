"""
Physics-informed attention mechanism for irregular meshes.

This module implements a specialized attention mechanism that can process
point data in irregular meshes using physics-informed principles. It supports
data in 1D, 2D or 3D space through a balltree-based approach combined
with an ErwinTransformer for enhanced feature interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from ..components import ErwinFlashTransformer as ErwinTransformer
from ..components.balltree import partition_balltree


class BallTreeAttention(nn.Module):
    """Ball Tree Attention module for extracting supernodes from spatial regions.
    
    This module takes points partitioned into balls/regions and uses attention
    to extract a single representative supernode for each ball.
    """
    
    def __init__(self, dim_head: int, num_heads: int, ball_size: int, dimensionality: int = 3):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.dimensionality = dimensionality
        
        # Attention components
        self.q_proj = nn.Linear(dim_head, dim_head)
        self.k_proj = nn.Linear(dim_head, dim_head)
        self.v_proj = nn.Linear(dim_head, dim_head)
        self.out_proj = nn.Linear(dim_head, dim_head)
        
        # Positional encoding
        self.pos_encoder = nn.Linear(dimensionality, dim_head)
        
        # Learnable query for supernode extraction
        self.supernode_query = nn.Parameter(torch.randn(1, 1, dim_head))
        nn.init.xavier_uniform_(self.supernode_query)
        
        self.scale = dim_head ** -0.5
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract supernodes from balls using attention.
        
        Args:
            x: Features [B*H, N, D] where N is points per ball
            pos: Positions [B*H, N, dimensionality]
            
        Returns:
            supernodes: [B*H, D] - one supernode per ball
            supernode_pos: [B*H, dimensionality] - position of each supernode
        """
        B_H, N, D = x.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoder(pos)  # [B*H, N, D]
        x_with_pos = x + pos_enc
        
        # Compute K, V from all points in the ball
        k = self.k_proj(x_with_pos)  # [B*H, N, D]
        v = self.v_proj(x_with_pos)  # [B*H, N, D]
        
        # Compute Q from learnable supernode query
        q = self.supernode_query.expand(B_H, -1, -1)  # [B*H, 1, D]
        q = self.q_proj(q)  # [B*H, 1, D]
        
        # Attention mechanism
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*H, 1, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*H, 1, N]
        
        # Extract supernode features
        supernodes = torch.matmul(attn_weights, v).squeeze(1)  # [B*H, D]
        
        # Compute supernode positions as weighted average
        supernode_pos = torch.matmul(attn_weights, pos).squeeze(1)  # [B*H, dimensionality]
        
        # Final projection
        supernodes = self.out_proj(supernodes)  # [B*H, D]
        
        return supernodes, supernode_pos


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics-informed attention for irregular mesh data with BallTree partitioning.

    This attention mechanism processes irregular mesh data through three main steps:
    1. Partitioning: Uses BallTree to partition input points into spatial regions
    2. Ball Attention: Processes each ball/region using attention to extract supernodes
    3. Transformation: Processes supernodes using the ErwinTransformer and upsamples back

    This approach allows efficient processing of irregular point clouds while
    maintaining strong spatial locality and awareness of neighborhood relationships.

    Attributes:
        dim_head (int): Dimension of each attention head
        heads (int): Number of attention heads
        dimensionality (int): Spatial dimensionality (3 for irregular meshes)
        ball_size (int): Number of points per ball/region
        num_balls (int): Number of balls to partition points into
        in_project_x (nn.Linear): Linear projection for input features
        ball_attention (BallTreeAttention): Attention module for extracting supernodes
        erwin (ErwinTransformer): Transformer for processing supernodes
        to_out (nn.Sequential): Output projection
    """

    def __init__(
        self, 
        dim, 
        heads=8, 
        dim_head=64, 
        dropout=0.0, 
        ball_size=32,         # Size of each ball/region
        num_balls=None,       # Number of balls (auto-computed if None)
        radius: float = 1.0,
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
        attention_heads=1,
    ):
        """Initialize the Physics_Attention_Irregular_Mesh module with BallTree partitioning.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            ball_size (int): Number of points per ball/region
            num_balls (int): Number of balls to partition points into (auto-computed if None)
            radius (float): Radius parameter for ErwinTransformer
            dimensionality (int): Spatial dimensionality
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
        self.attn_heads = attention_heads
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * self.attn_heads  # Inner dimension for multi-head attention
        self.ball_size = ball_size
        self.num_balls = num_balls  # Will be computed dynamically based on input size
        self.dimensionality = dimensionality  # Spatial dimensionality for the irregular mesh
        self.radius = radius    # Store the radius parameter

        # For BallTree approach, we project input to multi-head representation
        self.in_project_x = nn.Linear(dim, inner_dim)

        # Ball attention module for extracting supernodes from each ball
        self.ball_attention = BallTreeAttention(
            dim_head=dim_head,
            num_heads=self.attn_heads,
            ball_size=ball_size,
            dimensionality=dimensionality
        )

        # Set default ErwinTransformer parameters if not provided
        if c_hidden is None:
            c_hidden = [dim_head, dim_head * 2]
        if ball_sizes is None:
            # Use adaptive ball sizes based on expected number of supernodes
            ball_sizes = [max(32, ball_size // 2), max(32, ball_size // 2)]
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

        # Hierarchical transformer for processing supernodes
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
    
    def partition_points_into_balls(self, pos: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Partition points into balls using balltree with target number of partitions.
        
        Args:
            pos: Point positions [N, dimensionality]
            batch_idx: Batch indices [N]
            
        Returns:
            ball_indices: [N] - ball assignment for each point
        """
        N = pos.shape[0]
        
        # Compute number of balls dynamically if not set
        if self.num_balls is None:
            # Target approximately ball_size points per ball
            num_balls = max(1, N // self.ball_size)
        else:
            num_balls = self.num_balls
            
        # Compute target partitioning level
        # Each partition level doubles the number of regions
        target_level = max(0, int(math.log2(num_balls)) if num_balls > 1 else 0)
        
        # Use balltree partitioning
        ball_indices = partition_balltree(pos, batch_idx, target_level)
        
        return ball_indices

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the physics-informed attention module with BallTree partitioning.
        
        Implements BallTree-based Physics-Attention with Supernodes:
        1. Partition points into spatial balls using balltree
        2. Extract supernodes from each ball using attention
        3. Process supernodes with ErwinTransformer
        4. Upsample back to original points

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, num_points, channels]
            pos (torch.Tensor): Position tensor of shape [batch_size, num_points, dimensionality]

        Returns:
            torch.Tensor: Output feature tensor of shape [batch_size, num_points, channels]
        """
        if pos is None:
            raise ValueError("Position tensor is required for BallTree partitioning")
            
        # Extract batch size, number of points, and channels
        B, N, C = x.shape
        
        # Flatten for processing
        x_flat = x.view(B * N, C)  # [B*N, C]
        pos_flat = pos.view(B * N, self.dimensionality)  # [B*N, D]
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)  # [B*N]

        ### (1) Project input features to multi-head representation
        x_proj = self.in_project_x(x_flat)  # [B*N, inner_dim]
        x_proj = x_proj.view(B * N, self.attn_heads, self.dim_head)  # [B*N, H, D]

        ### (2) Partition points into balls using balltree
        ball_indices = self.partition_points_into_balls(pos_flat, batch_idx)  # [B*N]
        
        # Group points by ball
        unique_balls, inverse_indices = torch.unique(ball_indices, return_inverse=True)
        num_balls = len(unique_balls)
        
        # Prepare data for ball processing
        supernodes_list = []
        supernode_pos_list = []
        ball_batch_idx_list = []
        
        for head_idx in range(self.attn_heads):
            head_supernodes = []
            head_supernode_pos = []
            head_ball_batch = []
            
            for ball_id in unique_balls:
                # Get points in this ball
                mask = (ball_indices == ball_id)
                ball_points = x_proj[mask, head_idx, :]  # [points_in_ball, D]
                ball_pos = pos_flat[mask]  # [points_in_ball, dimensionality]
                ball_batch = batch_idx[mask]  # [points_in_ball]
                
                if len(ball_points) == 0:
                    continue
                    
                # Pad or truncate to ball_size for consistent processing
                if len(ball_points) < self.ball_size:
                    # Pad with last point
                    pad_size = self.ball_size - len(ball_points)
                    last_point = ball_points[-1:].expand(pad_size, -1)
                    last_pos = ball_pos[-1:].expand(pad_size, -1) 
                    ball_points = torch.cat([ball_points, last_point], dim=0)
                    ball_pos = torch.cat([ball_pos, last_pos], dim=0)
                elif len(ball_points) > self.ball_size:
                    # Truncate to ball_size
                    ball_points = ball_points[:self.ball_size]
                    ball_pos = ball_pos[:self.ball_size]
                
                # Extract supernode using attention
                ball_points = ball_points.unsqueeze(0)  # [1, ball_size, D]
                ball_pos = ball_pos.unsqueeze(0)  # [1, ball_size, dimensionality]
                
                supernode, supernode_pos = self.ball_attention(ball_points, ball_pos)
                supernode = supernode.squeeze(0)  # [D]
                supernode_pos = supernode_pos.squeeze(0)  # [dimensionality]
                
                head_supernodes.append(supernode)
                head_supernode_pos.append(supernode_pos)
                # Use the batch index of the first point in the ball
                head_ball_batch.append(ball_batch[0])
            
            if head_supernodes:
                supernodes_list.append(torch.stack(head_supernodes))  # [num_balls, D]
                supernode_pos_list.append(torch.stack(head_supernode_pos))  # [num_balls, dimensionality]
                ball_batch_idx_list.append(torch.stack(head_ball_batch))  # [num_balls]

        if not supernodes_list:
            # Fallback if no balls were created
            return x
            
        # Stack across heads
        supernodes = torch.stack(supernodes_list, dim=1)  # [num_balls, H, D]
        supernode_pos = supernode_pos_list[0]  # [num_balls, dimensionality] - same for all heads
        supernode_batch = ball_batch_idx_list[0]  # [num_balls] - same for all heads

        ### (3) Process supernodes with ErwinTransformer
        # Flatten for ErwinTransformer processing
        supernodes_flat = supernodes.view(-1, self.dim_head)  # [num_balls*H, D]
        supernode_pos_expanded = supernode_pos.unsqueeze(1).expand(-1, self.attn_heads, -1).reshape(-1, self.dimensionality)
        supernode_batch_expanded = supernode_batch.unsqueeze(1).expand(-1, self.attn_heads).reshape(-1)
        
        # Process with ErwinTransformer
        processed_supernodes = self.erwin(
            supernodes_flat, 
            supernode_pos_expanded, 
            supernode_batch_expanded, 
            radius=self.radius
        )  # [num_balls*H, D]
        
        # Reshape back
        processed_supernodes = processed_supernodes.view(num_balls, self.attn_heads, self.dim_head)

        ### (4) Upsample back to original points
        # For each original point, find its ball and use the corresponding processed supernode
        output_features = []
        for head_idx in range(self.attn_heads):
            head_output = torch.zeros(B * N, self.dim_head, device=x.device)
            
            for i, ball_id in enumerate(unique_balls):
                mask = (ball_indices == ball_id)
                head_output[mask] = processed_supernodes[i, head_idx].unsqueeze(0).expand(mask.sum(), -1)
            
            output_features.append(head_output)
        
        # Concatenate across heads
        output = torch.cat(output_features, dim=-1)  # [B*N, H*D]
        
        # Reshape back to original shape
        output = output.view(B, N, -1)  # [B, N, H*D]
        
        return self.to_out(output)
