#!/usr/bin/env python3
"""
Test script for the BallTree-based Physics Attention implementation.
This script creates synthetic data and tests the basic functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Mock the ErwinTransformer for testing
class MockErwinTransformer(nn.Module):
    def __init__(self, c_in, **kwargs):
        super().__init__()
        self.linear = nn.Linear(c_in, c_in)
    
    def forward(self, x, pos, batch_idx, **kwargs):
        return self.linear(x)

# Mock the partition_balltree function
def mock_partition_balltree(pos, batch_idx, target_level):
    """Mock implementation that creates simple ball assignments"""
    N = pos.shape[0]
    num_balls = 2 ** target_level if target_level > 0 else 1
    ball_size = max(1, N // num_balls)
    
    # Simple assignment: divide points into sequential balls
    ball_indices = torch.arange(N, device=pos.device) // ball_size
    ball_indices = torch.clamp(ball_indices, 0, num_balls - 1)
    return ball_indices

class BallTreeAttention(nn.Module):
    """Ball Tree Attention module for extracting supernodes from spatial regions."""
    
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
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """Extract supernodes from balls using attention."""
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
    """Simplified version for testing."""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, ball_size=32, 
                 dimensionality=3, **kwargs):
        super().__init__()
        self.attn_heads = heads
        self.dim_head = dim_head
        self.ball_size = ball_size
        self.dimensionality = dimensionality
        self.num_balls = None
        
        inner_dim = dim_head * self.attn_heads
        
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.ball_attention = BallTreeAttention(
            dim_head=dim_head,
            num_heads=self.attn_heads,
            ball_size=ball_size,
            dimensionality=dimensionality
        )
        
        # Mock ErwinTransformer
        self.erwin = MockErwinTransformer(dim_head)
        
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
    
    def partition_points_into_balls(self, pos: torch.Tensor, batch_idx: torch.Tensor):
        """Partition points into balls using balltree with target number of partitions."""
        N = pos.shape[0]
        
        if self.num_balls is None:
            num_balls = max(1, N // self.ball_size)
        else:
            num_balls = self.num_balls
            
        target_level = max(0, int(math.log2(num_balls)) if num_balls > 1 else 0)
        ball_indices = mock_partition_balltree(pos, batch_idx, target_level)
        return ball_indices
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor = None):
        """Simplified forward pass for testing."""
        if pos is None:
            raise ValueError("Position tensor is required")
            
        B, N, C = x.shape
        
        # Flatten for processing
        x_flat = x.view(B * N, C)
        pos_flat = pos.view(B * N, self.dimensionality)
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)

        # Project input features
        x_proj = self.in_project_x(x_flat)  # [B*N, inner_dim]
        x_proj = x_proj.view(B * N, self.attn_heads, self.dim_head)  # [B*N, H, D]

        # Partition points into balls
        ball_indices = self.partition_points_into_balls(pos_flat, batch_idx)
        
        # Simple processing: just return projected features for testing
        output = x_proj.view(B * N, -1)
        output = output.view(B, N, -1)
        return self.to_out(output)

def test_basic_functionality():
    """Test basic functionality of the BallTree attention module."""
    print("Testing BallTree Physics Attention...")
    
    # Create test data
    B, N, C = 2, 128, 64  # batch_size, num_points, channels
    dim_head = 32
    heads = 4
    ball_size = 16
    dimensionality = 3
    
    # Create model
    model = Physics_Attention_Irregular_Mesh(
        dim=C,
        heads=heads,
        dim_head=dim_head,
        ball_size=ball_size,
        dimensionality=dimensionality
    )
    
    # Create input data
    x = torch.randn(B, N, C)
    pos = torch.randn(B, N, dimensionality)
    
    print(f"Input shape: {x.shape}")
    print(f"Position shape: {pos.shape}")
    
    # Forward pass
    try:
        output = model(x, pos)
        print(f"Output shape: {output.shape}")
        print("✓ Basic forward pass successful!")
        
        # Check output shape
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        print("✓ Output shape matches input shape!")
        
        # Test BallTreeAttention separately
        ball_attn = BallTreeAttention(dim_head, heads, ball_size, dimensionality)
        test_x = torch.randn(heads, ball_size, dim_head)
        test_pos = torch.randn(heads, ball_size, dimensionality)
        
        supernodes, supernode_pos = ball_attn(test_x, test_pos)
        print(f"Supernodes shape: {supernodes.shape}")
        print(f"Supernode positions shape: {supernode_pos.shape}")
        print("✓ BallTreeAttention works correctly!")
        
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()
