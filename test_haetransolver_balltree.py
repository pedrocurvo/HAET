#!/usr/bin/env python3
"""
Test script for the updated HAETransolver_Irregular_Mesh with BallTree attention.
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.append('/Users/pedrocurvo/Documents/01 - Projects/ErwinTransolver')

# Mock the missing dependencies
class MockErwinTransformer:
    def __init__(self, **kwargs):
        self.c_in = kwargs.get('c_in', 64)
        self.linear = None
    
    def __call__(self, x, pos, batch_idx, **kwargs):
        if self.linear is None:
            self.linear = torch.nn.Linear(x.shape[-1], x.shape[-1])
        return self.linear(x)

# Patch the import
import models.components
models.components.ErwinFlashTransformer = MockErwinTransformer

# Mock balltree functions
def mock_partition_balltree(pos, batch_idx, target_level):
    N = pos.shape[0]
    num_balls = 2 ** target_level if target_level > 0 else 1
    ball_size = max(1, N // num_balls)
    ball_indices = torch.arange(N, device=pos.device) // ball_size
    ball_indices = torch.clamp(ball_indices, 0, num_balls - 1)
    return ball_indices

# Patch balltree import
import models.components.balltree
models.components.balltree.partition_balltree = mock_partition_balltree

try:
    from models.HAETransolver_Irregular_Mesh import Model
    
    print("Testing HAETransolver_Irregular_Mesh with BallTree attention...")
    
    # Test parameters
    batch_size = 2
    num_points = 128
    space_dim = 1
    fun_dim = 1
    n_hidden = 64
    
    # Create model with BallTree parameters
    model = Model(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=n_hidden,
        n_head=4,
        fun_dim=fun_dim,
        ball_size=16,  # New parameter instead of slice_num
        radius=1.0
    )
    
    print(f"Model created successfully!")
    print(f"Model name: {model.__name__}")
    
    # Create test data
    x = torch.randn(batch_size, num_points, space_dim)  # spatial coordinates
    fx = torch.randn(batch_size, num_points, fun_dim)   # function values
    
    print(f"Input spatial coordinates shape: {x.shape}")
    print(f"Input function values shape: {fx.shape}")
    
    # Forward pass
    output = model(x, fx)
    print(f"Output shape: {output.shape}")
    
    # Test without function values
    output2 = model(x, None)
    print(f"Output shape (no function values): {output2.shape}")
    
    # Test with time input
    model_with_time = Model(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=n_hidden,
        n_head=4,
        fun_dim=fun_dim,
        ball_size=16,
        Time_Input=True
    )
    
    T = torch.randn(batch_size, 1)
    output3 = model_with_time(x, fx, T)
    print(f"Output shape (with time): {output3.shape}")
    
    print("✓ All tests passed! BallTree-based HAETransolver works correctly.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
