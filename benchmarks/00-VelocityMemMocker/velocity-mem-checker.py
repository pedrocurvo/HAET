"""
This script checks the memory usage of the HAET model
"""

import time
import torch
import sys
import os
from torch.amp import autocast # Updated to use torch.amp instead of torch.cuda.amp

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.HAETransolver_Irregular_Mesh import Model


def benchmark_model(num_points, space_dim=1, fun_dim=1, n_hidden=256):
    """
    Benchmarks the HAETransolver model for a given number of points.

    Args:
        num_points (int): The number of points to test.
        space_dim (int): The dimension of the spatial coordinates.
        fun_dim (int): The dimension of the input function values.
        n_hidden (int): The hidden dimension size of the model.
    """
    print(f"--- Benchmarking with {num_points} points ---")

    # Create dummy input data
    x = torch.rand(1, num_points, space_dim).cuda()  # Batch size of 1
    fx = torch.rand(1, num_points, fun_dim).cuda()  # Batch size of 1

    # Instantiate the model
    model = Model(
        space_dim=space_dim,
        fun_dim=fun_dim,
        n_hidden=n_hidden,
        n_layers=1,
        n_head=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=32,
        unified_pos=0
    ).cuda()
    model.eval()  # Set model to evaluation mode

    # Warm-up GPU
    for _ in range(3):
        with autocast(device_type='cuda'): # Updated to include device_type
            _ = model(x, fx)
        torch.cuda.synchronize()

    # Measure forward pass time
    torch.cuda.synchronize()  # Wait for all kernels to complete before starting timer
    start_time = time.time()
    with autocast(device_type='cuda'): # Updated to include device_type
        output = model(x, fx)
    torch.cuda.synchronize()  # Wait for model forward pass to complete
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Forward pass time: {time_taken:.4f} seconds")

    # Measure memory usage
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats before the operation
    initial_memory = torch.cuda.memory_allocated()

    # Perform the operation for which memory is to be measured
    with autocast(device_type='cuda'): # Updated to include device_type
        output = model(x, fx)  # Re-run forward pass
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()  # Peak memory since last reset

    print(f"Initial GPU memory allocated: {initial_memory / 1024**2:.2f} MB")
    print(f"Final GPU memory allocated (after fwd pass): {final_memory / 1024**2:.2f} MB")
    print(f"Peak GPU memory allocated during fwd pass: {peak_memory / 1024**2:.2f} MB")

    return time_taken, peak_memory


if __name__ == "__main__":
    point_counts = [1000, 10000, 100000, 1000000, 2000000, 3000000]
    space_dim_test = 3
    fun_dim_test = 1
    n_hidden_test = 256
    
    # Clear CUDA cache before starting benchmark
    torch.cuda.empty_cache()
    print("Starting HAETransolver Benchmark...")
    for points in point_counts:
        try:
            # Clear CUDA cache before each benchmark run
            torch.cuda.empty_cache()
            time_taken, memory_used = benchmark_model(
                num_points=points,
                space_dim=space_dim_test,
                fun_dim=fun_dim_test,
                n_hidden=n_hidden_test
            )
            print(f"Successfully benchmarked {points} points. Time: {time_taken:.4f}s, Peak Memory: {memory_used / 1024**2:.2f} MB")
            print("-" * 30)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Ran out of memory with {points} points.")
                print(f"Error: {e}")
                break
            else:
                print(f"A runtime error occurred with {points} points: {e}")
                break
        except Exception as e:
            print(f"An unexpected error occurred with {points} points: {e}")
            break
    print("Benchmark finished.")

