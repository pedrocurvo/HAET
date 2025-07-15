"""
This script checks the memory usage of the HAET model
"""

import time
import torch
import sys
import os
from torch.amp import autocast # Updated to use torch.amp instead of torch.cuda.amp
from transolver import Model as transolver_model

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.HAETransolver_Irregular_Mesh import Model


def benchmark_model(num_points, space_dim=1, fun_dim=1, n_hidden=256, slice_num=32, model_type='HAETransolver'):
    """
    Benchmarks the HAETransolver model for a given number of points.

    Args:
        num_points (int): The number of points to test.
        space_dim (int): The dimension of the spatial coordinates.
        fun_dim (int): The dimension of the input function values.
        n_hidden (int): The hidden dimension size of the model.
    """
    # print(f"--- Benchmarking with {num_points} points ---")

    # Create dummy input data
    x = torch.rand(1, num_points, space_dim).cuda()  # Batch size of 2
    fx = torch.rand(1, num_points, fun_dim).cuda()  # Batch size of 2

    # Instantiate the model
    if model_type == 'HAETransolver':
        model = Model(
            space_dim=space_dim,
            fun_dim=fun_dim,
            n_hidden=n_hidden,
            n_layers=1,
            n_head=8,
            out_dim=1,
            slice_num=slice_num,
            unified_pos=0
        ).cuda()
    elif model_type == 'Transolver':
        model = transolver_model(
            space_dim=space_dim,
            fun_dim=fun_dim,
            n_hidden=n_hidden,
            n_layers=1,
            n_head=8,
            out_dim=1,
            slice_num=slice_num
        ).cuda()
    
    # Compile the model
    # model = torch.compile(model)  # Use torch.compile for dynamic shapes
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
    # print(f"Forward pass time: {time_taken:.4f} seconds")

    # Measure memory usage
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats before the operation
    initial_memory = torch.cuda.memory_allocated()

    # Perform the operation for which memory is to be measured
    with autocast(device_type='cuda'): # Updated to include device_type
        output = model(x, fx)  # Re-run forward pass
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()  # Peak memory since last reset

    # print(f"Initial GPU memory allocated: {initial_memory / 1024**2:.2f} MB")
    # print(f"Final GPU memory allocated (after fwd pass): {final_memory / 1024**2:.2f} MB")
    # print(f"Peak GPU memory allocated during fwd pass: {peak_memory / 1024**2:.2f} MB")

    return time_taken, peak_memory


if __name__ == "__main__":
    point_counts = [5000, 10000, 50000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    space_dim_test = 3
    fun_dim_test = 1
    n_hidden_test = 256
    num_slices = [64, 128, 256, 512, 1024]  # Different slice numbers to test

    # pd keep the results in a pandas DataFrame
    import pandas as pd
    results = pd.DataFrame(columns=['Model', 'Slices', 'Points', 'Time (s)', 'Peak Memory (MB)'])
    
    # Clear CUDA cache before starting benchmark
    torch.cuda.empty_cache()
    # print("Starting HAETransolver Benchmark...")
    for points in point_counts:
        for model_type in ['HAETransolver', 'Transolver']:
            # print(f"Benchmarking {model_type} with {points} points...")
            try:
                # Clear CUDA cache before each benchmark run
                torch.cuda.empty_cache()
                for slice_num in num_slices:
                    time_taken, memory_used = benchmark_model(
                        num_points=points,
                        space_dim=space_dim_test,
                        fun_dim=fun_dim_test,
                        n_hidden=n_hidden_test,
                        slice_num=slice_num,
                        model_type=model_type
                    )
                    print(f"{model_type}: Successfully benchmarked {points} points with {slice_num} slices. Time: {time_taken:.4f}s, Peak Memory: {memory_used / 1024**2:.2f} MB")
                    # Append results to DataFrame
                    new_row = {
                        'Model': model_type,
                        'Slices': slice_num,
                        'Points': points,
                        'Time (s)': time_taken,
                        'Peak Memory (MB)': memory_used / 1024**2
                    }
                    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                    print("-" * 30)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Ran out of memory with {points} points.")
                    print(f"Error: {e}")
                    # Continue to the next point count
                    torch.cuda.empty_cache()
                    # Append a placeholder result for this run
                    new_row =  pd.DataFrame({
                        'Model': model_type,
                        'Slices': slice_num,
                        'Points': points,
                        'Time (s)': None,
                        'Peak Memory (MB)': None
                    }, index=[0])
                    results = pd.concat([results, new_row], ignore_index=True)
                    continue
                    # If it's the last point count, break the loop

                else:
                    print(f"A runtime error occurred with {points} points: {e}")
                    break
            except Exception as e:
                print(f"An unexpected error occurred with {points} points: {e}")
                break
    print("Benchmark finished.")

    # Order by Slices and then by Points and then by Model
    results = results.sort_values(by=['Slices', 'Points', 'Model']).reset_index(drop=True)
    # Reset index to ensure it starts from 0
    results.index = range(len(results))
    # Print results DataFrame
    print(results)

    # Print results as latex table
    print(results.to_latex(index=False, float_format="%.2f", column_format="lcccc"))
    # Save results to a CSV file
    results.to_csv('benchmark_results.csv', index=False)

