# This module provides visualization utilities for 3D car models and their associated slice weights
# It creates multiple visualizations:
# 1. The full car mesh
# 2. Top slices with their weights overlaid on the mesh
# 3. Individual slice visualizations
# 4. Heatmaps of weights across the car model
# 5. 2D projections onto different planes

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from torch.cuda.amp import autocast
import matplotlib.tri as mtri


def visualize_car_and_slices(sample_idx, results_dir, model, dataset, args=None):
    """
    Visualize a car model along with its relevant slices from the model.
    
    Args:
        sample_idx (int): The index of the sample in the dataset to visualize
        results_dir (str): Directory where results will be saved
        model: The trained model that provides slice weights
        dataset: The dataset containing car models
        args: Optional arguments, including wandb logging settings
    
    Returns:
        None: Saves visualizations to disk and optionally logs to wandb
    """
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Get data for the specified sample
    data, geom = dataset.get(sample_idx)
    device = next(model.parameters()).device
    data = data.to(device)
    geom = geom.to(device)

    # Get slice weights from the model by running inference
    with torch.no_grad():
        with autocast():
            _ = model((data, geom))
            slice_weights = model.get_last_block_slice_weights()
            if slice_weights is None:
                print(f"Warning: No slice weights available for sample {sample_idx}")
                return

    # Extract surface points for visualization - these are the visible points of the car
    surf_idx = data.surf.cpu().numpy()
    if surf_idx.dtype == bool:
        surf_indices = np.where(surf_idx)[0]
    else:
        surf_indices = surf_idx
    pos = data.pos[surf_indices].cpu().numpy()
    pos = pos[:, [0, 2, 1]]  # Swap Y and Z for better visualization (standard convention)

    # Helper function to ensure 3D plots have equal scaling on all axes
    def set_axes_equal(ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc. This is achieved by setting the limits of all axes
        to have the same range centered around the middle of the plot.
        
        Args:
            ax: A matplotlib 3D axis object
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    # === Plot full car mesh ===
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=5, c='gray', alpha=0.8)
    ax1.set_title(f'Full Car Mesh - Sample {sample_idx}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_axes_equal(ax1)
    ax1.view_init(elev=20, azim=300)  # Set camera angle for optimal view
    ax1.grid(False)
    fig.patch.set_facecolor('white')

    # === Process slice weights ===
    # Convert weights to numpy and scale for better visibility
    weights = slice_weights.cpu().numpy()
    weights = weights * 1000  # Scale weights to make them more visible
    print(f"weights shape: {weights.shape}")
    
    # Handle different weight tensor shapes that can come from various model architectures
    if len(weights.shape) == 4:  # [B, H, N, G] as expected (Batch, Heads, Points, Slices)
        print("Processing [B, H, N, G] shaped weights")
        avg_weights = weights[0].sum(axis=0)  # Average over heads -> [N, G]
        print(f"avg_weights shape after mean: {avg_weights.shape}")
        
        # Handle case where dimensions might be transposed 
        if avg_weights.shape[0] == 32:  # If first dimension is 32 (slices), not points
            print("Transposing weights as dimensions are swapped")
            avg_weights = avg_weights.T  # Transpose to [N, G]
            print(f"avg_weights shape after transpose: {avg_weights.shape}")
        
        # Select only surface points for visualization with safety check
        if np.max(surf_indices) >= avg_weights.shape[0]:
            print(f"WARNING: surf_indices max value {np.max(surf_indices)} exceeds avg_weights dimension {avg_weights.shape[0]}")
            print(f"Using all weights without surface-point filtering")
            surf_weights = avg_weights
        else:
            surf_weights = avg_weights[surf_indices, :]
        print(f"surf_weights shape: {surf_weights.shape}")
    elif len(weights.shape) == 3:  # Alternative format: [B, N, G] (Batch, Points, Slices)
        print("Processing [B, N, G] shaped weights")
        avg_weights = weights[0]  # [N, G]
        print(f"avg_weights shape: {avg_weights.shape}")
        
        # Handle case where dimensions might be transposed
        if avg_weights.shape[0] == 32:  # If first dimension is 32 (slices), not points
            print("Transposing weights as dimensions are swapped")
            avg_weights = avg_weights.T  # Transpose to [N, G]
            print(f"avg_weights shape after transpose: {avg_weights.shape}")
            
        # Safety check for surface point filtering
        if np.max(surf_indices) >= avg_weights.shape[0]:
            print(f"WARNING: surf_indices max value {np.max(surf_indices)} exceeds avg_weights dimension {avg_weights.shape[0]}")
            print(f"Using all weights without surface-point filtering")
            surf_weights = avg_weights
        else:
            surf_weights = avg_weights[surf_indices, :]
        print(f"surf_weights shape: {surf_weights.shape}")
    else:
        # Fallback for unexpected weight shapes - try to infer the right structure
        print(f"Unexpected weight shape: {weights.shape}")
        if weights.shape[-1] == 32:  # Assume last dimension is number of slices
            avg_weights = weights.reshape(-1, 32)  # Reshape to [N, G]
        else:
            # Try to infer the right shape based on the last dimension
            avg_weights = weights.reshape(-1, weights.shape[-1])
        print(f"Reshaped avg_weights shape: {avg_weights.shape}")
        
        # Safety check for surface point filtering
        if np.max(surf_indices) >= avg_weights.shape[0]:
            print(f"WARNING: surf_indices max value {np.max(surf_indices)} exceeds avg_weights dimension {avg_weights.shape[0]}")
            print(f"Using all weights without surface-point filtering")
            surf_weights = avg_weights
        else:
            surf_weights = avg_weights[surf_indices, :]
        print(f"surf_weights shape: {surf_weights.shape}")
    
    # Find the most important slices based on total weight across points
    slice_importance = surf_weights.sum(axis=0)
    top_slices = np.argsort(slice_importance)[::-1][:5]  # Get indices of top 5 slices

    # Plot the top 5 slices with different colors and markers
    ax2 = fig.add_subplot(212, projection='3d')
    cmaps = [cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.cividis]  # Different colormaps for each slice
    markers = ['o', '^', 's', 'D', '*']  # Different markers for each slice

    for i, slice_idx in enumerate(top_slices):
        slice_weight = surf_weights[:, slice_idx]
        scatter = ax2.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            s=10,
            c=slice_weight,
            cmap=cmaps[i % len(cmaps)],
            marker=markers[i % len(markers)],
            alpha=0.7,
            label=f'Slice {slice_idx}'
        )
        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.7, pad=0.1)
        cbar.set_label(f'Slice {slice_idx} Weight')

    ax2.set_title('Top Slice Weights')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_axes_equal(ax2)
    ax2.view_init(elev=20, azim=300)  # Consistent camera angle with first plot
    ax2.grid(False)
    fig.patch.set_facecolor('white')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'car_slices_{sample_idx}.png'), dpi=300)
    plt.close()

    # === Plot each slice separately ===
    # For each slice, create a visualization where points are colored 
    # based on whether this slice has the maximum weight for that point
    num_slices = surf_weights.shape[1]
    for slice_idx in range(num_slices):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        slice_weight = surf_weights[:, slice_idx]
        
        # Find points where this slice has the highest activation
        max_weights_per_point = surf_weights.max(axis=1)
        is_max_slice = slice_weight >= max_weights_per_point
        
        # Convert to binary mask (1 where this slice dominates, 0 elsewhere)
        slice_weight = np.where(is_max_slice, 
                                np.ones_like(slice_weight), 
                                np.zeros_like(slice_weight))
        scatter = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            s=10,
            c=slice_weight,
            cmap=cm.viridis,
            alpha=0.7
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label(f'Slice {slice_idx} Weight')
        ax.set_title(f'Slice {slice_idx} Weights - Sample {sample_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)
        ax.view_init(elev=20, azim=300)
        ax.grid(False)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'slice_{slice_idx}_sample_{sample_idx}.png'), dpi=300)
        plt.close()

    # === Create heatmap showing the sum of all slice weights across the car ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    total_weight = surf_weights.sum(axis=1)  # Sum weights across all slices for each point
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=total_weight, cmap=cm.viridis, s=5, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Total Weight Across All Slices')
    ax.set_title(f'Combined Slice Weights - Sample {sample_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.view_init(elev=20, azim=300)
    ax.grid(False)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'slice_heatmap_{sample_idx}.png'), dpi=300)
    plt.close()

    # === Create 2D projections of the car onto XY, XZ, and YZ planes ===
    def plot_2d_projection(points, weights, title, filename, xy_plane=True, xz_plane=True, yz_plane=True):
        """
        Plot 2D projections of 3D points onto specified planes with color based on weights.
        
        Args:
            points (np.array): 3D points to project
            weights (np.array): Weight values for coloring each point
            title (str): Base title for the plots
            filename (str): Filename to save the visualization
            xy_plane (bool): Whether to create XY plane projection
            xz_plane (bool): Whether to create XZ plane projection
            yz_plane (bool): Whether to create YZ plane projection
        """
        # Determine number of subplots based on which planes are requested
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) if all([xy_plane, xz_plane, yz_plane]) else \
                    plt.subplots(1, 2, figsize=(12, 6)) if sum([xy_plane, xz_plane, yz_plane]) == 2 else \
                    plt.subplots(1, 1, figsize=(8, 6))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]  # Make it iterable for single subplot
            
        ax_index = 0
        
        # Create XY projection (top view of the car)
        if xy_plane:
            ax = axes[ax_index]
            sc = ax.scatter(points[:, 0], points[:, 1], c=weights, cmap=cm.viridis, s=3, alpha=0.7)
            ax.set_title(f'{title} - XY Projection')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(False)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            ax_index += 1
            
        # Create XZ projection (side view of the car)
        if xz_plane:
            ax = axes[ax_index]
            sc = ax.scatter(points[:, 0], points[:, 2], c=weights, cmap=cm.viridis, s=3, alpha=0.7)
            ax.set_title(f'{title} - XZ Projection')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_aspect('equal')
            ax.grid(False)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            ax_index += 1
            
        # Create YZ projection (front/rear view of the car)
        if yz_plane:
            ax = axes[ax_index]
            sc = ax.scatter(points[:, 1], points[:, 2], c=weights, cmap=cm.viridis, s=3, alpha=0.7)
            ax.set_title(f'{title} - YZ Projection')
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_aspect('equal')
            ax.grid(False)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, filename), dpi=300)
        plt.close()

    # Create 2D projections for the full car using all points (not just surface)
    all_points = data.pos.cpu().numpy()
    # Swap Y and Z axes to match 3D visualization
    all_points = all_points[:, [0, 2, 1]]
    
    # Plot projections with uniform color first (to see the car structure)
    uniform_weights = np.ones(all_points.shape[0])
    plot_2d_projection(all_points, uniform_weights, 
                      f'Full Car Structure - Sample {sample_idx}', 
                      f'car_projections_{sample_idx}.png')
    
    # For each slice, create 2D projections of slice weights across all points
    for slice_idx in range(num_slices):
        # Get weights for all points, not just surface points
        # Need to handle the same possible weight shapes as before
        if len(weights.shape) == 4:  # [B, H, N, G]
            all_weights = weights[0].mean(axis=0)  # [N, G]
            if all_weights.shape[0] == 32:  # If first dimension is 32 (slices), not points
                all_weights = all_weights.T  # Transpose to [N, G]
        elif len(weights.shape) == 3:  # [B, N, G]
            all_weights = weights[0]  # [N, G]
            if all_weights.shape[0] == 32:  # If first dimension is 32 (slices), not points
                all_weights = all_weights.T  # Transpose to [N, G]
        else:
            if weights.shape[-1] == 32:  # Assume last dimension is slices
                all_weights = weights.reshape(-1, 32)  # [N, G]
            else:
                all_weights = weights.reshape(-1, weights.shape[-1])
                
        # Make sure weights match the shape of all_points before plotting
        if all_weights.shape[0] == all_points.shape[0]:
            slice_weight = all_weights[:, slice_idx]
            plot_2d_projection(all_points, slice_weight, 
                              f'Slice {slice_idx} Weights - Sample {sample_idx}', 
                              f'slice_{slice_idx}_projections_{sample_idx}.png')
        else:
            print(f"Warning: Cannot create projections for slice {slice_idx} - shape mismatch")
    
    # Create projection for total weight across all slices
    if all_weights.shape[0] == all_points.shape[0]:
        total_all_weight = all_weights.sum(axis=1)
        plot_2d_projection(all_points, total_all_weight, 
                          f'Combined Slice Weights - Sample {sample_idx}', 
                          f'total_projections_{sample_idx}.png')
    else:
        print(f"Warning: Cannot create total weight projections - shape mismatch")

    print(f"Visualizations for sample {sample_idx} saved to {vis_dir}")
    
    # Optional: log generated images to Weights & Biases if enabled
    if args and hasattr(args, 'disable_wandb') and not args.disable_wandb:
        import wandb
        # Log individual slice images
        for slice_idx in range(num_slices):
            wandb.log({f"slice_{slice_idx}_sample_{sample_idx}": wandb.Image(os.path.join(vis_dir, f'slice_{slice_idx}_sample_{sample_idx}.png'))})
        # Log the combined heatmap
        wandb.log({
            f"slice_heatmap_{sample_idx}": wandb.Image(os.path.join(vis_dir, f'slice_heatmap_{sample_idx}.png'))
        })