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
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    data, geom = dataset.get(sample_idx)
    device = next(model.parameters()).device
    data = data.to(device)
    geom = geom.to(device)

    with torch.no_grad():
        with autocast():
            _ = model((data, geom))
            slice_weights = model.get_last_block_slice_weights()
            if slice_weights is None:
                print(f"Warning: No slice weights available for sample {sample_idx}")
                return

    # Use only surface points for visualization
    surf_idx = data.surf.cpu().numpy()
    if surf_idx.dtype == bool:
        surf_indices = np.where(surf_idx)[0]
    else:
        surf_indices = surf_idx
    pos = data.pos[surf_indices].cpu().numpy()
    pos = pos[:, [0, 2, 1]]  # Swap Y and Z

    # Helper to set equal aspect ratio
    def set_axes_equal(ax):
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
    ax1.view_init(elev=20, azim=300)  # Changed from 120 to 300 to flip the view
    ax1.grid(False)
    fig.patch.set_facecolor('white')

    # === Process slice weights ===
    weights = slice_weights.cpu().numpy()  # Get weights as numpy
    # Scale weights by 1000 to make them more visible in visualizations
    weights = weights * 1000
    print(f"weights shape: {weights.shape}")
    
    # Check dimensions to understand the structure
    if len(weights.shape) == 4:  # [B, H, N, G] as expected
        print("Processing [B, H, N, G] shaped weights")
        avg_weights = weights[0].sum(axis=0)  # [N, G]
        print(f"avg_weights shape after mean: {avg_weights.shape}")
        
        if avg_weights.shape[0] == 32:  # If first dimension is 32 (slices), not points
            # Then it's [G, N] instead of [N, G], so we need to transpose
            print("Transposing weights as dimensions are swapped")
            avg_weights = avg_weights.T  # Transpose to [N, G]
            print(f"avg_weights shape after transpose: {avg_weights.shape}")
        
        # Now select only surface points - add safety check for out of bounds indices
        if np.max(surf_indices) >= avg_weights.shape[0]:
            print(f"WARNING: surf_indices max value {np.max(surf_indices)} exceeds avg_weights dimension {avg_weights.shape[0]}")
            print(f"Using all weights without surface-point filtering")
            surf_weights = avg_weights
        else:
            surf_weights = avg_weights[surf_indices, :]
        print(f"surf_weights shape: {surf_weights.shape}")
    elif len(weights.shape) == 3:  # Could be [B, N, G]
        print("Processing [B, N, G] shaped weights")
        avg_weights = weights[0]  # [N, G]
        print(f"avg_weights shape: {avg_weights.shape}")
        
        if avg_weights.shape[0] == 32:  # If first dimension is 32 (slices), not points
            # Then it's [G, N] instead of [N, G], so we need to transpose
            print("Transposing weights as dimensions are swapped")
            avg_weights = avg_weights.T  # Transpose to [N, G]
            print(f"avg_weights shape after transpose: {avg_weights.shape}")
            
        # Now select only surface points - add safety check for out of bounds indices
        if np.max(surf_indices) >= avg_weights.shape[0]:
            print(f"WARNING: surf_indices max value {np.max(surf_indices)} exceeds avg_weights dimension {avg_weights.shape[0]}")
            print(f"Using all weights without surface-point filtering")
            surf_weights = avg_weights
        else:
            surf_weights = avg_weights[surf_indices, :]
        print(f"surf_weights shape: {surf_weights.shape}")
    else:
        print(f"Unexpected weight shape: {weights.shape}")
        # Use a safer approach - reshape if needed
        if weights.shape[-1] == 32:  # Assume last dimension is number of slices
            avg_weights = weights.reshape(-1, 32)  # Reshape to [N, G]
        else:
            # Try to infer the right shape
            avg_weights = weights.reshape(-1, weights.shape[-1])
        print(f"Reshaped avg_weights shape: {avg_weights.shape}")
        # Add safety check for out of bounds indices
        if np.max(surf_indices) >= avg_weights.shape[0]:
            print(f"WARNING: surf_indices max value {np.max(surf_indices)} exceeds avg_weights dimension {avg_weights.shape[0]}")
            print(f"Using all weights without surface-point filtering")
            surf_weights = avg_weights
        else:
            surf_weights = avg_weights[surf_indices, :]
        print(f"surf_weights shape: {surf_weights.shape}")
    
    # Use surf_weights for all further plotting
    slice_importance = surf_weights.sum(axis=0)
    top_slices = np.argsort(slice_importance)[::-1][:5]

    ax2 = fig.add_subplot(212, projection='3d')
    cmaps = [cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.cividis]
    markers = ['o', '^', 's', 'D', '*']

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
    ax2.view_init(elev=20, azim=300)  # Changed from 120 to 300 to flip the view
    ax2.grid(False)
    fig.patch.set_facecolor('white')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'car_slices_{sample_idx}.png'), dpi=300)
    plt.close()

    # === Plot each slice separately ===
    num_slices = surf_weights.shape[1]
    for slice_idx in range(num_slices):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        slice_weight = surf_weights[:, slice_idx]
        # Compare each point's weight in this slice to the max weight across all slices for that point
        max_weights_per_point = surf_weights.max(axis=1)
        # Create a mask where this slice has the highest weight for each point
        is_max_slice = slice_weight >= max_weights_per_point
        # Use the original weights rather than binary 0/1 values for better visualization
        # If you want binary values instead, uncomment the next lines
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
        ax.view_init(elev=20, azim=300)  # Changed from 120 to 300 to flip the view
        ax.grid(False)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'slice_{slice_idx}_sample_{sample_idx}.png'), dpi=300)
        plt.close()

    # === Heatmap over entire car ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    total_weight = surf_weights.sum(axis=1)
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=total_weight, cmap=cm.viridis, s=5, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Total Weight Across All Slices')
    ax.set_title(f'Combined Slice Weights - Sample {sample_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.view_init(elev=20, azim=300)  # Changed from 120 to 300 to flip the view
    ax.grid(False)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'slice_heatmap_{sample_idx}.png'), dpi=300)
    plt.close()

    # === Plot 2D projections of the car onto XY, XZ, and YZ planes ===
    def plot_2d_projection(points, weights, title, filename, xy_plane=True, xz_plane=True, yz_plane=True):
        """Plot 2D projections of 3D points onto specified planes with color based on weights."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) if all([xy_plane, xz_plane, yz_plane]) else \
                    plt.subplots(1, 2, figsize=(12, 6)) if sum([xy_plane, xz_plane, yz_plane]) == 2 else \
                    plt.subplots(1, 1, figsize=(8, 6))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]  # Make it iterable for single subplot
            
        ax_index = 0
        
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

    # Create 2D projections for the full car (using all points, not just surface)
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
                
        # Make sure weights match the shape of all_points
        if all_weights.shape[0] == all_points.shape[0]:
            slice_weight = all_weights[:, slice_idx]
            plot_2d_projection(all_points, slice_weight, 
                              f'Slice {slice_idx} Weights - Sample {sample_idx}', 
                              f'slice_{slice_idx}_projections_{sample_idx}.png')
        else:
            print(f"Warning: Cannot create projections for slice {slice_idx} - shape mismatch")
    
    # Also create projection for total weight
    if all_weights.shape[0] == all_points.shape[0]:
        total_all_weight = all_weights.sum(axis=1)
        plot_2d_projection(all_points, total_all_weight, 
                          f'Combined Slice Weights - Sample {sample_idx}', 
                          f'total_projections_{sample_idx}.png')
    else:
        print(f"Warning: Cannot create total weight projections - shape mismatch")

    print(f"Visualizations for sample {sample_idx} saved to {vis_dir}")
    
    # Optional: log to wandb
    if args and hasattr(args, 'disable_wandb') and not args.disable_wandb:
        import wandb
        # Optionally log all slice images
        for slice_idx in range(num_slices):
            wandb.log({f"slice_{slice_idx}_sample_{sample_idx}": wandb.Image(os.path.join(vis_dir, f'slice_{slice_idx}_sample_{sample_idx}.png'))})
        wandb.log({
            f"slice_heatmap_{sample_idx}": wandb.Image(os.path.join(vis_dir, f'slice_heatmap_{sample_idx}.png'))
        })