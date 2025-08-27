import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv

def rk4_step(f, pos, h):
    """
    Performs one RK4 integration step. Supports pos of shape (3,) or (N, 3).

    Parameters:
        f   : velocity field function.
        pos : current position (numpy array of shape (3,) or (N, 3)).
        h   : integration step size.

    Returns:
        New position after one RK4 step.
    """
    k1 = f(pos)
    k2 = f(pos + 0.5 * h * k1)
    k3 = f(pos + 0.5 * h * k2)
    k4 = f(pos + h * k3)
    return pos + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def compute_streamlines(seeds, f, h=0.1, max_steps=500, tolerance=1e-6, method="rk4"):
    """
    Computes streamlines for multiple seed points (shape (N, 3)). Each streamline is integrated
    in parallel until either max_steps is reached or the velocity magnitude falls below tolerance.

    Parameters:
        seeds     : array of seed points, shape (N, 3) (or (3,) for a single seed).
        f         : velocity field function.
        h         : integration step size.
        max_steps : maximum number of integration steps.
        tolerance : integration stops if the velocity magnitude falls below this.
        method    : integration method ('rk4' or 'euler').

    Returns:
        A list of length N, where each element is a (M_i, 3) array representing the computed streamline.
    """
    if seeds.ndim == 1:
        # If a single seed is provided, convert to shape (1, 3)
        seeds = seeds[np.newaxis, :]

    B = seeds.shape[0]
    # Initialize the trajectory for each seed.
    trajectories = [[seed] for seed in seeds]
    current_pos = seeds.copy()  # shape (B, 3)
    active = np.ones(B, dtype=bool)  # tracks which streamlines are still "active"

    # Select the appropriate integration function.
    if method == "rk4":
        step_func = rk4_step
    elif method == "euler":
        step_func = euler_step
    else:
        raise ValueError("Unknown integration method. Use 'rk4' or 'euler'.")

    for _ in range(max_steps):
        # Evaluate the velocity for all streamlines at once.
        v = f(current_pos)  # shape (B, 3)
        speeds = np.linalg.norm(v, axis=1)
        # Mark streamlines with low speed as finished.
        finished = (speeds < tolerance) & active
        active[finished] = False

        # If all streamlines have terminated, break early.
        if not np.any(active):
            break

        new_pos = step_func(f, current_pos, h)
        # For those streamlines that have finished, keep the position constant.
        new_pos[~active] = current_pos[~active]

        # Append the new positions into the trajectory for each seed.
        for i in range(B):
            trajectories[i].append(new_pos[i])
        current_pos = new_pos

    # Convert each trajectory into a NumPy array.
    trajectories = [np.array(traj) for traj in trajectories]
    return trajectories

def compute_streamlines_direction(seeds, f, h=0.1, max_steps=500, tolerance=1e-6, method="rk4", direction="forward"):
    """
    Computes streamlines for multiple seed points in the specified direction.

    Parameters:
        seeds     : (N, 3) array of seed points.
        f         : velocity field function.
        h         : integration step size.
        max_steps : maximum number of integration steps.
        tolerance : stops integration if the velocity magnitude falls below this.
        method    : integration method ('rk4' or 'euler').
        direction : 'forward', 'backward', or 'both'.
                    - 'backward' integrates using the negative velocity field and then reverses the result.
                    - 'both' concatenates the backward and forward streamlines (omitting the duplicate seed).

    Returns:
        A list of streamlines, each a (M_i, 3) numpy array.
    """
    if direction == "forward":
        return compute_streamlines(seeds, f, h, max_steps, tolerance, method)
    elif direction == "backward":

        def f_neg(x):
            return -f(x)

        backward_streams = compute_streamlines(seeds, f_neg, h, max_steps, tolerance, method)
        # Reverse each streamline for proper ordering.
        backward_streams = [stream[::-1] for stream in backward_streams]
        return backward_streams,velocity_stream
    elif direction == "both":
        forward_streams = compute_streamlines(seeds, f, h, max_steps, tolerance, method)

        def f_neg(x):
            return -f(x)

        backward_streams = compute_streamlines(seeds, f_neg, h, max_steps, tolerance, method)
        backward_streams = [stream[::-1] for stream in backward_streams]
        combined_streams = []
        for backward, forward in zip(backward_streams, forward_streams):
            # Omit the duplicate seed point
            combined = np.vstack([backward, forward[1:]])
            combined_streams.append(combined)
        return combined_streams
    else:
        raise ValueError("Invalid direction. Must be 'forward', 'backward', or 'both'.")

def sample_points_from_spheres(centers, radii, n_points_per_sphere, on_surface=True):
    """
    Samples points randomly from a collection of spheres.

    Each sphere is defined by its center (given in an array of shape (N, 3)) and radius.
    Returns an (N * n_points_per_sphere, 3) array of points.

    Parameters:
        centers             : (N, 3) numpy array of sphere centers.
        radii               : sequence of length N containing the radii for each sphere.
        n_points_per_sphere : int, number of points to sample per sphere.
        on_surface          : bool, if True sample on the sphere surface; if False, sample in the volume.

    Returns:
        seeds : (N * n_points_per_sphere, 3) numpy array of sampled points.
    """
    seeds = []
    for center, radius in zip(centers, radii):
        for _ in range(n_points_per_sphere):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * np.random.uniform() - 1)
            r = radius if on_surface else radius * (np.random.uniform() ** (1.0 / 3.0))
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            seeds.append(center + np.array([x, y, z]))
    return np.array(seeds)

def create_field_knn(points, velocities, k=6, p=2):
    """
    Creates an interpolated velocity field using a kd-tree based nearest-neighbor search
    and inverse-distance weighting. Now accepts both a single query (3,) or a batch (N, 3).

    Parameters:
        points     : (N, 3) array of sample locations.
        velocities : (N, 3) array of corresponding velocity vectors.
        k          : number of nearest neighbors to use.
        p          : exponent for inverse-distance weighting.

    Returns:
        field : a callable function that takes either a (3,) point or (N, 3) array of points and
                returns the interpolated velocity/velocities.
    """
    kd = cKDTree(points)

    def field(pos):
        if pos.ndim == 1:
            dists, idx = kd.query(pos, k=k)
            # Ensure arrays if a scalar is returned.
            if np.isscalar(dists):
                dists = np.array([dists])
                idx = np.array([idx])
            # If any distance is nearly zero, return the corresponding velocity directly.
            if np.any(dists < 1e-8):
                return velocities[idx[np.argmin(dists)]]
            weights = 1.0 / (dists**p)
            weights /= np.sum(weights)
            return np.dot(weights, velocities[idx])
        elif pos.ndim == 2:
            M = pos.shape[0]
            dists, idx = kd.query(pos, k=k)
            results = np.empty((M, 3))
            for i in range(M):
                if np.any(dists[i] < 1e-8):
                    results[i] = velocities[idx[i][np.argmin(dists[i])]]
                else:
                    weights = 1.0 / (dists[i] ** p)
                    weights /= np.sum(weights)
                    results[i] = np.dot(weights, velocities[idx[i]])
            return results
        else:
            raise ValueError("Input pos must be of shape (3,) or (N,3)")

    return field


def plot_streamlines(
    surface_position,
    volume_position_gt,
    volume_velocity_gt,
    volume_position_pred,
    volume_velocity_pred,
):
    car_mesh = pv.PolyData(surface_position.numpy())
    screenshot_gt = create_streamlines_screenshot(
        car_mesh=car_mesh,
        surface_position=surface_position,
        volume_position=volume_position_gt,
        volume_velocity=volume_velocity_gt,
    )
    screenshot_pred = create_streamlines_screenshot(
        car_mesh=car_mesh,
        surface_position=surface_position,
        volume_position=volume_position_pred,
        volume_velocity=volume_velocity_pred,
    )
    fig, ax = plt.subplots(ncols=2, figsize=(15, 10))
    ax[0].imshow(screenshot_gt)
    ax[0].axis('off')
    ax[0].set_title("HRLES")
    ax[1].imshow(screenshot_pred)
    ax[1].axis('off')
    ax[1].set_title("AB-UPT")
    plt.tight_layout()
    plt.show()

def create_streamlines_screenshot(car_mesh, surface_position, volume_position, volume_velocity):
    x_max_surface, y_max_surface, z_max_surface = surface_position.max(dim=0).values
    x_min_surface, y_min_surface, z_min_surface = surface_position.min(dim=0).values

    h = 0.001  # Step size
    max_steps = 200  # Maximum integration steps
    n = 15  # number of liner
    y_margin = -0.7  # distance from the car

    # make a 3D grid from where the streamlines start (i.e, a plane in front of the car)
    arr1 = np.array([x_min_surface - 0.2])
    arr2 = np.linspace(y_min_surface - y_margin, y_max_surface + y_margin, n)
    arr3 = np.linspace(z_min_surface, z_max_surface - 0.8, n)
    grid1, grid2, grid3 = np.meshgrid(arr1, arr2, arr3, indexing='ij')
    grid_front_of_the_car = np.stack((grid1, grid2, grid3), axis=-1)
    front_plane = grid_front_of_the_car.reshape(-1, 3)

    streamlines_front = compute_streamlines_direction(
        front_plane,
        create_field_knn(volume_position, volume_velocity, k=1, p=3),
        h,
        max_steps,
        tolerance=1e-6,
        method="rk4",
        direction="forward",
    )

    points = np.array(streamlines_front).reshape(-1, 3)
    lines = np.hstack(
        [
            [
                np.append(np.array([[len(stream_line)]]), np.arange(0, len(stream_line)) + (i * len(stream_line)))
                for i, stream_line in enumerate(streamlines_front)
            ]
        ],
    )
    streamlines_front = pv.PolyData(points, lines=lines)

    plotter = pv.Plotter(window_size=[1000, 500], notebook=True)
    plotter.add_mesh(car_mesh, color='gray', style='points', point_size=10)
    plotter.add_mesh(streamlines_front.tube(radius=.01), color='blue', opacity=.3)

    plotter.camera.azimuth = 90
    plotter.camera.elevation = 0
    plotter.enable_anti_aliasing()
    plotter.add_axes()

    screenshot = plotter.screenshot(return_img=True)
    return screenshot