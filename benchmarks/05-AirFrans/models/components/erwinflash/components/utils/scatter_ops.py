import torch
import torch_scatter


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """
    Averages all values from src into the receivers at the indices specified by idx.
    Uses torch_scatter's optimized implementation.

    Args:
        src (torch.Tensor): Source tensor of shape (N, D).
        idx (torch.Tensor): Indices tensor of shape (N,).
        num_receivers (int): Number of receivers (usually the maximum index in idx + 1).

    Returns:
        torch.Tensor: Result tensor of shape (num_receivers, D).
    """
    return torch_scatter.scatter_mean(src, idx, dim=0, dim_size=num_receivers)
