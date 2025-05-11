from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import scatter_mean


class MPNN(nn.Module):
    """
    Message Passing Neural Network (see Gilmer et al., 2017).
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update

    Optimized version using fused operations and reduced memory allocations.
    """

    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        # Fuse message and update functions into single modules for better performance
        self.fused_message_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * dim + dimensionality, dim),
                    nn.GELU(),
                    nn.LayerNorm(
                        dim, elementwise_affine=False
                    ),  # Use non-affine LayerNorm for speed
                )
                for _ in range(mp_steps)
            ]
        )

        self.update_fns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * dim, dim),
                    nn.LayerNorm(
                        dim, elementwise_affine=False
                    ),  # Use non-affine LayerNorm for speed
                )
                for _ in range(mp_steps)
            ]
        )

    def layer(
        self,
        message_fn: nn.Module,
        update_fn: nn.Module,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        row, col = edge_index
        # Pre-allocate memory for messages
        messages = torch.cat([h[row], h[col], edge_attr], dim=-1)
        messages = message_fn(messages)
        message = scatter_mean(messages, col, h.size(0))
        # Fuse concatenation and update
        update = update_fn(torch.cat([h, message], dim=-1))
        h.add_(update)  # In-place addition
        return h

    @torch.no_grad()
    def compute_edge_attr(self, pos, edge_index):
        # Compute edge attributes directly without intermediate allocations
        return pos.index_select(0, edge_index[0]) - pos.index_select(0, edge_index[1])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        edge_attr = self.compute_edge_attr(pos, edge_index)
        for message_fn, update_fn in zip(self.fused_message_update, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)
        return x
