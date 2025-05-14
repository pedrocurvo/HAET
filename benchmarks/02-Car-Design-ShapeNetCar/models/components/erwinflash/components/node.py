from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Node:
    """Dataclass to store the hierarchical node information."""

    x: torch.Tensor
    pos: torch.Tensor
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: Node | None = None
