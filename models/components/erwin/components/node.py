from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from einops import rearrange, reduce

from typing import Literal, List
from dataclasses import dataclass

from balltree import build_balltree_with_rotations

@dataclass
class Node:
    """ Dataclass to store the hierarchical node information."""
    x: torch.Tensor
    pos: torch.Tensor 
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: Node | None = None