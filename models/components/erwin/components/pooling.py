from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from einops import rearrange, reduce

from typing import Literal, List
from dataclasses import dataclass
from .node import Node
from balltree import build_balltree_with_rotations


class BallPooling(nn.Module):
    """
    Pooling of leaf nodes in a ball (eq. 12):
        1. select balls of size 'stride'.
        2. concatenate leaf nodes inside each ball along with their relative positions to the ball center.
        3. apply linear projection and batch normalization.
        4. the output is the center of each ball endowed with the pooled features.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(stride * in_dim + stride * dimensionality, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        if self.stride == 1:  # no pooling
            return Node(x=node.x, pos=node.pos, batch_idx=node.batch_idx, children=node)

        with torch.no_grad():
            batch_idx = node.batch_idx[:: self.stride]
            centers = reduce(node.pos, "(n s) d -> n d", "mean", s=self.stride)
            pos = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)
            rel_pos = rearrange(pos - centers[:, None], "n s d -> n (s d)")

        x = torch.cat(
            [rearrange(node.x, "(n s) c -> n (s c)", s=self.stride), rel_pos], dim=1
        )
        x = self.norm(self.proj(x))

        return Node(x=x, pos=centers, batch_idx=batch_idx, children=node)


class BallUnpooling(nn.Module):
    """
    Ball unpooling (refinement; eq. 13):
        1. compute relative positions of children (from before pooling) to the center of the ball.
        2. concatenate the pooled features with the relative positions.
        3. apply linear projection and self-connection followed by batch normalization.
        4. the output is a refined tree with the same number of nodes as before pooling.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_dim + stride * dimensionality, stride * out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            rel_pos = (
                rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride)
                - node.pos[:, None]
            )
            rel_pos = rearrange(rel_pos, "n m d -> n (m d)")

        x = torch.cat([node.x, rel_pos], dim=-1)
        node.children.x = self.norm(
            node.children.x
            + rearrange(self.proj(x), "n (m d) -> (n m) d", m=self.stride)
        )

        return node.children
