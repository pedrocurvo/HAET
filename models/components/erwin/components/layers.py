from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from components.mpnn import MPNN
from components.pooling import BallPooling, BallUnpooling
from components.node import Node
from components.attention import BallMSA

class SwiGLU(nn.Module):
    """ W_3 SiLU(W_1 x) ⊗ W_2 x """
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)
    
    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class ErwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ball_size: int, mlp_ratio: int, dimensionality: int = 3):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.BMSA = BallMSA(dim, num_heads, ball_size, dimensionality)
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.BMSA(self.norm1(x), pos)
        return x + self.swiglu(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        direction: Literal['down', 'up', None], # down: encoder, up: decoder, None: bottleneck
        depth: int,
        stride: int,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        rotate: bool,
        dimensionality: int = 3,

    ):
        super().__init__()
        hidden_dim = in_dim if direction == 'down' else out_dim

        self.blocks = nn.ModuleList([ErwinTransformerBlock(hidden_dim, num_heads, ball_size, mlp_ratio, dimensionality) for _ in range(depth)])
        self.rotate = [i % 2 for i in range(depth)] if rotate else [False] * depth

        self.pool = lambda node: node
        self.unpool = lambda node: node

        if direction == 'down' and stride is not None:
            self.pool = BallPooling(hidden_dim, out_dim, stride, dimensionality)
        elif direction == 'up' and stride is not None:
            self.unpool = BallUnpooling(in_dim, hidden_dim, stride, dimensionality)

    def forward(self, node: Node) -> Node:
        node = self.unpool(node)

        if len(self.rotate) > 1 and self.rotate[1]: # if rotation is enabled, it will be used in the second block
            assert node.tree_idx_rot is not None, "tree_idx_rot must be provided for rotation"
            tree_idx_rot_inv = torch.argsort(node.tree_idx_rot) # map from rotated to original

        for rotate, blk in zip(self.rotate, self.blocks):
            if rotate:
                node.x = blk(node.x[node.tree_idx_rot], node.pos[node.tree_idx_rot])[tree_idx_rot_inv]
            else:
                node.x = blk(node.x, node.pos)
        return self.pool(node)