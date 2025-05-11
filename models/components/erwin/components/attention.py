from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from einops import rearrange, reduce

class BallMSA(nn.Module):
    """ Ball Multi-Head Self-Attention (BMSA) module (eq. 8). """
    def __init__(self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """ Distance-based attention bias (eq. 10). """
        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """ Relative position of leafs wrt the center of the ball (eq. 9). """
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        q, k, v = rearrange(self.qkv(x), "(n m) (H E K) -> K n H m E", H=self.num_heads, m=self.ball_size, K=3)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=self.create_attention_mask(pos))
        x = rearrange(x, "n H m E -> (n m) (H E)", H=self.num_heads, m=self.ball_size)
        return self.proj(x)