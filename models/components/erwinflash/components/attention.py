from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_qkvpacked_func


class BallMSA(nn.Module):
    """Ball Multi-Head Self-Attention (BMSA) module (eq. 8) using Flash Attention with packed QKV."""

    def __init__(
        self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """Distance-based attention bias (eq. 10)."""
        pos = rearrange(pos, "(n m) d -> n m d", m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """Relative position of leafs wrt the center of the ball (eq. 9)."""
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        qkv_weight = self.qkv.weight.to(torch.float16)
        qkv_bias = (
            self.qkv.bias.to(torch.float16) if self.qkv.bias is not None else None
        )
        proj_weight = self.proj.weight.to(torch.float16)
        proj_bias = (
            self.proj.bias.to(torch.float16) if self.proj.bias is not None else None
        )
        pe_weight = self.pe_proj.weight.to(torch.float16)
        pe_bias = (
            self.pe_proj.bias.to(torch.float16)
            if self.pe_proj.bias is not None
            else None
        )

        # Compute positional encoding in fp16
        rel_pos = self.compute_rel_pos(pos)
        pe = F.linear(rel_pos, pe_weight, pe_bias)
        x = x + pe

        # QKV projection in fp16
        qkv = F.linear(x, qkv_weight, qkv_bias)

        # Reshape for flash attention with packed QKV
        B = x.shape[0] // self.ball_size  # number of balls
        qkv = rearrange(
            qkv,
            "(b n) (three h d) -> b n three h d",
            b=B,
            n=self.ball_size,
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        # Scale the QK product
        softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # Flash attention with packed QKV
        x = flash_attn_qkvpacked_func(
            qkv, dropout_p=0.0, softmax_scale=softmax_scale, causal=False
        )  # [B, ball_size, H, head_dim]

        # Reshape output and project
        x = rearrange(x, "b n h d -> (b n) (h d)")
        x = F.linear(x, proj_weight, proj_bias)

        # Convert back to original dtype
        return x
