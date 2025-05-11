import torch
import torch.nn as nn
from einops import rearrange

from ..components import ErwinTransformer


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        H=101,
        W=31,
        kernel=3,
    ):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.dimensionality = 2  # 2D space

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        # Replace with more sophisticated hierarchical ErwinTransformer
        self.erwin = ErwinTransformer(
            c_in=dim_head,
            c_hidden=[
                dim_head,
                dim_head * 2,
            ],  # Two levels of hierarchy with increasing channels
            ball_sizes=[
                min(32, slice_num),
                min(16, slice_num // 2),
            ],  # Progressive reduction
            enc_num_heads=[heads // 2, heads],  # Increasing heads in deeper layers
            enc_depths=[2, 2],  # Equal depth at each level
            dec_num_heads=[heads // 2],  # Matching encoder at the same level
            dec_depths=[2],  # Matching encoder depth
            strides=[2],  # Downsample by factor of 2
            rotate=1,  # Enable rotation for better cross-token mixing
            decode=True,  # We need the full resolution back
            mlp_ratio=4,  # Standard MLP expansion ratio
            dimensionality=self.dimensionality,
            mp_steps=0,  # No need for MPNN here
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = (
            x.reshape(B, self.H, self.W, C)
            .contiguous()
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # B C H W

        ### (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Transform with ErwinTransformer instead of attention
        # Process each batch and head independently
        B, H, G, C = slice_token.shape

        # Reshape for Erwin: [B, H, G, C] -> [B*H*G, C]
        slice_token_flat = slice_token.reshape(
            B * H * G, C
        )  # Flatten to [total_points, channels]

        # Create artificial positions for slice tokens in 2D space (for Structured_Mesh_2D)
        pos = torch.rand(
            B * H * G, self.dimensionality, device=slice_token.device
        )  # [total_points, 2]

        # Create batch indices - each batch and head combination gets its own batch index
        batch_idx = torch.arange(B * H, device=slice_token.device).repeat_interleave(G)

        # Process through ErwinTransformer - it expects [num_points, channels] for features
        processed_tokens = self.erwin(slice_token_flat, pos, batch_idx)

        # Reshape back to original format [B, H, G, C]
        out_slice_token = processed_tokens.reshape(B, H, G, C)

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)
