from functools import partial

import einops
import torch
from torch import nn

from modules.attention import (
    AnchorAttention,
    SharedweightsCrossattnAttention,
    SharedweightsSplitattnAttention,
)
from modules.blocks import TransformerBlock, PerceiverBlock
from modules.continuous_sincos_embed import ContinuousSincosEmbed
from modules.rope_frequency import RopeFrequency
from modules.supernode_pooling_posonly import SupernodePoolingPosonly

# import ErwinTransformer from your implementation
from erwin import ErwinTransformer


class AnchoredBranchedUPT(nn.Module):
    def __init__(
        self,
        ndim: int = 3,
        input_dim: int = 3,
        output_dim_surface: int = 4,
        output_dim_volume: int = 7,
        dim: int = 192,
        geometry_depth: int = 1,
        num_heads: int = 3,
        blocks: str = "pscscs",
        num_volume_blocks: int = 6,
        num_surface_blocks: int = 6,
        radius: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rope = RopeFrequency(dim=dim // num_heads, ndim=input_dim)
        # geometry encoder
        self.encoder = SupernodePoolingPosonly(
            hidden_dim=dim,
            ndim=ndim,
            radius=radius,
            mode="relpos",
        )
        self.geometry_blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads) for _ in range(geometry_depth)]
        )
        # pos_embed with separate MLP for surface/volume
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.surface_bias = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.volume_bias = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        # weight-shared blocks
        self.blocks = nn.ModuleList()
        for block in blocks:
            if block == "s":
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsSplitattnAttention)
            elif block == "c":
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsCrossattnAttention)
            elif block == "p":
                block_ctor = PerceiverBlock
            else:
                raise NotImplementedError
            self.blocks.append(block_ctor(dim=dim, num_heads=num_heads))

        # replace surface/volume modality-specific blocks with ErwinTransformer
        erwin_common = dict(
            c_in=dim,
            c_hidden=[dim, dim],
            ball_sizes=[16, 16],
            enc_num_heads=[num_heads, num_heads],
            enc_depths=[2, 1],
            dec_num_heads=[num_heads],
            dec_depths=[1],
            strides=[4],
            rotate=0,
            decode=True,
            mp_steps=0,
            embed=False,
        )
        self.surface_erwin = ErwinTransformer(**erwin_common)
        self.volume_erwin = ErwinTransformer(**erwin_common)

        # final linear projections
        self.surface_decoder = nn.Linear(dim, output_dim_surface)
        self.volume_decoder = nn.Linear(dim, output_dim_volume)

        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(
        self,
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor | None,
        surface_anchor_position: torch.Tensor,
        volume_anchor_position: torch.Tensor,
        surface_query_position: torch.Tensor | None = None,
        volume_query_position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = {}
        geometry_attn_kwargs = {}
        surface_decoder_attn_kwargs = {}
        volume_decoder_attn_kwargs = {}
        geometry_perceiver_attn_kwargs = {}
        shared_attn_kwargs = {}

        num_surface_anchor_positions = surface_anchor_position.size(1)
        num_volume_anchor_positions = volume_anchor_position.size(1)
        if surface_query_position is None and volume_query_position is None:
            split_size = [surface_anchor_position.size(1), volume_anchor_position.size(1)]
            surface_position_all = surface_anchor_position
            volume_position_all = volume_anchor_position
        elif surface_query_position is not None and volume_query_position is not None:
            split_size = [
                surface_anchor_position.size(1),
                surface_query_position.size(1),
                volume_anchor_position.size(1),
                volume_query_position.size(1),
            ]
            surface_decoder_attn_kwargs["num_anchor_tokens"] = surface_anchor_position.size(1)
            volume_decoder_attn_kwargs["num_anchor_tokens"] = volume_anchor_position.size(1)
            surface_position_all = torch.concat(
                [surface_anchor_position, surface_query_position], dim=1
            )
            volume_position_all = torch.concat(
                [volume_anchor_position, volume_query_position], dim=1
            )
        else:
            raise NotImplementedError

        assert geometry_batch_idx is None or geometry_batch_idx.unique().numel() == 1
        geometry_rope = self.rope(geometry_position[geometry_supernode_idx].unsqueeze(0))
        geometry_attn_kwargs["freqs"] = geometry_rope
        rope_surface_all = self.rope(surface_position_all)
        rope_volume_all = self.rope(volume_position_all)
        rope_all = torch.concat([rope_surface_all, rope_volume_all], dim=1)
        surface_decoder_attn_kwargs["freqs"] = rope_surface_all
        geometry_perceiver_attn_kwargs["q_freqs"] = rope_all
        geometry_perceiver_attn_kwargs["k_freqs"] = geometry_rope
        volume_decoder_attn_kwargs["freqs"] = rope_volume_all
        shared_attn_kwargs["freqs"] = rope_all

        # geometry encoding
        x = self.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
        )
        for block in self.geometry_blocks:
            x = block(x, attn_kwargs=geometry_attn_kwargs)
        geometry_encoding = x

        # shared-weight trunk
        surface_all_pos_embed = self.surface_bias(self.pos_embed(surface_position_all))
        volume_all_pos_embed = self.volume_bias(self.pos_embed(volume_position_all))
        x = torch.concat([surface_all_pos_embed, volume_all_pos_embed], dim=1)
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                x = block(x, attn_kwargs=dict(split_size=split_size, **shared_attn_kwargs))
            elif isinstance(block, PerceiverBlock):
                x = block(q=x, kv=geometry_encoding, attn_kwargs=geometry_perceiver_attn_kwargs)
            else:
                raise NotImplementedError

        # split surface/volume
        if surface_query_position is None and volume_query_position is None:
            x_surface, x_volume = x.split(split_size, dim=1)
        elif surface_query_position is not None and volume_query_position is not None:
            x_surface, x_volume = x.split(
                [split_size[0] + split_size[1], split_size[2] + split_size[3]], dim=1
            )
        else:
            raise NotImplementedError

        # === SURFACE with Erwin ===
        bs, S, dim = x_surface.shape
        surface_feats = x_surface.reshape(-1, dim)
        surface_pos = surface_position_all.reshape(-1, surface_position_all.size(-1))
        surface_batch_idx = torch.arange(bs, device=x_surface.device).repeat_interleave(S)
        surface_out = self.surface_erwin(
            node_features=surface_feats,
            node_positions=surface_pos,
            batch_idx=surface_batch_idx,
            radius=0.25,
        )
        surface_out = surface_out.view(bs, S, -1)
        x = self.surface_decoder(surface_out)

        if surface_query_position is None:
            outputs["surface_anchor_pressure"] = einops.rearrange(x[:, :, :1], "bs seqlen dim -> (bs seqlen) dim")
            outputs["surface_anchor_wallshearstress"] = einops.rearrange(x[:, :, 1:], "bs seqlen dim -> (bs seqlen) dim")
        else:
            x_surface_anchor = x[:, :num_surface_anchor_positions]
            x_surface_query = x[:, num_surface_anchor_positions:]
            outputs["surface_anchor_pressure"] = einops.rearrange(
                x_surface_anchor[:, :, :1], "bs seqlen dim -> (bs seqlen) dim"
            )
            outputs["surface_anchor_wallshearstress"] = einops.rearrange(
                x_surface_anchor[:, :, 1:], "bs seqlen dim -> (bs seqlen) dim"
            )
            outputs["surface_query_pressure"] = einops.rearrange(
                x_surface_query[:, :, :1], "bs seqlen dim -> (bs seqlen) dim"
            )
            outputs["surface_query_wallshearstress"] = einops.rearrange(
                x_surface_query[:, :, 1:], "bs seqlen dim -> (bs seqlen) dim"
            )

        # === VOLUME with Erwin ===
        bs, V, dim = x_volume.shape
        volume_feats = x_volume.reshape(-1, dim)
        volume_pos = volume_position_all.reshape(-1, volume_position_all.size(-1))
        volume_batch_idx = torch.arange(bs, device=x_volume.device).repeat_interleave(V)
        volume_out = self.volume_erwin(
            node_features=volume_feats,
            node_positions=volume_pos,
            batch_idx=volume_batch_idx,
            radius=0.25,
        )
        volume_out = volume_out.view(bs, V, -1)
        x = self.volume_decoder(volume_out)

        if volume_query_position is None:
            outputs["volume_anchor_totalpcoeff"] = einops.rearrange(x[:, :, :1], "bs seqlen dim -> (bs seqlen) dim")
            outputs["volume_anchor_velocity"] = einops.rearrange(x[:, :, 1:4], "bs seqlen dim -> (bs seqlen) dim")
            outputs["volume_anchor_vorticity"] = einops.rearrange(x[:, :, 4:], "bs seqlen dim -> (bs seqlen) dim")
        else:
            x1 = x[:, :num_volume_anchor_positions]
            x2 = x[:, num_volume_anchor_positions:]
            outputs["volume_anchor_totalpcoeff"] = einops.rearrange(x1[:, :, :1], "bs seqlen dim -> (bs seqlen) dim")
            outputs["volume_anchor_velocity"] = einops.rearrange(x1[:, :, 1:4], "bs seqlen dim -> (bs seqlen) dim")
            outputs["volume_anchor_vorticity"] = einops.rearrange(x1[:, :, 4:], "bs seqlen dim -> (bs seqlen) dim")
            outputs["volume_query_totalpcoeff"] = einops.rearrange(x2[:, :, :1], "bs seqlen dim -> (bs seqlen) dim")
            outputs["volume_query_velocity"] = einops.rearrange(x2[:, :, 1:4], "bs seqlen dim -> (bs seqlen) dim")
            outputs["volume_query_vorticity"] = einops.rearrange(x2[:, :, 4:], "bs seqlen dim -> (bs seqlen) dim")

        return outputs


def main():
    torch.manual_seed(0)
    num_geometry_positions = 655
    num_geometry_supernodes = 123
    num_surface_anchors = 234
    num_volume_anchors = 280
    num_surface_queries = 301
    num_volume_queries = 321
    data = dict(
        geometry_position=torch.rand(num_geometry_positions, 3) * 1000,
        geometry_supernode_idx=torch.randperm(num_geometry_positions)[:num_geometry_supernodes],
        geometry_batch_idx=None,
        surface_anchor_position=torch.rand(1, num_surface_anchors, 3) * 1000,
        volume_anchor_position=torch.rand(1, num_volume_anchors, 3) * 1000,
        surface_query_position=torch.rand(1, num_surface_queries, 3) * 1000,
        volume_query_position=torch.rand(1, num_volume_queries, 3) * 1000,
    )
    model = AnchoredBranchedUPT()
    outputs = model(**data)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")


if __name__ == "__main__":
    main()