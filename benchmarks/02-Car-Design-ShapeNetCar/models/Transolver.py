import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange, repeat

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


from .components import ErwinFlashTransformer as ErwinTransformer

class ErwinTransolver(nn.Module):
    """Combines Transolver++'s slicing with adaptive temperature and eidetic states with Erwin's hierarchical processing.
    Implements the Transolver++ algorithm (Algorithm 1) with Eidetic States, using Rep-Slice and Ada-Temp mechanisms.
    """
    def __init__(
        self, 
        dim: int,
        slice_num: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dimensionality: int = 3,
        base_temp: float = 0.5,
        epsilon: float = 1e-6,  # For the log(-log(ε)) term in Rep-Slice
        radius: float = 1.0     # Add radius parameter with default value
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.dimensionality = dimensionality
        self.epsilon = epsilon
        self.radius = radius    # Store the radius parameter
        
        # Input projections for slicing (just x, not fx to save 50% memory as per algorithm)
        self.in_project_x = nn.Linear(dim, inner_dim)

        # Input positions for slicing - learn position representations
        self.pos_projector = nn.Sequential(
            nn.Linear(dimensionality, dimensionality * heads),
            nn.GELU(),
            nn.Linear(dimensionality * heads, heads * self.dimensionality),
        )
        
        # Rep-Slice projection
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.in_project_slice._is_rep_slice = True  # Tag for custom init
        nn.init.orthogonal_(self.in_project_slice.weight)
        
        # Ada-Temp: Base temperature + adaptive component
        self.base_temp = base_temp
        self.ada_temp_norm = nn.LayerNorm(dim_head)
        self.ada_temp_linear = nn.Linear(dim_head, 1)  # Adaptive temperature adjustment
        self.ada_temp_linear._is_ada_temp = True
        
        # Erwin network for processing eidetic states
        self.erwin = ErwinTransformer(
            c_in=dim_head,
            c_hidden=[dim_head, dim_head*2],  # Two levels of hierarchy
            ball_sizes=[min(32, slice_num), min(16, slice_num//2)],  # Progressive reduction
            enc_num_heads=[heads, heads],
            enc_depths=[4, 4],
            dec_num_heads=[heads],
            dec_depths=[4],
            strides=[2],
            rotate=45,  # Enable rotation for better cross-token mixing
            decode=True,  # We need the full resolution back
            mlp_ratio=4,
            dimensionality=dimensionality,
            mp_steps=0  # No need for MPNN here
        )
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Initialize slice weights attribute
        self.slice_weights = None

    def gumbel_softmax_sample(self, logits, tau, training=True):
        """Gumbel-Softmax sampling as in Eq. (4)"""
        if training:
            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            return torch.softmax((logits + gumbel_noise) / tau, dim=-1)
        else:
            return torch.softmax(logits / tau, dim=-1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        Implements Transolver++ Algorithm 1: Parallel Physics-Attention with Eidetic States
        """
        B, N, C = x.shape

        # Project input
        x_proj = self.in_project_x(x).view(B, N, self.heads, self.dim_head).transpose(1, 2)  # [B, H, N, D]
        pos_proj = self.pos_projector(pos).view(B, N, self.heads, self.dimensionality).transpose(1, 2)  # [B, H, N, D]

        # Adaptive temperature
        tau = torch.clamp(self.base_temp + self.ada_temp_linear(self.ada_temp_norm(x_proj)), min=0.1, max=2.0)

        # Rep-Slice: Get raw logits and apply Gumbel-Softmax
        raw_logits = self.in_project_slice(x_proj)  # [B, H, N, G]
        slice_weights = self.gumbel_softmax_sample(raw_logits, tau, self.training)  # [B, H, N, G]

        if not self.training:
            self.slice_weights = slice_weights  # Save for inspection

        # Normalize weights (for eidetic state computation)
        norm = slice_weights.sum(2, keepdim=True)  # [B, H, 1, G]

        # Eidetic state computation
        eidetic_states = torch.matmul(slice_weights.transpose(-2, -1), x_proj) / (norm.transpose(-1, -2) + 1e-3)  # Increased epsilon
        eidetic_pos = torch.matmul(slice_weights.transpose(-2, -1), pos_proj) / (norm.transpose(-1, -2) + 1e-3)

        # Prepare for Erwin
        B, H, G, D = eidetic_states.shape
        states_flat = eidetic_states.reshape(B * H * G, D)
        pos_flat = eidetic_pos.reshape(B * H * G, self.dimensionality)
        batch_idx = torch.arange(B * H, device=x.device).repeat_interleave(G)

        # Erwin attention
        updated = self.erwin(states_flat, pos_flat, batch_idx, radius=self.radius)
        updated = updated.view(B, H, G, D)

        # Deslice
        out = torch.matmul(slice_weights, updated)  # [B, H, N, G] @ [B, H, G, D] = [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)  # [B, N, C]
        return self.to_out(out)

    def get_slice_weights(self):
        """Return the slice weights from the latest forward pass."""
        return self.slice_weights


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            radius=1.0
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = ErwinTransolver(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                     dropout=dropout, slice_num=slice_num, radius=radius)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, pos=None):
        fx = self.Attn(self.ln_1(fx), pos) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx
    
    def get_slice_weights(self):
        """Return the slice weights from the attention module."""
        return self.Attn.get_slice_weights()


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 radius=1.0
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + space_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      radius=radius,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, 'weight') and m.weight is not None:
                if getattr(m, '_is_rep_slice', False):
                    nn.init.orthogonal_(m.weight)  # Special init for Rep-Slice
                elif getattr(m, '_is_ada_temp', False):  # Add this
                    nn.init.zeros_(m.weight)  # Start with no temperature adjustment
                else:
                    trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        cfd_data, geom_data = data
        x, fx, T = cfd_data.x, None, None
        x = x[None, :, :]
        
        # Store original position coordinates for use in attention
        original_pos = cfd_data.pos[None, :, :] if hasattr(cfd_data, 'pos') and cfd_data.pos is not None else None
        
        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx, original_pos)

        return fx[0]
        
    def get_last_block_slice_weights(self):
        """Return the slice weights from the last transformer block."""
        return self.blocks[-1].get_slice_weights()
