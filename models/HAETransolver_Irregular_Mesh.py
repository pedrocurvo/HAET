"""
Transolver model implementation for irregular mesh data structures.
This module provides a transformer-based solver architecture 
specifically designed to handle irregular meshes in 1D.
"""

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .components import MLP, timestep_embedding
from .PhysicsAttention import Physics_Attention_Irregular_Mesh


class TransolverErwinBlock(nn.Module):
    """Transformer encoder block for irregular mesh processing.

    This block consists of a physics-informed attention mechanism followed by
    an MLP, both with residual connections and layer normalization.

    Attributes:
        last_layer (bool): Flag indicating if this is the final layer in the network
        ln_1 (nn.LayerNorm): Layer normalization before attention
        Attn (Physics_Attention_Irregular_Mesh): Physics-informed attention mechanism
        ln_2 (nn.LayerNorm): Layer normalization before MLP
        mlp (MLP): Multi-layer perceptron for feature transformation
        ln_3 (nn.LayerNorm): Layer normalization in the last layer (if applicable)
        mlp2 (nn.Linear): Final projection layer (if last_layer is True)
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        radius=1.0,
        dimensionality=1,
        # ErwinTransformer parameters
        c_hidden=None,
        ball_sizes=None,
        enc_num_heads=None,
        enc_depths=None,
        dec_num_heads=None,
        dec_depths=None,
        strides=None,
        rotate=45,
        decode=True,
        mp_steps=0,
        embed=False,
    ):
        """Initialize a Transolver block for irregular meshes.

        Args:
            num_heads: Number of attention heads
            hidden_dim: Dimension of hidden features
            dropout: Dropout rate
            act: Activation function used in MLP
            mlp_ratio: Expansion ratio for hidden dimension in MLP
            last_layer: Whether this is the final layer in the network
            out_dim: Output dimension (only used if last_layer=True)
            slice_num: Number of slices for attention computation
            c_hidden: Hidden channel dimensions for each hierarchical level
            ball_sizes: Ball sizes for each hierarchical level
            enc_num_heads: Number of attention heads for each encoder level
            enc_depths: Depth of each encoder level
            dec_num_heads: Number of attention heads for each decoder level
            dec_depths: Depth of each decoder level
            strides: Stride values for each level
            rotate: Rotate flag for geometric awareness
            decode: Whether to decode/upsample back to original resolution
            mp_steps: Number of message passing steps
            embed: Whether to use ErwinEmbedding
        """
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            radius=radius,
            dimensionality=dimensionality,
            # Pass the ErwinTransformer parameters
            c_hidden=c_hidden,
            ball_sizes=ball_sizes,
            enc_num_heads=enc_num_heads,
            enc_depths=enc_depths,
            dec_num_heads=dec_num_heads,
            dec_depths=dec_depths,
            strides=strides,
            rotate=rotate,
            decode=decode,
            mlp_ratio=mlp_ratio,
            mp_steps=mp_steps,
            embed=embed,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, pos=None):
        """Forward pass through the Transolver block.

        Args:
            fx: Input feature tensor [batch_size, num_points, hidden_dim]

        Returns:
            Processed feature tensor with the same shape as input,
            or [batch_size, num_points, out_dim] if last_layer=True
        """
        # Apply attention with residual connection
        fx = self.Attn(self.ln_1(fx), pos) + fx

        # Apply MLP with residual connection
        fx = self.mlp(self.ln_2(fx)) + fx

        # Apply final projection if this is the last layer
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    """Transolver model for irregular mesh data.

    This model uses a transformer-based architecture with physics-informed attention
    to process irregular mesh data. It can handle spatial coordinates and optional
    time inputs for time-dependent problems.

    Attributes:
        __name__ (str): Model identifier
        ref (int): Reference grid resolution
        unified_pos (bool): Whether to use unified position encoding
        Time_Input (bool): Whether time is included as an input
        n_hidden (int): Hidden dimension size
        space_dim (int): Spatial dimension
        preprocess (MLP): Preprocessing MLP for input features
        time_fc (nn.Sequential): Time embedding network (if Time_Input=True)
        blocks (nn.ModuleList): List of Transolver blocks
        placeholder (nn.Parameter): Learnable placeholder parameter
    """

    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        # ErwinTransformer parameters
        radius=1.0,
        c_hidden=None,
        ball_sizes=None,
        enc_num_heads=None,
        enc_depths=None,
        dec_num_heads=None,
        dec_depths=None,
        strides=None,
        rotate=45,
        decode=True,
        mp_steps=0,
        embed=False,
    ):
        """Initialize the Transolver model for irregular meshes.

        Args:
            space_dim: Dimension of spatial coordinates (1D, 2D, etc.)
            n_layers: Number of Transolver blocks
            n_hidden: Hidden dimension size
            dropout: Dropout rate
            n_head: Number of attention heads
            Time_Input: Whether to include time as an input
            act: Activation function
            mlp_ratio: Expansion ratio for hidden dimension in MLP
            fun_dim: Dimension of input function values
            out_dim: Dimension of output
            slice_num: Number of slices for attention computation
            ref: Reference grid resolution
            unified_pos: Whether to use unified position encoding
            c_hidden: Hidden channel dimensions for each hierarchical level in ErwinTransformer
            ball_sizes: Ball sizes for each hierarchical level in ErwinTransformer
            enc_num_heads: Number of attention heads for each encoder level in ErwinTransformer
            enc_depths: Depth of each encoder level in ErwinTransformer
            dec_num_heads: Number of attention heads for each decoder level in ErwinTransformer
            dec_depths: Depth of each decoder level in ErwinTransformer
            strides: Stride values for each level in ErwinTransformer
            rotate: Rotate flag for geometric awareness in ErwinTransformer
            decode: Whether to decode/upsample back to original resolution
            mp_steps: Number of message passing steps in ErwinTransformer
            embed: Whether to use ErwinEmbedding in ErwinTransformer
        """
        super(Model, self).__init__()
        self.__name__ = "Transolver_1D"
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if self.unified_pos:
            self.preprocess = MLP(
                fun_dim + self.ref ** space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.preprocess = MLP(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        if Time_Input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

        self.blocks = nn.ModuleList(
            [
                TransolverErwinBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(_ == n_layers - 1),
                    dimensionality=space_dim,
                    # Pass the ErwinTransformer parameters
                    radius=radius,
                    c_hidden=c_hidden,
                    ball_sizes=ball_sizes,
                    enc_num_heads=enc_num_heads,
                    enc_depths=enc_depths,
                    dec_num_heads=dec_num_heads,
                    dec_depths=dec_depths,
                    strides=strides,
                    rotate=rotate,
                    decode=decode,
                    mp_steps=mp_steps,
                    embed=embed,
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float)
        )

    def initialize_weights(self):
        """Initialize model weights using the _init_weights method."""
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

    def get_grid(self, x, batchsize=1):
        """Generate a reference grid and compute distances to input points.

        This method creates a regular reference grid within [0,1] × [0,1], and then
        computes the Euclidean distances between each input point and each point
        in the reference grid.

        Args:
            x: Input point coordinates tensor of shape [batch_size, num_points, coordinate_dim]
            batchsize: Batch size for the generated grid

        Returns:
            Tensor of distances between input points and reference grid points
            Shape: [batch_size, num_points, ref*ref]
        """
        # Create reference grid in x direction (gridx shape: [1, ref, 1, 1])
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])

        # Create reference grid in y direction (gridy shape: [1, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])

        # Combine into a single grid (shape: [batchsize, ref*ref, 2])
        grid_ref = (
            torch.cat((gridx, gridy), dim=-1)
            .cuda()
            .reshape(batchsize, self.ref * self.ref, 2)
        )

        # Compute distances between input points and reference grid points
        # Result shape: [batchsize, num_points, ref*ref]
        pos = (
            torch.sqrt(
                torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)
            )
            .reshape(batchsize, x.shape[1], self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, x, fx, T=None):
        """Forward pass of the Transolver model.

        Args:
            x: Spatial coordinates tensor of shape [batch_size, num_points, space_dim]
            fx: Function values tensor of shape [batch_size, num_points, fun_dim]
               or None if no function values are provided
            T: Time values tensor of shape [batch_size, 1] or None for time-independent problems

        Returns:
            Output tensor of shape [batch_size, num_points, out_dim]
        """
        # Original positons
        original_pos = x
        # Apply position encoding if unified_pos is enabled
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])

        # Preprocess input features
        if fx is not None:
            # Combine spatial coordinates with function values
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            # Use only spatial coordinates
            fx = self.preprocess(x)

        # Add learnable placeholder parameter
        fx = fx + self.placeholder[None, None, :]

        # Add time embedding if time input is provided
        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        # Process through Transolver blocks
        for block in self.blocks:
            fx = block(fx, pos=original_pos)

        return fx
    
    def get_last_block_slice_weights(self):
        """Return the slice weights from the last transformer block."""
        return self.blocks[-1].get_slice_weights()
