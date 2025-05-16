"""
Transolver model implementation for 3D structured mesh data.
This module provides a transformer-based solver architecture 
that leverages the volumetric grid structure of 3D structured meshes
for efficient computation of physical quantities in three dimensions.
The implementation supports gradient checkpointing for memory efficiency 
when handling large 3D volumes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_

from .components import MLP, timestep_embedding
from .PhysicsAttention import Physics_Attention_Structured_Mesh_3D


class TransolverErwinBlock(nn.Module):
    """Transformer encoder block for 3D structured mesh data.

    This block consists of a physics-informed attention mechanism tailored for
    3D structured meshes, followed by an MLP. Both components have residual
    connections and layer normalization.

    Attributes:
        last_layer (bool): Flag indicating if this is the final layer in the network
        ln_1 (nn.LayerNorm): Layer normalization before attention
        Attn (Physics_Attention_Structured_Mesh_3D): Physics-informed attention for 3D structured meshes
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
        H: int = 32,
        W: int = 32,
        D: int = 32,
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
        """Initialize a Transolver block for 3D structured meshes.

        Args:
            num_heads: Number of attention heads
            hidden_dim: Dimension of hidden features
            dropout: Dropout rate
            act: Activation function used in MLP
            mlp_ratio: Expansion ratio for hidden dimension in MLP
            last_layer: Whether this is the final layer in the network
            out_dim: Output dimension (only used if last_layer=True)
            slice_num: Number of slices for attention computation
            H: Height of the 3D mesh
            W: Width of the 3D mesh
            D: Depth of the 3D mesh
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
        self.Attn = Physics_Attention_Structured_Mesh_3D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            H=H,
            W=W,
            D=D,
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

    def forward(self, fx):
        """Forward pass through the 3D Transolver block.

        Args:
            fx: Input feature tensor [batch_size, H*W*D, hidden_dim]

        Returns:
            Processed feature tensor with the same shape as input,
            or [batch_size, H*W*D, out_dim] if last_layer=True
        """
        # Apply attention with residual connection
        fx = self.Attn(self.ln_1(fx)) + fx

        # Apply MLP with residual connection
        fx = self.mlp(self.ln_2(fx)) + fx

        # Apply final projection if this is the last layer
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    """Transolver model for 3D structured mesh data.

    This model uses a transformer-based architecture with physics-informed attention
    specifically designed for 3D structured volumetric meshes. It can handle spatial
    coordinates and optional time inputs for time-dependent problems.

    Memory efficiency is enhanced through an optional gradient checkpointing mechanism
    for handling large 3D datasets.

    Attributes:
        __name__ (str): Model identifier
        use_checkpoint (bool): Whether to use gradient checkpointing for memory efficiency
        H (int): Height of the 3D mesh
        W (int): Width of the 3D mesh
        D (int): Depth of the 3D mesh
        ref (int): Reference grid resolution for position encoding
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
        H=32,
        W=32,
        D=32,
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
        """Initialize the Transolver model for 3D structured meshes.

        Args:
            space_dim: Dimension of spatial coordinates
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
            ref: Reference grid resolution for position encoding
            unified_pos: Whether to use unified position encoding
            H: Height of the 3D mesh
            W: Width of the 3D mesh
            D: Depth of the 3D mesh
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
        self.__name__ = "Transolver_3D"
        self.use_checkpoint = False  # Flag for gradient checkpointing
        self.H = H
        self.W = W
        self.D = D
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(
                fun_dim + self.ref * self.ref * self.ref,
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

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
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
                    H=H,
                    W=W,
                    D=D,
                    last_layer=(_ == n_layers - 1),
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
        """Initialize weights for specific layer types.

        Args:
            m: Module to initialize

        Note:
            - Linear layers: Weights initialized with truncated normal distribution
            - LayerNorm/BatchNorm1d: Bias initialized to 0, weight to 1
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        """Generate a 3D structured grid and compute relative distances.

        This method creates two grids:
        1. A main grid representing the 3D structured mesh (size H×W×D)
        2. A reference grid of size ref×ref×ref

        It then computes the distances between each point in the main grid
        and each point in the reference grid.

        Args:
            batchsize: Batch size for the generated grid

        Returns:
            Tensor of distances between main grid points and reference grid points
            Shape: [batchsize, H, W, D, ref*ref*ref]
        """
        size_x, size_y, size_z = self.H, self.W, self.D

        # Create main grid x coordinates [batchsize, H, 1, 1, 1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )

        # Create main grid y coordinates [batchsize, 1, W, 1, 1]
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )

        # Create main grid z coordinates [batchsize, 1, 1, D, 1]
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )

        # Combine into main grid [batchsize, H, W, D, 3]
        grid = torch.cat((gridx, gridy, gridz), dim=-1).cuda()

        # Create reference grid x coordinates [batchsize, ref, 1, 1, 1]
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(
            [batchsize, 1, self.ref, self.ref, 1]
        )

        # Create reference grid y coordinates [batchsize, 1, ref, 1, 1]
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(
            [batchsize, self.ref, 1, self.ref, 1]
        )

        # Create reference grid z coordinates [batchsize, 1, 1, ref, 1]
        gridz = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(
            [batchsize, self.ref, self.ref, 1, 1]
        )

        # Combine into reference grid [batchsize, ref, ref, ref, 3]
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda()

        # Compute Euclidean distances between main grid and reference grid points
        # Result shape: [batchsize, H, W, D, ref*ref*ref]
        pos = (
            torch.sqrt(
                torch.sum(
                    (
                        grid[:, :, :, :, None, None, None, :]
                        - grid_ref[:, None, None, None, :, :, :, :]
                    )
                    ** 2,
                    dim=-1,
                )
            )
            .reshape(batchsize, size_x, size_y, size_z, self.ref * self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, x, fx, T=None):
        """Forward pass of the 3D Transolver model.

        Args:
            x: Spatial coordinates tensor of shape [batch_size, H*W*D, space_dim]
               or [batch_size, H, W, D, space_dim] if unified_pos=True
            fx: Function values tensor of shape [batch_size, H*W*D, fun_dim]
               or None if no function values are provided
            T: Time values tensor of shape [batch_size, 1] or None for time-independent problems

        Returns:
            Output tensor of shape [batch_size, H*W*D, out_dim]

        Note:
            When use_checkpoint=True, gradient checkpointing is used to reduce memory
            consumption during training with large 3D volumes.
        """
        # Apply position encoding if unified_pos is enabled
        if self.unified_pos:
            # Reshape precomputed position encodings to match batch size
            x = self.pos.repeat(x.shape[0], 1, 1, 1, 1).reshape(
                x.shape[0], self.H * self.W * self.D, self.ref * self.ref * self.ref
            )

        # Preprocess input features
        if fx is not None:
            # Combine spatial coordinates with function values
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            # Use only spatial coordinates
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        # Add time embedding if time input is provided
        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        # Process through Transolver blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.use_checkpoint:
                # Use gradient checkpointing to save memory
                fx = checkpoint.checkpoint(block, fx)
            else:
                fx = block(fx)

        return fx
