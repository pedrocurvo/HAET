"""
Physics-informed attention mechanism for 3D structured meshes.

This module implements a specialized attention mechanism optimized for
volumetric data arranged in regular 3D grids. It leverages 3D convolutions
for local feature extraction and a slicing-and-deslicing approach combined
with an ErwinTransformer for global feature interactions.

Key features of the architecture:
1. Memory-efficient representation through Rep-Slice tokenization
2. Adaptive temperature mechanism for improved training stability
3. Memory tokens for enhanced global information exchange
4. Spatial-aware attention through position encoding
5. High performance for volumetric scientific data processing

This implementation is specifically designed for physics simulations and
scientific computing applications that operate on 3D volumetric data.
"""

import torch
import torch.nn as nn
from einops import rearrange

from ..components import ErwinFlashTransformer as ErwinTransformer


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    """Physics-informed attention for 3D structured mesh data with Transolver++.

    This attention mechanism processes 3D structured mesh data through three main steps:
    1. Slicing: Projects input features into a reduced set of eidetic states using 3D convolutions and Rep-Slice
    2. Transformation: Processes eidetic states using the ErwinTransformer
    3. De-slicing: Projects transformed eidetic states back to the original volumetric points

    The use of 3D convolutions allows efficient local feature extraction that
    respects the spatial structure of volumetric data, while Transolver++ with adaptive
    temperature and eidetic states enhances memory efficiency and performance.

    Attributes:
        dim_head (int): Dimension of each attention head
        heads (int): Number of attention heads
        H (int): Height of the 3D mesh
        W (int): Width of the 3D mesh
        D (int): Depth of the 3D mesh
        dimensionality (int): Spatial dimensionality (3 for 3D meshes)
        epsilon (float): Small constant for Rep-Slice computation
        base_temp (float): Base temperature for adaptive temperature scaling
        in_project_x (nn.Conv3d): 3D convolution for input features
        in_project_slice (nn.Linear): Linear projection for slice weights
        ada_temp_linear (nn.Linear): Linear projection for adaptive temperature adjustment
        erwin (ErwinTransformer): Transformer for processing eidetic states
        to_out (nn.Sequential): Output projection
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=32,
        H=32,
        W=32,
        D=32,
        kernel=3,
        base_temp=0.5,
        epsilon=1e-6,
        radius: float = 1.0,     # Add radius parameter with default value
        dimensionality: int = 3,
        # ErwinTransformer parameters
        c_hidden=None,
        ball_sizes=None,
        enc_num_heads=None,
        enc_depths=None,
        dec_num_heads=None,
        dec_depths=None,
        strides=None,
        rotate=1,
        decode=True,
        mlp_ratio=4,
        mp_steps=0,
        embed=False,
        memory_tokens=1,  # Number of memory tokens for eidetic states
        attention_heads=1,  # Number of attention heads for eidetic states
    ):
        """Initialize the Physics_Attention_Structured_Mesh_3D module with Transolver++.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            slice_num (int): Number of slice tokens to use
            H (int): Height of the 3D mesh
            W (int): Width of the 3D mesh
            D (int): Depth of the 3D mesh
            kernel (int): Size of convolution kernel for local feature extraction
            base_temp (float): Base temperature for adaptive temperature scaling
            epsilon (float): Small constant for the log(-log(ε)) term in Rep-Slice
            radius (float): Radius for the ErwinTransformer ball query
            dimensionality (int): Dimensionality of the input space (3 for 3D)
            c_hidden (list): Hidden channel dimensions for each hierarchical level
            ball_sizes (list): Ball sizes for each hierarchical level
            enc_num_heads (list): Number of attention heads for each encoder level
            enc_depths (list): Depth of each encoder level
            dec_num_heads (list): Number of attention heads for each decoder level
            dec_depths (list): Depth of each decoder level
            strides (list): Stride values for each level
            rotate (int): Rotate flag for geometric awareness
            decode (bool): Whether to decode/upsample back to original resolution
            mlp_ratio (int): Expansion ratio in MLP blocks
            mp_steps (int): Number of message passing steps
            embed (bool): Whether to use ErwinEmbedding (True) or direct projection (False)
            memory_tokens (int): Number of memory tokens for eidetic states
            attention_heads (int): Number of attention heads for eidetic states
        """
        super().__init__()
        self.attn_heads = attention_heads
        inner_dim = dim_head * self.attn_heads
        self.dim_head = dim_head
        self.heads = heads
        self.H = H
        self.W = W
        self.D = D
        self.dimensionality = dimensionality  # Should be 3 for 3D meshes
        self.radius = radius
        self.epsilon = epsilon
        self.base_temp = base_temp
        self.slice_num = slice_num
        self.memory_tokens = self.slice_num // 32

        # Memory tokens - learnable parameters that act as global information aggregators
        # Shape: [1, attention_heads, memory_tokens, dim_head]
        # These memory states serve as persistent, long-range information carriers
        # that can attend to all positions in the 3D volume
        self.memory_states = nn.Parameter(torch.randn(1, self.attn_heads, self.memory_tokens, dim_head))
        
        # Initialize memory token positions in a uniform grid pattern
        # These positions are learnable and help memory tokens develop spatial awareness
        init_pos = self.uniform_memory_positions(self.attn_heads, self.memory_tokens, dimensionality).clone()
        self.memory_positions = nn.Parameter(init_pos)
        
        # Initialize memory tokens with Xavier uniform initialization for better gradient flow
        # This helps with faster convergence at the beginning of training
        nn.init.xavier_uniform_(self.memory_states)
        # Memory positions are already initialized uniformly, so no need for Xavier init
        # nn.init.xavier_uniform_(self.memory_positions)

        # For Transolver++, we employ a memory-efficient projection strategy
        # 3D Convolutional layer for volumetric feature extraction
        # This captures local spatial patterns in the 3D volume before tokenization
        # kernel_size=kernel, stride=1, padding=kernel//2 to maintain spatial dimensions
        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        
        # Rep-Slice projection - Representational Slicing mechanism
        # This projects features to slice tokens in a differentiable way
        # The slice operation converts the dense 3D grid to a sparse set of tokens
        # which dramatically reduces the computational complexity for attention
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.in_project_slice._is_rep_slice = True  # Tag for custom initialization in optimizers
        # Orthogonal initialization improves the quality of the slice representation
        # by making the projection directions more distinct
        nn.init.orthogonal_(self.in_project_slice.weight)
        
        # Ada-Temp: Adaptive Temperature mechanism
        # This dynamic temperature scaling improves Gumbel-Softmax training stability
        # by adjusting the temperature based on the input content
        self.base_temp = base_temp  # Base temperature value (default starting point)
        self.ada_temp_norm = nn.LayerNorm(dim_head)  # Normalize before temperature prediction
        self.ada_temp_linear = nn.Linear(dim_head, 1)  # Predicts temperature adjustment
        self.ada_temp_linear._is_ada_temp = True  # Tag for special handling in optimizers

        # Set default ErwinTransformer parameters if not provided
        # These defaults are carefully chosen for 3D volumetric data processing
        if c_hidden is None:
            # Progressive channel expansion for hierarchical feature extraction
            c_hidden = [dim_head, dim_head * 2]  # [features at level 1, features at level 2]
        if ball_sizes is None:
            # Ball sizes control the receptive field at each level of the hierarchy
            # For 3D volumes, we use larger ball sizes to capture more context
            ball_sizes = [max(64, int(0.25 * slice_num)), max(64, int(0.25 * slice_num))]
        if enc_num_heads is None:
            # Progressive increase in attention heads for the encoder
            enc_num_heads = [heads // 2, heads]  # [heads at level 1, heads at level 2]
        if enc_depths is None:
            # Number of transformer blocks at each encoder level
            enc_depths = [2, 2]  # [depth at level 1, depth at level 2]
        if dec_num_heads is None:
            # Number of attention heads for the decoder
            dec_num_heads = [heads // 2]  # [heads at level 1]
        if dec_depths is None:
            # Number of transformer blocks for the decoder
            dec_depths = [2]  # [depth at level 1]
        if strides is None:
            # Downsampling factor between hierarchical levels
            strides = [2]  # [stride from level 1 to level 2]

        # ErwinTransformer: Hierarchical transformer for processing sliced tokens
        # This is a specialized transformer architecture that builds a multi-scale
        # representation of the input data, enabling efficient long-range interactions
        # while maintaining computational efficiency.
        self.erwin = ErwinTransformer(
            c_in=dim_head,          # Input channel dimension matches head dimension
            c_hidden=c_hidden,      # Hidden channel dimensions for each hierarchical level
            ball_sizes=ball_sizes,  # Controls the local neighborhood size at each level
            enc_num_heads=enc_num_heads,  # Attention heads for each encoder level
            enc_depths=enc_depths,  # Number of transformer blocks per encoder level
            dec_num_heads=dec_num_heads,  # Attention heads for each decoder level
            dec_depths=dec_depths,  # Number of transformer blocks per decoder level
            strides=strides,        # Downsampling factor between levels
            rotate=rotate,          # Enable rotation for better geometric awareness
            decode=decode,          # Enable upsampling back to original resolution
            mlp_ratio=mlp_ratio,    # Expansion ratio in MLP blocks
            dimensionality=self.dimensionality,  # 3D space
            mp_steps=mp_steps,      # Additional message passing steps for refinement
            embed=embed,            # Whether to use ErwinEmbedding or direct projection
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Initialize slice weights attribute
        self.slice_weights = None

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the physics-informed attention module for 3D structured meshes with Transolver++.
        
        Implements Transolver++ Algorithm 1: Parallel Physics-Attention with Eidetic States

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, H*W*D, channels]
            pos (torch.Tensor, optional): Position tensor of shape [batch_size, H*W*D, dimensionality]

        Returns:
            torch.Tensor: Output feature tensor of shape [batch_size, H*W*D, channels]
        """
        # Extract batch size, number of points, and channels from input tensor
        B, N, C = x.shape  # B: batch size, N: number of mesh points (H*W*D), C: channels
        
        # STEP 0: Reshape from flattened representation to 3D structured grid
        # Input comes as [B, N, C] where N = H*W*D (flattened 3D volume)
        # We reshape it to [B, H, W, D, C] then permute to [B, C, H, W, D] for Conv3D
        # This preserves the spatial structure of the 3D data for convolutional processing
        x = (
            x.reshape(B, self.H, self.W, self.D, C)  # Restore the 3D spatial structure
            .contiguous()  # Ensure memory layout is contiguous for efficiency
            .permute(0, 4, 1, 2, 3)  # Rearrange to [B, C, H, W, D] for Conv3D
            .contiguous()  # Ensure memory layout is contiguous after permute
        )

        ### STEP 1: Slice operation - Convert dense 3D volume to sparse tokens
        # This is the first key step of Transolver++: Rep-Slice tokenization
        # Project features using 3D convolution - extract spatial features while
        # also splitting into attention heads for multi-head processing
        x_proj = (
            self.in_project_x(x)  # [B, inner_dim, H, W, D] - Apply 3D convolution
            .permute(0, 2, 3, 4, 1)  # [B, H, W, D, inner_dim] - Move channels to end
            .contiguous()  # Ensure memory layout is contiguous
            .reshape(B, N, self.attn_heads, self.dim_head)  # [B, H*W*D, heads, dim_head] - Split into heads
            .permute(0, 2, 1, 3)  # [B, heads, H*W*D, dim_head] - Multi-head format
            .contiguous()  # Ensure memory layout is contiguous
        )

        # Prepare position information for attention mechanism
        # Reshape positions to match attention head structure
        # This ensures positions are processed per-head, preserving spatial awareness
        pos_proj = pos.view(B, N, 1, self.dimensionality)  # [B, N, 1, 3]
        pos_proj = pos_proj.expand(B, N, self.attn_heads, self.dimensionality)  # [B, N, H, 3]
        pos_proj = pos_proj.transpose(1, 2)  # [B, H, N, 3]

        # STEP 2: Compute adaptive temperature (Ada-Temp)
        # The temperature parameter controls the softness of the Gumbel-Softmax distribution
        # Ada-Temp dynamically adjusts temperature based on input content:
        # τ = τ0 + f(x), clamped to reasonable range
        # Normalize features before predicting temperature for stability
        tau = torch.clamp(
            self.base_temp + self.ada_temp_linear(self.ada_temp_norm(x_proj)),  # Base + adaptive component
            min=0.1,  # Prevent temperature from getting too low (too discrete)
            max=2.0   # Prevent temperature from getting too high (too uniform)
        )

        # STEP 3: Rep-Slice - Compute slice weights using Gumbel-Softmax
        # This is the core of the Representational Slicing mechanism
        # It creates a differentiable, sparse selection of slice tokens
        raw_logits = self.in_project_slice(x_proj)  # [B, H, N, G] - Project each point to slice logits
        # Apply Gumbel-Softmax to get differentiable one-hot approximations
        slice_weights = self.gumbel_softmax_sample(raw_logits, tau, self.training)  # [B, H, N, G]

        # Save slice weights for visualization during inference
        if not self.training:
            self.slice_weights = slice_weights  # Save for inspection and analysis

        # STEP 4: Normalize weights for stable eidetic state computation
        # Sum weights along the mesh points dimension to get normalization factor
        norm = slice_weights.sum(2, keepdim=True)  # [B, H, 1, G]

        # STEP 5: Compute eidetic states - weighted combination of features
        # Eidetic states are weighted centers of mass for each slice token
        # s_k = Σ(w_ik * x_i) / Σ(w_ik) for each slice token k
        # For features:
        eidetic_states = torch.matmul(
            slice_weights.transpose(-2, -1),  # [B, H, G, N]
            x_proj                           # [B, H, N, D]
        ) / (norm.transpose(-1, -2) + 1e-3)  # [B, H, G, D] / [B, H, G, 1]
        
        # For positions - maintain spatial awareness in token space:
        eidetic_pos = torch.matmul(
            slice_weights.transpose(-2, -1),  # [B, H, G, N]
            pos_proj                         # [B, H, N, 3]
        ) / (norm.transpose(-1, -2) + 1e-3)  # [B, H, G, 3] / [B, H, G, 1]

        ### STEP 6: Add memory tokens to eidetic states
        # Memory tokens are global information aggregators that can attend
        # to all positions in the volume, enabling long-range dependencies
        
        # Expand learnable memory states to match the current batch size
        # Memory tokens maintain their learned representations but are expanded
        # to process each example in the batch
        memory_states_expanded = self.memory_states.expand(B, -1, -1, -1)  # [B, H, M, D]
        memory_pos_expanded = self.memory_positions.expand(B, -1, -1, -1)  # [B, H, M, D_pos]
        
        # Concatenate memory tokens with eidetic states to create the complete token set
        # This combines domain-specific tokens (from Rep-Slice) with global memory tokens
        eidetic_states = torch.cat([eidetic_states, memory_states_expanded], dim=2)  # [B, H, G+M, D]
        eidetic_pos = torch.cat([eidetic_pos, memory_pos_expanded], dim=2)  # [B, H, G+M, D_pos]

        ### STEP 7: Process combined tokens with ErwinTransformer
        # ErwinTransformer enables hierarchical information exchange between tokens
        # while maintaining awareness of their spatial relationships
        
        # Prepare inputs for ErwinTransformer by reshaping tensors
        # ErwinTransformer expects flattened inputs with batch indices
        B, H, G_plus_M, D = eidetic_states.shape
        
        # Flatten tokens from all batches and heads into a single dimension
        states_flat = eidetic_states.reshape(B * H * G_plus_M, D)  # [B*H*(G+M), D]
        pos_flat = eidetic_pos.reshape(B * H * G_plus_M, self.dimensionality)  # [B*H*(G+M), 3]
        
        # Create batch indices to separate different batch-head combinations
        # Each batch-head pair forms a separate point cloud for the transformer
        batch_idx = torch.arange(B * H, device=x.device).repeat_interleave(G_plus_M)  # [B*H*(G+M)]

        # Apply ErwinTransformer to process tokens
        # The transformer enables information exchange between tokens based on
        # their feature similarity and spatial proximity (controlled by radius)
        updated = self.erwin(states_flat, pos_flat, batch_idx, radius=self.radius)
        
        # Reshape back to multi-head format with separate batch and head dimensions
        updated = updated.view(B, H, G_plus_M, D)  # [B, H, G+M, D]
        
        ### STEP 8: Separate updated tokens into slice tokens and memory tokens
        # After the ErwinTransformer, we need to separate the combined token set
        # back into regular slice tokens and memory tokens
        updated_slices = updated[:, :, :self.slice_num, :]  # [B, H, G, D]
        updated_memory = updated[:, :, self.slice_num:, :]  # [B, H, M, D]
        # Note: updated_memory captures global information but is not directly used in output
        # It influences the output indirectly through the transformer's self-attention

        ### STEP 9: De-slice operation - Project tokens back to the original mesh points
        # This is the reverse of the slice operation, distributing updated token
        # information back to the full 3D volume
        # We use the same slice_weights for consistency between slice and de-slice
        out = torch.matmul(slice_weights, updated_slices)  # [B, H, N, D]
        
        ### STEP 10: Final projection and reshape to match expected output format
        # Rearrange from multi-head format [B, H, N, D] to concatenated format [B, N, H*D]
        out = out.transpose(1, 2)  # [B, N, H, D]
        out = out.reshape(B, N, self.attn_heads * self.dim_head)  # [B, N, H*D]
        
        # Final projection to match the desired output dimension and apply dropout
        return self.to_out(out)  # [B, N, C]

    def uniform_memory_positions(self, heads, memory_tokens, dimensionality):
        """Initialize memory token positions in a uniform grid across the unit cube.
        
        This creates a regular grid of positions that spans the entire 3D space,
        allowing memory tokens to have broad spatial coverage. These positions
        serve as the initial spatial anchors for memory tokens before training.
        
        Args:
            heads (int): Number of attention heads
            memory_tokens (int): Number of memory tokens per head
            dimensionality (int): Spatial dimensionality (3 for 3D space)
            
        Returns:
            torch.Tensor: Tensor of shape [1, heads, memory_tokens, dimensionality]
                containing uniform grid positions
        """
        # Create a uniform grid from 0 to 1 with memory_tokens points
        grid = torch.linspace(0, 1, steps=memory_tokens)
        
        # Create positional grid based on dimensionality
        if dimensionality == 1:
            # For 1D, use the grid directly
            base = grid
        elif dimensionality == 2:
            # For 2D, create a 2D grid of positions
            base = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).view(-1, 2)
        elif dimensionality == 3:
            # For 3D, create a 3D grid of positions (cube)
            base = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).view(-1, 3)
        else:
            raise ValueError("Unsupported dimensionality")

        # Ensure we only have the requested number of memory tokens
        # (meshgrid might create more positions than needed)
        base = base[:memory_tokens]  # Ensure size matches memory_tokens
        
        # Expand to format [1, heads, memory_tokens, dimensionality]
        # This creates identical positions for each head in the batch
        base = base.unsqueeze(0).unsqueeze(0).expand(1, heads, -1, -1)  # [1, H, M, D]
        return base
    
    def gumbel_softmax_sample(self, logits, tau, training=True):
        """Differentiable Gumbel-Softmax sampling for discrete selection with backprop support.
        
        This implements the reparameterization trick from the Gumbel-Softmax paper
        (Jang et al., 2017) to allow backpropagation through the discrete selection
        of slice tokens. During training, it adds Gumbel noise to create a differentiable
        approximation of argmax. During inference, it becomes a standard softmax.
        
        Args:
            logits (torch.Tensor): Raw logits for each category
            tau (torch.Tensor): Temperature parameter that controls discreteness
                                Lower values make distribution more discrete
            training (bool): Whether in training mode (True) or inference mode (False)
            
        Returns:
            torch.Tensor: Differentiable one-hot approximation with the same shape as logits
        """
        if training:
            # Sample Gumbel noise from Gumbel(0,1) distribution
            # g = -log(-log(u)) where u ~ Uniform(0,1)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            
            # Apply the reparameterization trick:
            # y = softmax((logits + g) / τ)
            return torch.softmax((logits + gumbel_noise) / tau, dim=-1)
        else:
            # During inference, just use standard softmax with temperature
            return torch.softmax(logits / tau, dim=-1)
