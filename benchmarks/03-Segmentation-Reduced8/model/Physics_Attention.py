import torch.nn as nn
import torch
from einops import rearrange, repeat
from .erwin_flash import ErwinTransformer, Node
import sys
import os
from torch.utils.checkpoint import checkpoint

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from chunked_ops import chunked_matmul


def permute_block(x):
    """Helper function for checkpoint to free memory during backprop"""
    # This function won't be used directly anymore - using chunked_permute instead
    # to avoid OOM errors
    return x.permute(0, 1, 3, 2).contiguous()


def chunked_permute(x, chunk_size=1000000):
    """
    Performs permutation operation in chunks to reduce memory consumption
    
    Args:
        x (torch.Tensor): The tensor to permute, shape [B, H, N, G]
        chunk_size (int): Maximum number of elements to process in a single chunk
        
    Returns:
        torch.Tensor: Permuted tensor with shape [B, H, G, N]
    """
    B, H, N, G = x.shape
    
    # If tensor is small enough, just do regular permutation
    if N * G <= chunk_size:
        return x.permute(0, 1, 3, 2).contiguous()
    
    # For very large tensors, use an even more memory-efficient approach
    # by processing smaller chunks
    max_elements_per_chunk = min(chunk_size, 500000)  # Limit max elements per chunk
    n_chunks = (N + (max_elements_per_chunk // G) - 1) // (max_elements_per_chunk // G)  # Ceiling division
    
    # Process in chunks along the N dimension
    chunks = []
    for i in range(n_chunks):
        start_idx = i * (max_elements_per_chunk // G)
        end_idx = min((i + 1) * (max_elements_per_chunk // G), N)
        
        # Extract chunk and permute
        chunk = x[:, :, start_idx:end_idx, :].detach()  # Detach to save memory
        with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for this operation
            permuted_chunk = chunk.permute(0, 1, 3, 2).contiguous()
        chunks.append(permuted_chunk)
        
        # Explicitly free memory
        del chunk
        torch.cuda.empty_cache()
    
    # Concatenate along the new third dimension (which was the second dimension before permutation)
    result = torch.cat(chunks, dim=3)
    return result


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics-informed attention for irregular mesh data with Transolver++.

    This attention mechanism processes irregular mesh data through three main steps:
    1. Slicing: Projects input features into a reduced set of eidetic states using Rep-Slice
    2. Transformation: Processes eidetic states using the ErwinTransformer
    3. De-slicing: Projects transformed eidetic states back to the original points

    This approach allows efficient processing of irregular point clouds while
    maintaining awareness of spatial relationships.

    Attributes:
        dim_head (int): Dimension of each attention head
        heads (int): Number of attention heads
        dimensionality (int): Spatial dimensionality (3 for irregular meshes)
        epsilon (float): Small constant for Rep-Slice computation
        base_temp (float): Base temperature for adaptive temperature scaling
        in_project_x (nn.Linear): Linear projection for input features
        in_project_slice (nn.Linear): Linear projection for slice weights
        ada_temp_linear (nn.Linear): Linear projection for adaptive temperature
        erwin (ErwinTransformer): Transformer for processing eidetic states
        to_out (nn.Sequential): Output projection
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, base_temp=0.5, epsilon=1e-6, chunk_size=1000000):
        """Initialize the Physics_Attention_Irregular_Mesh module with Transolver++.

        Args:
            dim (int): Input feature dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            slice_num (int): Number of slice tokens to use
            base_temp (float): Base temperature for adaptive temperature scaling
            epsilon (float): Small constant for the log(-log(ε)) term in Rep-Slice
            chunk_size (int): Maximum number of points to process in a single chunk
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dimensionality = 3  # Spatial dimensionality for the irregular mesh
        self.epsilon = epsilon
        self.chunk_size = chunk_size
        self.use_checkpoint = True  # Use checkpoint to save memory
        
        # For Transolver++, we only need one projection to save memory
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        
        # Ada-Temp: Base temperature + adaptive component
        self.base_temp = base_temp
        self.ada_temp_linear = nn.Linear(dim_head, 1)  # Adaptive temperature adjustment
        
        torch.nn.init.orthogonal_(self.in_project_slice.weight)  # use a principled initialization

        # Hierarchical transformer for processing sliced tokens
        # This design uses a multi-level approach to capture features at different scales
        self.erwin = ErwinTransformer(
            c_in=dim_head,  # Input channel dimension matches head dimension
            c_hidden=[
                dim_head,  # First level maintains dimension
                dim_head * 2,  # Second level expands channels for higher expressivity
            ],
            ball_sizes=[
                min(32, slice_num),  # First level considers more neighbors
                min(16, slice_num // 2),  # Second level reduces neighborhood size
            ],
            enc_num_heads=[heads // 2, heads],  # More attention heads at deeper levels
            enc_depths=[2, 2],  # Same depth at each hierarchical level
            dec_num_heads=[
                heads // 2
            ],  # Decoder head count matches encoder at same level
            dec_depths=[2],  # Decoder depth matches encoder
            strides=[2],  # Coarsens the point cloud by factor of 2
            rotate=45,  # Enable rotation for better geometric awareness
            decode=True,  # Enable upsampling back to original resolution
            mlp_ratio=4,  # Standard expansion ratio in MLP blocks,
            mp_steps=0,  # No need for MPNN here
            dimensionality=self.dimensionality,  # Dimensionality of the space (3D for irregular mesh)
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        """Forward pass of the physics-informed attention module with Transolver++.
        
        Implements Transolver++ Algorithm 1: Parallel Physics-Attention with Eidetic States

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, num_points, channels]

        Returns:
            torch.Tensor: Output feature tensor of shape [batch_size, num_points, channels]
        """
        # Extract batch size, number of points, and channels
        B, N, C = x.shape

        ### (1) Slice operation: Project input features to a reduced set of tokens
        # Project input - drop fx to save 50% memory as per algorithm
        x_proj = (
            self.in_project_x(x)  # Linear projection [B, N, inner_dim]
            .reshape(
                B, N, self.heads, self.dim_head
            )  # Reshape for multi-head [B, N, H, C]
            .permute(0, 2, 1, 3)  # Reorder to [B, H, N, C]
            .contiguous()
        )
        
        # Free memory by deleting the input tensor if it's no longer needed
        del x
        torch.cuda.empty_cache()
        
        # Compute adaptive temperature (Ada-Temp): τ = τ0 + Linear(xi)
        # Implementation: τ(k) ←τ0 + Ada-Temp(x(k))
        adaptive_temp = self.base_temp + self.ada_temp_linear(x_proj).clamp(min=-0.4, max=0.4)
        
        # Compute Rep-Slice: Softmax(Linear(x) - log(-log(ε))) / τ
        # Implementation: w(k) ← Rep-Slice(x(k),τ(k))
        log_neg_log_epsilon = torch.log(-torch.log(torch.tensor(self.epsilon, device=x_proj.device)))
        slice_logits = self.in_project_slice(x_proj) - log_neg_log_epsilon
        
        # Free up memory
        del log_neg_log_epsilon
        torch.cuda.empty_cache()
        
        slice_weights = torch.softmax(slice_logits / adaptive_temp, dim=2)  # [B, H, N, G]
        
        # Free up memory
        del slice_logits, adaptive_temp
        torch.cuda.empty_cache()

        # Compute weights norm: w(k)_norm ← sum_i(w(k)_i)
        slice_norm = slice_weights.sum(2)  # [B, H, G]

        # Compute eidetic states: s(k) ← w(k)T x(k) / w_norm
        # We use x_proj both as x and f to save memory
        # Use chunked matmul for large point clouds to avoid CUBLAS errors
        if N > self.chunk_size:
            # First permute slice_weights from [B,H,N,G] to [B,H,G,N] for matmul
            slice_weights_t = chunked_permute(slice_weights, self.chunk_size // 4).contiguous()
            
            # Use chunked_matmul for the actual computation
            eidetic_states = chunked_matmul(slice_weights_t, x_proj, self.chunk_size // 4)  # [B, H, G, C]
        else:
            # For smaller matrices, use einsum which is more readable
            eidetic_states = torch.einsum(
                "bhnc,bhng->bhgc", x_proj, slice_weights
            )  # [B, H, G, C]

        # Free memory
        del x_proj
        torch.cuda.empty_cache()
        
        # Normalize eidetic states by the sum of weights to maintain scale
        eidetic_states = eidetic_states / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )
        
        # Free more memory
        del slice_norm
        torch.cuda.empty_cache()

        ### (2) Transform eidetic states with ErwinTransformer
        # Extract dimensions of eidetic states tensor
        B, H, G, C = eidetic_states.shape

        # Reshape eidetic states for ErwinTransformer input format
        # ErwinTransformer expects a flattened representation [num_points, channels]
        eidetic_states_flat = eidetic_states.reshape(B * H * G, C)

        # Free memory
        del eidetic_states
        torch.cuda.empty_cache()

        # Use center of mass positions for eidetic states instead of random positions
        # Since eidetic states are already weighted averages (center of mass) of the input features,
        # we compute their spatial representation in the unit cube based on their relative positions
        # in the feature space, normalized across each batch and head
        
        # Compute center of mass positions by normalizing features to unit cube
        feat_min = eidetic_states_flat.min(dim=0, keepdim=True)[0]
        feat_max = eidetic_states_flat.max(dim=0, keepdim=True)[0]
        feat_range = feat_max - feat_min + 1e-8  # Add epsilon to avoid division by zero
        
        # Use first dimensionality components as spatial positions, normalized to [0,1]
        pos = (eidetic_states_flat[:, :self.dimensionality] - feat_min[:, :self.dimensionality]) / feat_range[:, :self.dimensionality]

        # Create batch indices to separate different batch and head combinations
        # Each batch-head pair is treated as a separate point cloud by the transformer
        batch_idx = torch.arange(B * H, device=eidetic_states_flat.device).repeat_interleave(G)

        # Process through ErwinTransformer to allow interactions between eidetic states
        # This enables global information exchange and feature refinement
        # This corresponds to: Update eidetic states s′← Attention(s)
        processed_states = self.erwin(eidetic_states_flat, pos, batch_idx)
        
        # Free memory
        del eidetic_states_flat, pos, batch_idx
        torch.cuda.empty_cache()

        # Reshape the processed states back to the multi-head format
        out_eidetic_states = processed_states.reshape(B, H, G, C)
        
        # Free memory
        del processed_states
        torch.cuda.empty_cache()

        ### (3) Deslice operation: Project processed states back to original points
        # Distribute processed eidetic states information back to original points
        # Using the same slice weights ensures information flows properly between
        # the slicing and de-slicing operations
        # Implementation: x′(k) ← Deslice(s′, w(k))
        # Use chunked matmul for large point clouds to avoid CUBLAS errors
        if N > self.chunk_size:
            # First permute slice_weights for deslice operation
            slice_weights_t = chunked_permute(slice_weights, self.chunk_size // 4).contiguous()
                    
            # Use chunked_matmul for the deslice operation
            out_x = chunked_matmul(out_eidetic_states, slice_weights_t, self.chunk_size // 4)
        else:
            # For smaller point clouds, use einsum
            out_x = torch.einsum("bhgc,bhng->bhnc", out_eidetic_states, slice_weights)
        
        # Free memory
        del out_eidetic_states, slice_weights
        torch.cuda.empty_cache()

        # Rearrange from multi-head format to batch-first format
        # Concatenating all heads into a single feature dimension
        out_x = rearrange(out_x, "b h n d -> b n (h d)")

        # Project concatenated features back to the original dimension
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=101, W=31, kernel=3):  # kernel=3):
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
            c_hidden=[dim_head, dim_head*2],  # Two levels of hierarchy with increasing channels
            ball_sizes=[min(32, slice_num), min(16, slice_num//2)],  # Progressive reduction
            enc_num_heads=[heads//2, heads],  # Increasing heads in deeper layers
            enc_depths=[2, 2],  # Equal depth at each level
            dec_num_heads=[heads//2],  # Matching encoder at the same level
            dec_depths=[2],  # Matching encoder depth
            strides=[2],  # Downsample by factor of 2
            rotate=1,  # Enable rotation for better cross-token mixing
            decode=True,  # We need the full resolution back
            mlp_ratio=4,  # Standard MLP expansion ratio
            dimensionality=self.dimensionality,
            mp_steps=0  # No need for MPNN here
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Transform with ErwinTransformer instead of attention
        # Process each batch and head independently
        B, H, G, C = slice_token.shape
        
        # Reshape for Erwin: [B, H, G, C] -> [B*H*G, C]
        slice_token_flat = slice_token.reshape(B*H*G, C)  # Flatten to [total_points, channels]
        
        # Create artificial positions for slice tokens in 2D space (for Structured_Mesh_2D)
        pos = torch.rand(B*H*G, self.dimensionality, device=slice_token.device)  # [total_points, 2]
        
        # Create batch indices - each batch and head combination gets its own batch index
        batch_idx = torch.arange(B*H, device=slice_token.device).repeat_interleave(G)
        
        # Process through ErwinTransformer - it expects [num_points, channels] for features
        processed_tokens = self.erwin(slice_token_flat, pos, batch_idx)
        
        # Reshape back to original format [B, H, G, C]
        out_slice_token = processed_tokens.reshape(B, H, G, C)

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D
        self.dimensionality = 3  # 3D space

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        
        # Replace with more sophisticated hierarchical ErwinTransformer
        self.erwin = ErwinTransformer(
            c_in=dim_head,
            c_hidden=[dim_head, dim_head*2],  # Two levels of hierarchy with increasing channels
            ball_sizes=[min(32, slice_num), min(16, slice_num//2)],  # Progressive reduction
            enc_num_heads=[heads//2, heads],  # Increasing heads in deeper layers
            enc_depths=[2, 2],  # Equal depth at each level
            dec_num_heads=[heads//2],  # Matching encoder at the same level
            dec_depths=[2],  # Matching encoder depth
            strides=[2],  # Downsample by factor of 2
            rotate=1,  # Enable rotation for better cross-token mixing
            decode=True,  # We need the full resolution back
            mlp_ratio=4,  # Standard MLP expansion ratio
            dimensionality=self.dimensionality,
            mp_steps=0  # No need for MPNN here
        )
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Transform with ErwinTransformer instead of attention
        # Process each batch and head independently
        B, H, G, C = slice_token.shape
        
        # Reshape for Erwin: [B, H, G, C] -> [B*H*G, C]
        slice_token_flat = slice_token.reshape(B*H*G, C)  # Flatten to [total_points, channels]
        
        # Create artificial positions for slice tokens in 3D space
        pos = torch.rand(B*H*G, self.dimensionality, device=slice_token.device)  # [total_points, 3]
        
        # Create batch indices - each batch and head combination gets its own batch index
        batch_idx = torch.arange(B*H, device=slice_token.device).repeat_interleave(G)
        
        # Process through ErwinTransformer - it expects [num_points, channels] for features
        processed_tokens = self.erwin(slice_token_flat, pos, batch_idx)
        
        # Reshape back to original format [B, H, G, C]
        out_slice_token = processed_tokens.reshape(B, H, G, C)

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
