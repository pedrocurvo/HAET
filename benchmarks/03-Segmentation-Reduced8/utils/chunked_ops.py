import torch

def chunked_matmul(mat1, mat2, chunk_size=1000000):
    """
    Performs matrix multiplication in chunks to avoid CUBLAS_STATUS_NOT_SUPPORTED errors
    when dealing with very large matrices.
    
    Args:
        mat1 (torch.Tensor): First matrix for multiplication
        mat2 (torch.Tensor): Second matrix for multiplication
        chunk_size (int): Maximum number of elements to process in a single chunk
        
    Returns:
        torch.Tensor: Result equivalent to torch.matmul(mat1, mat2)
    """
    # Special case for attention calculation
    if mat1.dim() == 4 and mat2.dim() == 4:
        # Get dimensions
        B1, H1, *rest1 = mat1.shape
        B2, H2, *rest2 = mat2.shape
        
        # Make sure batch and head dimensions match
        assert B1 == B2 and H1 == H2, f"Batch {B1}!={B2} or head {H1}!={H2} dimensions don't match"
        
        # Check if this is the slice operation (weights @ features -> eidetic states)
        if mat1.shape[2] != mat2.shape[2] and mat1.shape[3] == mat2.shape[2]:
            # For the first matmul: slice_weights_t (B,H,G,N) @ x_proj (B,H,N,C) -> (B,H,G,C)
            # mat1 shape: [B, H, G, N]
            # mat2 shape: [B, H, N, C]
            G, N = mat1.shape[2], mat1.shape[3]
            N2, C = mat2.shape[2], mat2.shape[3]
            
            # The sequence dimension should match
            assert N == N2, f"Sequence length dimensions don't match {N}!={N2}"
            
            # For large N, we need to chunk along that dimension
            if N > chunk_size:
                result = None
                num_chunks = (N + chunk_size - 1) // chunk_size  # Ceiling division
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, N)
                    
                    # Extract chunk of each matrix for this calculation
                    mat1_chunk = mat1[..., start_idx:end_idx]  # [B, H, G, chunk]
                    mat2_chunk = mat2[..., start_idx:end_idx, :]  # [B, H, chunk, C]
                    
                    # Compute the chunk result
                    chunk_result = torch.matmul(mat1_chunk, mat2_chunk)  # [B, H, G, C]
                    
                    # Accumulate results
                    if result is None:
                        result = chunk_result
                    else:
                        result += chunk_result
                
                return result
            else:
                # For smaller matrices, compute directly
                return torch.matmul(mat1, mat2)
        
        # Check if this is the deslice operation (eidetic_states @ slice_weights -> output)
        elif mat1.shape[2] == mat2.shape[2]:
            # For the second matmul: out_eidetic_states (B,H,G,C) @ slice_weights_t (B,H,G,N) -> (B,H,N,C)
            # mat1 shape: [B, H, G, C]
            # mat2 shape: [B, H, G, N]
            G1, C = mat1.shape[2], mat1.shape[3]
            G2, N = mat2.shape[2], mat2.shape[3]
            
            # The G dimension should match
            assert G1 == G2, f"Inner dimension G doesn't match {G1}!={G2}"
            
            # For large N, we chunk along the N dimension
            if N > chunk_size:
                chunks = []
                num_chunks = (N + chunk_size - 1) // chunk_size  # Ceiling division
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, N)
                    
                    # Extract chunk of N dimension for this calculation
                    mat2_chunk = mat2[..., start_idx:end_idx]  # [B, H, G, chunk]
                    
                    # Transpose and matmul
                    chunk_result = torch.matmul(mat1.transpose(2, 3), mat2_chunk)  # [B, H, C, chunk]
                    chunk_result = chunk_result.permute(0, 1, 3, 2)  # [B, H, chunk, C]
                    chunks.append(chunk_result)
                
                # Concatenate along the N dimension (dim=2 after permutation)
                return torch.cat(chunks, dim=2)
            else:
                # For smaller matrices, compute directly
                result = torch.matmul(mat1.transpose(2, 3), mat2)  # [B, H, C, N]
                return result.permute(0, 1, 3, 2)  # [B, H, N, C]
    
    # Standard 3D tensor matmul
    # Handle normal case when mat2 is not too large
    if mat2.size(1) <= chunk_size:
        return torch.matmul(mat1, mat2)
    
    # For large matrices, break into chunks along the second dimension
    chunks = []
    num_chunks = (mat2.size(1) + chunk_size - 1) // chunk_size  # Ceiling division
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, mat2.size(1))
        
        # Extract chunk and compute matmul for this chunk
        chunk_result = torch.matmul(mat1, mat2[:, start_idx:end_idx, :])
        chunks.append(chunk_result)
    
    # Concatenate results along the second dimension
    return torch.cat(chunks, dim=1)

def chunked_attention(query, key, value, attention_mask=None, chunk_size=1000000):
    """
    Performs attention computation in chunks to avoid CUBLAS_STATUS_NOT_SUPPORTED errors.
    
    This implements a chunked version of the standard attention formula:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        query (torch.Tensor): Query tensor [B, N_q, d_k]
        key (torch.Tensor): Key tensor [B, N_k, d_k]
        value (torch.Tensor): Value tensor [B, N_k, d_v]
        attention_mask (torch.Tensor, optional): Mask to apply before softmax [B, N_q, N_k]
        chunk_size (int): Maximum number of tokens to process in a single chunk
        
    Returns:
        torch.Tensor: Result of the attention operation [B, N_q, d_v]
    """
    d_k = query.size(-1)
    scaling_factor = d_k ** 0.5
    
    # Handle normal case when key is not too large
    if key.size(1) <= chunk_size and query.size(1) <= chunk_size:
        # Standard attention calculation
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
        
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
            
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value)
    
    # For large attention matrices, process in chunks
    batch_size, n_q, d_k = query.size()
    n_k = key.size(1)
    d_v = value.size(2)
    
    # Determine chunk sizes for queries and keys
    q_chunk_size = min(n_q, chunk_size)
    k_chunk_size = min(n_k, chunk_size)
    
    result = torch.zeros((batch_size, n_q, d_v), device=query.device, dtype=query.dtype)
    
    # Process in chunks
    for q_start in range(0, n_q, q_chunk_size):
        q_end = min(q_start + q_chunk_size, n_q)
        q_chunk = query[:, q_start:q_end, :]
        
        # Initialize attention weights for this query chunk
        attn_weights = torch.zeros((batch_size, q_end - q_start, n_k), 
                                 device=query.device, dtype=query.dtype)
        
        # Process key chunks
        for k_start in range(0, n_k, k_chunk_size):
            k_end = min(k_start + k_chunk_size, n_k)
            k_chunk = key[:, k_start:k_end, :]
            v_chunk = value[:, k_start:k_end, :]
            
            # Compute partial attention weights
            chunk_weights = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / scaling_factor
            
            # Apply mask if provided
            if attention_mask is not None:
                chunk_mask = attention_mask[:, q_start:q_end, k_start:k_end]
                chunk_weights = chunk_weights.masked_fill(chunk_mask == 0, -1e9)
                
            # Add to full attention weights matrix
            attn_weights[:, :, k_start:k_end] = chunk_weights
        
        # Apply softmax to complete attention weights for this query chunk
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Compute weighted sum in chunks
        chunk_result = torch.zeros((batch_size, q_end - q_start, d_v), 
                                 device=query.device, dtype=query.dtype)
        
        for k_start in range(0, n_k, k_chunk_size):
            k_end = min(k_start + k_chunk_size, n_k)
            v_chunk = value[:, k_start:k_end, :]
            
            # Multiply chunk probabilities with values
            chunk_attn = attn_probs[:, :, k_start:k_end]
            chunk_result += torch.matmul(chunk_attn, v_chunk)
        
        # Add to result
        result[:, q_start:q_end, :] = chunk_result
    
    return result 