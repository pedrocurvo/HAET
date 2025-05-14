"""
Components module for the Transolver models.

This module provides essential building blocks for the Transolver models:

- MLP: Customizable multi-layer perceptron with residual connections
- Embedding functions: Various positional and time encodings
  - timestep_embedding: Sinusoidal time embeddings for time-dependent problems
  - PositionalEncoding: Standard Transformer positional encoding
  - RotaryEmbedding: Rotary position embeddings for enhanced position awareness
- Transformer components: Specialized transformer implementations
  - ErwinTransformer: Base transformer implementation
  - ErwinFlashTransformer: Memory-efficient transformer with FlashAttention
"""

from .embedding import (
    PositionalEncoding,
    RotaryEmbedding,
    apply_2d_rotary_pos_emb,
    apply_rotary_pos_emb,
    rotate_half,
    timestep_embedding,
)
from .erwin import ErwinTransformer
from .erwinflash import ErwinTransformer as ErwinFlashTransformer
from .mlp import MLP

__all__ = [
    "MLP",
    "ErwinTransformer",
    "ErwinFlashTransformer",
    "timestep_embedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_2d_rotary_pos_emb",
    "PositionalEncoding",
    "RotaryEmbedding",
]
