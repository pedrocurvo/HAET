Architecture
============

HAET: Hierarchical Attention Erwin Transolver
---------------------------------------------

HAET combines the strengths of two powerful neural architectures to efficiently process mesh data at multiple scales:

1. **Transolver**: A transformer-based architecture for solving PDEs on general geometries
2. **Erwin**: A tree-based hierarchical transformer for large-scale physical systems

Problem Statement
------------------

Traditional approaches face key challenges when processing complex mesh structures:

- **Transolver** efficiently processes mesh data through token slicing but suffers from quadratic complexity in attention between slice tokens, limiting the number of slices.
- **Erwin Transformer** excels at hierarchical processing with ball-based attention but struggles with very large point clouds.

Core Architecture
-----------------

.. image:: ../images/haet_architecture.png
   :width: 800
   :alt: HAET Architecture Diagram

*Note: Add an architecture diagram to the images folder*

The HAET architecture works as follows:

1. **Mesh Tokenization**:
   
   - The input mesh is processed using Transolver's slicing mechanism
   - Tokens are created that represent different regions of the mesh
   - This initial tokenization allows processing of complex geometries

2. **Virtual Cube Embedding**:
   
   - Tokens are placed in a virtual 3D space
   - This spatial organization enables hierarchical processing
   - Position embeddings capture spatial relationships

3. **Hierarchical Ball Attention**:
   
   - Instead of standard self-attention between slice tokens (which is O(n²)), HAET applies Erwin's hierarchical ball attention
   - Points are organized into a tree structure
   - Attention is computed within local "balls" of points
   - Multi-scale features are captured through encoder-decoder architecture

Key Components
--------------

The architecture consists of several key components:

- **Transolver Models**: Handle different mesh types (2D/3D structured, irregular)
- **Physics Attention Module**: Integrates the Erwin transformer for token interaction
- **Hierarchical Processing**: Encoder-decoder structure captures information at multiple scales
- **Ball Multi-Head Self-Attention**: The core attention mechanism that replaces standard transformer attention

Mathematical Foundation
-----------------------

The hierarchical ball attention used in HAET provides significant computational advantages:

- **Computational Complexity**: O(n) vs O(n²) for standard attention
- **Receptive Field**: Global through hierarchical structure
- **Position Encoding**: Relative position encoding within each ball

Performance Benefits
--------------------

HAET achieves several key advantages over previous approaches:

1. **Computational Efficiency**: Linear complexity in the number of tokens
2. **Memory Efficiency**: Reduced memory footprint compared to standard transformers
3. **Scalability**: Can handle much larger meshes with more slices
4. **Multi-scale Features**: Captures both local and global patterns efficiently
