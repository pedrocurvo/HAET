Architecture
============

HAET: Hierarchical Attention Erwin Transolver
---------------------------------------------

HAET combines the strengths of two powerful neural architectures to efficiently process mesh data at multiple scales:

1. **HAETransolver++**: An enhanced transformer-based architecture for solving PDEs on general geometries with adaptive temperature and eidetic states
2. **Erwin**: A tree-based hierarchical transformer for large-scale physical systems

Problem Statement
------------------

Traditional approaches face key challenges when processing complex mesh structures:

- **Memory Efficiency**: Standard transformer approaches require substantial memory for processing complex mesh structures.
- **Attention Complexity**: Standard attention mechanisms scale quadratically with the number of tokens, limiting computational efficiency.
- **Multi-scale Features**: Capturing both local and global relationships in physics simulations requires efficient multi-scale feature extraction.

Core Architecture
-----------------

.. image:: ../images/haet_architecture.png
   :width: 800
   :alt: HAET Architecture Diagram

*Note: Add an updated architecture diagram to the images folder*

The HAET architecture works as follows:

1. **Rep-Slice with Adaptive Temperature**:
   
   - The input mesh is processed using HAETransolver++'s Rep-Slice mechanism with adaptive temperature
   - Eidetic states are created that represent different regions of the mesh
   - This tokenization allows memory-efficient processing of complex geometries
   - Adaptive temperature enables more accurate slice assignments

2. **Eidetic States Processing**:
   
   - Eidetic states capture essential physical properties of the mesh
   - These states reduce memory requirements by 50% compared to standard approaches
   - Spatial organization enables hierarchical processing

3. **Hierarchical Ball Attention**:
   
   - Instead of standard self-attention between tokens (which is O(n²)), HAETransolver applies Erwin's hierarchical ball attention
   - Points are organized into a tree structure
   - Attention is computed within local "balls" of points, with sizes controlled by the ``ball_sizes`` parameter
   - Multi-scale features are captured through encoder-decoder architecture
   - The number of heads (``enc_num_heads``, ``dec_num_heads``) and layers (``enc_depths``, ``dec_depths``) at each level can be tuned
   - Downsampling between levels is controlled by the ``strides`` parameter

Key Components
--------------

The architecture consists of several key components:

- **Rep-Slice with Ada-Temp**: Enhanced slicing mechanism with adaptive temperature scaling
- **Eidetic States**: Memory-efficient token representations that capture essential physical properties
- **Physics Attention Module**: Integrates the Erwin transformer for token interaction
- **Hierarchical Processing**: Encoder-decoder structure captures information at multiple scales
- **Ball Multi-Head Self-Attention**: The core attention mechanism that replaces standard transformer attention

Configurable Erwin Parameters
----------------------------

The Erwin hierarchical transformer component can be fine-tuned through several parameters:

- **Hidden Dimension** (``c_hidden``): Controls the dimension of the hidden representations in the Erwin transformer
- **Ball Sizes** (``ball_sizes``): List of radii for the attention balls at different hierarchical levels
- **Encoder Configuration**: 
  - ``enc_num_heads``: Number of attention heads in each encoder level
  - ``enc_depths``: Number of layers in each encoder level
- **Decoder Configuration**:
  - ``dec_num_heads``: Number of attention heads in each decoder level
  - ``dec_depths``: Number of layers in each decoder level
- **Structural Parameters**:
  - ``strides``: Controls downsampling between hierarchical levels
  - ``rotate``: Rotation angle (in degrees) for ball attention queries
  - ``decode``: Whether to use the decoder pathway
  - ``mp_steps``: Number of message passing steps (default 0)
  - ``embed``: Whether to use additional embedding for input features

Mathematical Foundation
-----------------------

The hierarchical ball attention used in HAET provides significant computational advantages:

- **Computational Complexity**: O(n) vs O(n²) for standard attention
- **Receptive Field**: Global through hierarchical structure
- **Position Encoding**: Relative position encoding within each ball

Performance Benefits
--------------------

HAET achieves several key advantages over previous approaches:

1. **Memory Efficiency**: 50% reduction in memory footprint through the eidetic states approach
2. **Computational Efficiency**: Linear complexity in the number of tokens through ball attention
3. **Adaptive Tokenization**: Improved slice token quality through adaptive temperature scaling
4. **Scalability**: Can handle much larger meshes with more slices
5. **Multi-scale Features**: Captures both local and global patterns efficiently
