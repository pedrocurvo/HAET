Models
======

Core Model Structure
------------------

The HAET implementation is organized into several key modules:

.. code-block:: none

    models/
    ├── __init__.py
    ├── Transolver_Irregular_Mesh.py
    ├── Transolver_Structured_Mesh_2D.py
    ├── Transolver_Structured_Mesh_3D.py
    ├── PhysicsAttention/
    │   ├── __init__.py
    │   ├── IrregularMesh.py      # Implements Transolver++ with Erwin
    │   ├── StructuredMesh2D.py   # Implements Transolver++ with Erwin for 2D structured meshes
    │   └── StructuredMesh3D.py   # Implements Transolver++ with Erwin for 3D structured meshes
    └── components/
        ├── __init__.py
        ├── embedding.py
        ├── mlp.py
        ├── balltree/
        ├── erwin/
        └── erwinflash/

Main Model Classes
----------------

The library provides three main model variants for different mesh types:

1. ``Transolver_Irregular_Mesh``: For processing irregular mesh structures
2. ``Transolver_Structured_Mesh_2D``: For processing 2D structured grid data
3. ``Transolver_Structured_Mesh_3D``: For processing 3D structured grid data

Transolver Models
---------------

These models serve as the main entry point for the HAET architecture. Each model:

- Processes input mesh data
- Applies Rep-Slice with adaptive temperature through Physics Attention modules
- Creates and processes eidetic states for memory efficiency
- Integrates with Erwin for hierarchical processing

.. code-block:: python

    from models import Transolver_Structured_Mesh_3D
    
    model = Transolver_Structured_Mesh_3D(
        space_dim=3,  # Spatial dimensions
        n_layers=5,   # Number of transformer layers
        n_hidden=256, # Hidden dimension size
        n_head=8,     # Number of attention heads
        slice_num=32, # Number of slice tokens
    )

Physics Attention with Transolver++
---------------------------

The Physics Attention modules handle the core mechanism of tokenization and attention using Transolver++ approach:

- ``Physics_Attention_Irregular_Mesh``: For irregular geometries
- ``Physics_Attention_Structured_Mesh_2D``: For 2D structured grids
- ``Physics_Attention_Structured_Mesh_3D``: For 3D structured grids

These modules implement the Transolver++-Erwin integration with several key improvements:

1. **Rep-Slice with Ada-Temp**: Enhanced slicing with adaptive temperature for better token quality
2. **Eidetic States**: Memory-efficient token representations that reduce memory usage by 50%
3. **Hierarchical Ball Attention**: Replaces standard attention with Erwin's efficient ball attention

Example implementation of Transolver++ approach:

.. code-block:: python

    # Compute adaptive temperature (Ada-Temp): τ = τ0 + Linear(xi)
    adaptive_temp = self.base_temp + self.ada_temp_linear(x_proj).clamp(min=-0.4, max=0.4)
    
    # Compute Rep-Slice: Softmax(Linear(x) - log(-log(ε))) / τ
    log_neg_log_epsilon = torch.log(-torch.log(torch.tensor(self.epsilon, device=x.device)))
    slice_logits = self.in_project_slice(x_proj) - log_neg_log_epsilon
    slice_weights = torch.softmax(slice_logits / adaptive_temp, dim=2)
    
    # Compute weights norm and eidetic states
    slice_norm = slice_weights.sum(2)
    eidetic_states = torch.einsum("bhnc,bhng->bhgc", x_proj, slice_weights)
    eidetic_states = eidetic_states / ((slice_norm + 1e-5)[:, :, :, None])

Erwin Components
--------------

The Erwin components provide the hierarchical ball attention mechanism:

- ``ErwinTransformer``: Standard implementation
- ``ErwinFlashTransformer``: Optimized implementation using Flash Attention

Ball Attention Mechanism
---------------------

The core of HAET's efficiency is the Ball Multi-Head Self-Attention (BMSA):

1. Points are organized into balls (local neighborhoods)
2. Attention is computed within each ball
3. Hierarchical structure enables information flow across the entire mesh
4. Computational complexity scales linearly with the number of points

.. code-block:: python

    # Example Ball Multi-Head Self-Attention
    class BallMSA(nn.Module):
        def __init__(self, dim, num_heads, ball_size, dimensionality):
            # Initialize attention mechanism
            
        def forward(self, x, pos):
            # Compute attention within balls
            # Return updated features
