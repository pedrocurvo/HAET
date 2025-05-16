# HAET Models Directory

This directory contains the implementation of the Hierarchical Attention Erwin Transolver (HAET) models and their components. HAET combines the strengths of Transolver's token slicing approach with Erwin's hierarchical ball attention mechanism to efficiently process mesh data at multiple scales.

## Directory Structure

```
models/
├── __init__.py                       # Package initialization and model exports
├── HAETransolver_Irregular_Mesh.py   # Implementation for irregular meshes
├── HAETransolver_Structured_Mesh_2D.py # Implementation for 2D structured meshes
├── HAETransolver_Structured_Mesh_3D.py # Implementation for 3D structured meshes
├── PhysicsAttention/                 # Physics attention implementations
│   ├── __init__.py
│   ├── IrregularMesh.py              # Physics attention for irregular meshes
│   ├── StructuredMesh2D.py           # Physics attention for 2D structured meshes
│   └── StructuredMesh3D.py           # Physics attention for 3D structured meshes
└── components/                       # Shared components and utilities
    ├── __init__.py
    ├── embedding.py                  # Positional and feature embeddings
    ├── mlp.py                        # MLP implementations
    ├── balltree/                     # Ball tree data structure for efficient neighbor searching
    └── erwinflash/                   # Optimized ErwinTransformer using FlashAttention
```

## Main Model Classes

The library provides three main model variants for different mesh types:

### HAETransolver_Irregular_Mesh

Designed for processing irregular mesh structures common in finite element analysis, CFD, and general engineering applications. This model handles point clouds without a regular grid structure.

### HAETransolver_Structured_Mesh_2D

Specialized for 2D structured grid data often found in image-like representations of physical fields, such as weather maps, terrain analysis, or 2D simulations.

### HAETransolver_Structured_Mesh_3D

Handles 3D structured grid data for volumetric simulations like fluid dynamics, heat transfer, or electromagnetic field analysis.

## Architecture Components

### TransolverErwinBlock

Each model contains one or more `TransolverErwinBlock` instances, which implement the core HAET architecture:

1. **Rep-Slice with Ada-Temp**: Points are softly assigned to slice tokens using an adaptive temperature mechanism
2. **Eidetic States**: Memory-efficient token representations that capture essential physical properties
3. **Physics Attention Module**: Integrates with the Erwin transformer for slice token interaction
4. **Multi-Layer Perceptron**: Processes the updated token information

### Physics Attention

The `PhysicsAttention` directory contains implementations for different mesh types:

- `Physics_Attention_Irregular_Mesh`: For irregular geometries
- `Physics_Attention_Structured_Mesh_2D`: For 2D structured grids
- `Physics_Attention_Structured_Mesh_3D`: For 3D structured grids

These modules handle the integration of HAETransolver++ tokenization with Erwin's hierarchical ball attention.

### Shared Components

The `components` directory contains shared utilities used across the models:

- `embedding.py`: Defines positional and feature embeddings
- `mlp.py`: Contains MLP implementations for token processing
- `balltree/`: Ball tree data structure for efficient neighbor searching
- `erwinflash/`: Optimized implementation of ErwinTransformer using FlashAttention for increased speed

## How to Use

Here's a simple example of how to use the HAETransolver model for a 3D structured mesh problem:

```python
import torch
from models import HAETransolver_Structured_Mesh_3D

# Create model
model = HAETransolver_Structured_Mesh_3D(
    space_dim=3,
    n_layers=5,
    n_hidden=256,
    n_head=8,
    slice_num=32,
    ref=8,  # Reference grid size
    H=32, W=32, D=32,  # Mesh dimensions
    # ErwinTransformer parameters
    c_hidden=64,  # Hidden dimension for Erwin transformer
    ball_sizes=[0.2, 0.4],  # Ball sizes for hierarchical attention
    enc_num_heads=[4, 4],  # Number of heads in each encoder layer
    enc_depths=[2, 2],  # Number of layers in each encoder
    dec_num_heads=[4, 4],  # Number of heads in each decoder layer
    dec_depths=[2, 2],  # Number of layers in each decoder
    strides=[2, 2],  # Strides for downsampling
    rotate=45,  # Rotation angle
    decode=True,  # Whether to use decoder
    mp_steps=0,  # Number of message passing steps (default 0)
    embed=False  # Whether to embed the input
)

# Create input mesh data
mesh_size = (1, 32*32*32, 3)  # (batch_size, num_points, space_dim)
mesh_data = torch.rand(mesh_size)

# Process through model
output = model(mesh_data, None)

# output has shape (batch_size, num_points, out_dim)
```

## Key Features

- **Efficient Tokenization**: Rep-Slice with adaptive temperature for better token quality
- **Memory Optimization**: Eidetic states reduce memory usage by 50% compared to standard approaches
- **Linear-time Attention**: Erwin's ball attention replaces quadratic-complexity attention
- **Multi-scale Processing**: Hierarchical structure captures information at multiple scales
- **Versatile Architecture**: Works with various mesh types (structured 2D/3D and irregular)

## Parameters Explanation

### Basic Parameters

- `space_dim`: Spatial dimensionality of the input (2D or 3D)
- `n_layers`: Number of HAETransolver blocks in the model
- `n_hidden`: Hidden dimension size for feature processing
- `n_head`: Number of attention heads in the Physics Attention module
- `slice_num`: Number of slice tokens to use for Rep-Slice
- `ref`: Reference grid size for coordinate normalization

### Mesh-specific Parameters

- `H`, `W`, `D`: Dimensions of the structured mesh (only for structured mesh models)

### ErwinTransformer Parameters

- `c_hidden`: Hidden dimension for Erwin transformer
- `ball_sizes`: Ball sizes for different hierarchical levels
- `enc_num_heads`: Number of attention heads for each encoder layer
- `enc_depths`: Number of layers in each encoder
- `dec_num_heads`: Number of attention heads for each decoder layer
- `dec_depths`: Number of layers in each decoder
- `strides`: Strides for downsampling
- `rotate`: Rotation angle for ball attention
- `decode`: Whether to use the decoder
- `mp_steps`: Number of message passing steps (default 0)
- `embed`: Whether to embed the input

## Implementation Details

### Rep-Slice with Adaptive Temperature

```python
# Compute adaptive temperature (Ada-Temp): τ = τ0 + Linear(xi)
adaptive_temp = self.base_temp + self.ada_temp_linear(x_proj).clamp(min=-0.4, max=0.4)

# Compute Rep-Slice: Softmax(Linear(x) - log(-log(ε))) / τ
log_neg_log_epsilon = torch.log(-torch.log(torch.tensor(self.epsilon, device=x.device)))
slice_logits = self.in_project_slice(x_proj) - log_neg_log_epsilon
slice_weights = torch.softmax(slice_logits / adaptive_temp, dim=2)
```

### Eidetic States Computation

```python
# Compute weights norm and eidetic states
slice_norm = slice_weights.sum(2)
eidetic_states = torch.einsum("bhnc,bhng->bhgc", x_proj, slice_weights)
eidetic_states = eidetic_states / ((slice_norm + 1e-5)[:, :, :, None])
```

### Ball Attention Integration

The Physics Attention modules pass the eidetic states to the ErwinTransformer, which processes them using ball attention at multiple scales. This allows efficient global information flow while maintaining computational efficiency.
