"""
Transolver Models Package

This package provides different Transolver model implementations for solving
physical problems on various mesh types:

- Transolver_Irregular_Mesh: For irregular meshes and point clouds
- Transolver_Structured_Mesh_2D: For regular 2D grid data
- Transolver_Structured_Mesh_3D: For volumetric 3D grid data

Each model leverages a physics-informed transformer architecture with
specialized attention mechanisms tailored to the specific mesh type.
"""

from .Transolver_Irregular_Mesh import Model as Transolver_Irregular_Mesh
from .Transolver_Structured_Mesh_2D import Model as Transolver_Structured_Mesh_2D
from .Transolver_Structured_Mesh_3D import Model as Transolver_Structured_Mesh_3D

__all__ = [
    "Transolver_Irregular_Mesh",
    "Transolver_Structured_Mesh_2D",
    "Transolver_Structured_Mesh_3D",
]
