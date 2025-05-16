"""
Transolver Models Package

This package provides different Transolver model implementations for solving
physical problems on various mesh types:

- HAETransolver_Irregular_Mesh: For irregular mesh data
- HAETransolver_Structured_Mesh_2D: For regular 2D grid data
- HAETransolver_Structured_Mesh_3D: For volumetric 3D grid data

Each model leverages a physics-informed transformer architecture with
specialized attention mechanisms tailored to the specific mesh type.
"""

from .HAETransolver_Irregular_Mesh import Model as HAETransolver_Irregular_Mesh
from .HAETransolver_Structured_Mesh_2D import Model as HAETransolver_Structured_Mesh_2D
from .HAETransolver_Structured_Mesh_3D import Model as HAETransolver_Structured_Mesh_3D

__all__ = [
    "HAETransolver_Irregular_Mesh",
    "HAETransolver_Structured_Mesh_2D",
    "HAETransolver_Structured_Mesh_3D",
]
