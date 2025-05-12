API
===

This section documents the API for the HAET (Hierarchical Attention Erwin Transolver) library.

.. note::

   This documentation is auto-generated from docstrings and code structure. For more
   detailed implementation information, please refer to the source code.

Core Models
--------------

These are the main model implementations for different types of meshes.

Transolver Irregular Mesh
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: models.Transolver_Irregular_Mesh
   :members:
   :undoc-members:
   :show-inheritance:

Transolver Structured Mesh 2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: models.Transolver_Structured_Mesh_2D
   :members:
   :undoc-members:
   :show-inheritance:

Transolver Structured Mesh 3D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: models.Transolver_Structured_Mesh_3D
   :members:
   :undoc-members:
   :show-inheritance:

Physics Attention Modules
---------------------------

These modules implement the attention mechanisms for different mesh types.

Irregular Mesh
^^^^^^^^^^^^^^

.. automodule:: models.PhysicsAttention.IrregularMesh
   :members:
   :undoc-members:
   :show-inheritance:

Structured Mesh 2D
^^^^^^^^^^^^^^^^^

.. automodule:: models.PhysicsAttention.StructuredMesh2D
   :members:
   :undoc-members:
   :show-inheritance:

Structured Mesh 3D
^^^^^^^^^^^^^^^^^

.. automodule:: models.PhysicsAttention.StructuredMesh3D
   :members:
   :undoc-members:
   :show-inheritance:

Transformer Components
-----------------------

These components implement the hierarchical ball attention mechanism.

Erwin Transformer
^^^^^^^^^^^^^^^^

.. automodule:: models.components.erwin.erwin_transformer
   :members:
   :undoc-members:
   :show-inheritance:

Erwin Flash Transformer
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: models.components.erwinflash.erwin_flash
   :members:
   :undoc-members:
   :show-inheritance:

Ball Multi-Head Self-Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: models.components.erwin.components.attention
   :members:
   :undoc-members:
   :show-inheritance:

Utility Components
-------------------

Common utility components used throughout the library.

MLP
^^^

.. automodule:: models.components.mlp
   :members:
   :undoc-members:
   :show-inheritance:

Embedding
^^^^^^^^^

.. automodule:: models.components.embedding
   :members:
   :undoc-members:
   :show-inheritance:
