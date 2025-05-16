Usage
=====

.. _installation:

Installation
------------

To install HAET, first clone the repository:

.. code-block:: console

   $ git clone https://github.com/pedrocurvo/HAET.git
   $ cd HAET

Then install the required dependencies:

.. code-block:: console

   $ pip install -e .

For development purposes, you can install additional dependencies:

.. code-block:: console

   $ pip install -r docs/requirements.txt

Basic Usage
----------

Here's a simple example of how to use HAET for a 3D structured mesh problem:

.. code-block:: python

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

Working with Different Mesh Types
-------------------------------

HAET supports various mesh types:

1. **3D Structured Mesh**:

.. code-block:: python

   from models import HAETransolver_Structured_Mesh_3D
   model = HAETransolver_Structured_Mesh_3D(...)

2. **2D Structured Mesh**:

.. code-block:: python

   from models import HAETransolver_Structured_Mesh_2D
   model = HAETransolver_Structured_Mesh_2D(...)

3. **Irregular Mesh**:

.. code-block:: python

   from models import HAETransolver_Irregular_Mesh
   model = HAETransolver_Irregular_Mesh(...)

Erwin Parameters Configuration
--------------------------

All HAETransolver models now include ErwinTransformer parameters that control the hierarchical ball attention mechanism:

.. code-block:: python

   # Required ErwinTransformer parameters 
   model = HAETransolver_Structured_Mesh_2D(
       # Basic parameters
       space_dim=2,
       n_layers=4,
       n_hidden=128,
       # ...
       
       # ErwinTransformer parameters
       c_hidden=64,             # Hidden dimension for Erwin transformer
       ball_sizes=[0.1, 0.3],   # Ball sizes for different attention layers
       enc_num_heads=[2, 4],    # Number of heads in each encoder layer
       enc_depths=[2, 2],       # Number of layers in each encoder
       dec_num_heads=[2, 4],    # Number of heads in each decoder layer
       dec_depths=[2, 2],       # Number of layers in each decoder
       strides=[2, 2],          # Strides for downsampling
       rotate=45,               # Rotation angle (degrees)
       decode=True,             # Whether to use decoder
       mp_steps=0,              # Number of message passing steps (default 0)
       embed=False              # Whether to embed the input
   )

These parameters control how the hierarchical ball attention mechanism processes the mesh data at different scales.

Example Applications
----------------

For practical examples, check the benchmarks directory:

.. code-block:: console

   $ cd benchmarks/02-Car-Design-ShapeNetCar
   $ python main.py

This runs the ShapeNetCar benchmark, demonstrating HAET's performance on computational fluid dynamics problems.

