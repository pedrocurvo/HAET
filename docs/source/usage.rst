Usage
=====

.. _installation:

Installation
------------

To install HAET, first clone the repository:

.. code-block:: console

   $ git clone https://github.com/pedrocurvo/ErwinTransolver.git
   $ cd ErwinTransolver

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
   from models import Transolver_Structured_Mesh_3D
   
   # Create model
   model = Transolver_Structured_Mesh_3D(
       space_dim=3,
       n_layers=5,
       n_hidden=256,
       n_head=8,
       slice_num=32,
       ref=8,  # Reference grid size
       H=32, W=32, D=32  # Mesh dimensions
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

   from models import Transolver_Structured_Mesh_3D
   model = Transolver_Structured_Mesh_3D(...)

2. **2D Structured Mesh**:

.. code-block:: python

   from models import Transolver_Structured_Mesh_2D
   model = Transolver_Structured_Mesh_2D(...)

3. **Irregular Mesh**:

.. code-block:: python

   from models import Transolver_Irregular_Mesh
   model = Transolver_Irregular_Mesh(...)

Example Applications
----------------

For practical examples, check the benchmarks directory:

.. code-block:: console

   $ cd benchmarks/02-Car-Design-ShapeNetCar
   $ python main.py

This runs the ShapeNetCar benchmark, demonstrating HAET's performance on computational fluid dynamics problems.

