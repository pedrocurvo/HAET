import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models import HAETransolver_Irregular_Mesh, HAETransolver_Structured_Mesh_2D, HAETransolver_Structured_Mesh_3D

def get_model(args):
    model_dict = {
        'HAETransolver_Irregular_Mesh': HAETransolver_Irregular_Mesh, # for PDEs in 1D space or in unstructured meshes
        'HAETransolver_Structured_Mesh_2D': HAETransolver_Structured_Mesh_2D,
        'HAETransolver_Structured_Mesh_3D': HAETransolver_Structured_Mesh_3D,
    }
    return model_dict[args.model]
