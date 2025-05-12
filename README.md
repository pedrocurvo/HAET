<div align="center">

<h3>HAET: Hierarchical Attention Erwin Transolver</h3>

<b> Authors: </b> <a href="https://pedrocurvo.com/">Pedro M. P. Curvo</a>, <a href="https://www.google.com">Mohammadmahdi Rahimi</a>, <a href="https://www.google.com">Salvador Torpes</a>

</div>

## Introduction

HAET (Hierarchical Attention Erwin Transolver) combines the strengths of two advanced architectures for processing complex mesh data: Transolver's token slicing approach ([Transolver](https://github.com/thuml/Transolver)) and Erwin's hierarchical ball attention mechanism ([Erwin](https://github.com/maxxxzdn/erwin)).

## Architecture Overview

### Problem Addressed

Traditional approaches face two main challenges:
- **Transolver**: While effective at processing mesh data through slicing, it still suffers from quadratic complexity in attention between slice tokens, limiting the number of slices that can be efficiently processed.
- **Erwin Transformer**: Excellent at hierarchical processing with ball-based attention, but struggles with very large point clouds due to computational demands.

### Our Solution

HAET addresses these limitations by:
1. Using Transolver's slicing mechanism to convert mesh data into tokens
2. Placing these tokens in a virtual cube
3. Processing them through Erwin's hierarchical ball attention instead of traditional self-attention

### Key Components

- **Mesh Tokenization**: Converts mesh structures (2D/3D/irregular) into token representations
- **Virtual Spatial Embedding**: Places tokens in a 3D space for hierarchical processing
- **Ball Hierarchical Attention**: Processes tokens using Erwin's ball-based attention for linear complexity
- **Multi-scale Processing**: Handles relationships at different scales through Erwin's encoder-decoder structure

## Benefits

- **Computational Efficiency**: Replaces quadratic attention complexity with linear-complexity ball attention
- **Scale Flexibility**: Allows processing of larger meshes with more slices
- **Hierarchical Features**: Captures multi-scale relationships in mesh data
- **Cross-domain Applicability**: Works with various mesh types (structured 2D/3D and irregular)

## Applications

Designed for computationally demanding physics and engineering simulations, including:
- Computational fluid dynamics
- Structural analysis
- Physical system modeling

## Acknowledgements

We appreciate the following GitHub repositories for their valuable code base and datasets:

- [Neural Operator](https://github.com/neuraloperator/neuraloperator)
- [Geo-FNO](https://github.com/neuraloperator/Geo-FNO)
- [Latent-Spectral-Models](https://github.com/thuml/Latent-Spectral-Models)
- [AirfRANS](https://github.com/Extrality/AirfRANS)
- [Transolver](https://github.com/thuml/Transolver)
  ```
  @inproceedings{wu2024Transolver,
    title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
    author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
    booktitle={International Conference on Machine Learning},
    year={2024}
  }
  ```
- [Erwin](https://github.com/maxxxzdn/erwin)
  ```
  @inproceedings{zhdanov2025erwin,
    title={Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems}, 
    author={Maksim Zhdanov and Max Welling and Jan-Willem van de Meent},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
  }
  ```