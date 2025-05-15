<div align="center">

<h3>HAET: Hierarchical Attention Erwin Transolver</h3>

<b> Authors: </b> <a href="https://pedrocurvo.com/">Pedro M. P. Curvo</a>, <a href="https://www.google.com">Mohammadmahdi Rahimi</a>, <a href="https://www.google.com">Salvador Torpes</a>

<p align="center">
	<img src="https://img.shields.io/github/license/pedrocurvo/ErwinTransolver" alt="license">
	<img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="python-version">
	<img src="https://img.shields.io/badge/PointClouds,%20PDESolver,%20Transformer,%20more-blue" alt="topics">
</p>

</div>

## Introduction

HAET (Hierarchical Attention Erwin Transolver) combines the strengths of two advanced architectures for processing complex mesh data: Transolver++'s token slicing approach with adaptive temperature and eidetic states ([Transolver](https://github.com/thuml/Transolver)) and Erwin's hierarchical ball attention mechanism ([Erwin](https://github.com/maxxxzdn/erwin)).

## Architecture Overview

### Problem Addressed

Traditional approaches face three main challenges:
- **Attention Complexity**: Standard attention mechanisms scale quadratically with the number of tokens, limiting computational efficiency for large meshes.
- **Memory Usage**: Traditional transformer architectures require substantial memory for processing complex mesh structures.
- **Multi-scale Features**: Capturing both local and global relationships in physics simulations requires efficient multi-scale feature extraction.

### Our Solution

HAET addresses these limitations by:
1. Implementing Transolver++ with adaptive temperature and eidetic states for improved memory efficiency
2. Using Rep-Slice with learnable temperature scaling for more accurate mesh tokenization
3. Processing tokens through Erwin's hierarchical ball attention instead of traditional self-attention

### Key Components

- **Rep-Slice Tokenization**: Converts mesh structures (2D/3D/irregular) into a reduced set of eidetic states using Rep-Slice with adaptive temperature
- **Eidetic States**: Memory-efficient token representations that capture essential physical properties
- **Ball Hierarchical Attention**: Processes tokens using Erwin's ball-based attention for linear complexity
- **Multi-scale Processing**: Handles relationships at different scales through Erwin's encoder-decoder structure

## Benefits

- **Memory Efficiency**: Reduces memory requirements by 50% through Transolver++'s eidetic states approach
- **Computational Efficiency**: Replaces quadratic attention complexity with linear-complexity ball attention
- **Adaptive Temperature**: Rep-Slice with adaptive temperature (Ada-Temp) for improved tokenization
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
- [Transolver++ Paper](https://arxiv.org/abs/2404.xxxxx)
  ```
  @misc{luo2025transolver,
    title={Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries},
    author={Huakun Luo and Haixu Wu and Hang Zhou and Lanxiang Xing and Yichen Di and Jianmin Wang and Mingsheng Long},
    year={2025},
    eprint={2502.02414},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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