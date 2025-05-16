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

HAET (Hierarchical Attention Erwin Transolver) is a hybrid architecture designed to process mesh-based physical systems at industrial scale by merging the strengths of two state-of-the-art methods: [Transolver](https://github.com/thuml/Transolver) and [Erwin](https://github.com/maxxxzdn/erwin).

Transolver++ introduces a slice-based attention mechanism that significantly reduces the number of tokens required for mesh processing, enabling the handling of millions of points. However, attention within slices still scales quadratically with the number of slices, which limits scalability beyond 32 or 64 slices. Erwin, on the other hand, uses a tree-based hierarchical attention mechanism with ball grouping, reducing attention complexity from O(N²) to O(N), but it struggles with large-scale problems and can lose geometric context in the ball hierarchy.

HAET resolves both limitations by using Transolver++ to generate physical-aware slices, then computing their center-of-mass embeddings (eidetic states) and processing them through Erwin’s hierarchical ball attention. This design enables HAET to efficiently capture global and multi-scale interactions while remaining scalable to extremely large point clouds.

## Architecture Overview

### Key Challenges

- ⚠️ Quadratic Attention Bottleneck: Standard and slice-level attention require O(N²) operations, limiting scalability.

- 🧱 Loss of Geometry in Coarse Attention: Ball-based methods may abstract away geometric structure.

- 💾 Memory Pressure on Full Attention Models: Full-resolution attention is impractical for large meshes.

### Our Solution

HAET introduces a modular hybrid pipeline:

- 🧩 Rep-Slice Tokenization (from Transolver++): Soft clustering of points into slices based on physical semantics, guided by an adaptive temperature mechanism.

- 🧠 Eidetic States: Each slice becomes a memory-efficient representation that summarizes physical and spatial properties.

- 🪄 Hierarchical Ball Attention (from Erwin): Eidetic tokens are passed into Erwin, which computes efficient attention using a hierarchical ball-tree, scaling linearly with token count.

- 📍 Center-of-Mass Positional Encoding: Slice positions are derived from physical centroids, preserving geometry during pooling.

This pipeline allows HAET to scale beyond previous limitations while maintaining strong inductive biases from physics and geometry.

### Key Features

- 🔁 Linear Attention Complexity: Erwin replaces quadratic attention with a hierarchical mechanism over slices.

- 📈 Scalable Mesh Processing: Easily handles millions of points with low memory footprint.

- 🧬 Physical & Geometric Awareness: Combines Transolver++'s physical slices with Erwin's geometric hierarchy.

- 🌐 Multi-Scale Representation: Captures both local and global interactions through coarse-to-fine Erwin layers.

- 🔧 Adaptable Tokenization: Uses Ada-Temp to flexibly assign points to slice tokens based on learned importance.

## Applications

HAET is ideal for large-scale physics and engineering simulations, including:

- 💨 Computational fluid dynamics (CFD)

- 🧮 Mesh-based PDE solving

- 🏗️ Structural and thermal analysis

- ⚙️ General physical system modeling with spatial structure

### Experiments

We evaluate HAET on a variety of benchmarks, including:
- **Car Design**: Predicting aerodynamic properties of car shapes using a dataset of 3D meshes.

For more details on how to run the experiments, please refer to the README files in the respective benchmark folders:
- [Car Design Benchmark](benchmarks/02-Car-Design-ShapeNetCar/README.md).

## Acknowledgements

We appreciate the following GitHub repositories for their valuable code base and datasets:

- [Neural Operator](https://github.com/neuraloperator/neuraloperator)
- [Geo-FNO](https://github.com/neuraloperator/Geo-FNO)
- [Latent-Spectral-Models](https://github.com/thuml/Latent-Spectral-Models)
- [AirfRANS](https://github.com/Extrality/AirfRANS)
- 📘 [Transolver](https://github.com/thuml/Transolver)
  ```
  @inproceedings{wu2024Transolver,
    title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
    author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
    booktitle={International Conference on Machine Learning},
    year={2024}
  }
  ```
- 📘 [Transolver++ Paper](https://arxiv.org/abs/2404.02414)
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
- 📘 [Erwin](https://github.com/maxxxzdn/erwin)
  ```
  @inproceedings{zhdanov2025erwin,
    title={Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems}, 
    author={Maksim Zhdanov and Max Welling and Jan-Willem van de Meent},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
  }
  ```

