# HAET Benchmarks

This directory contains benchmarks and experiments for evaluating the Hierarchical Attention Erwin Transolver (HAET) architecture. Each subdirectory contains a specific benchmark or experiment designed to test different aspects of the model's performance.

## Benchmark Overview

### 02-Car-Design-ShapeNetCar

This benchmark evaluates HAET performance on predicting aerodynamic properties (primarily drag coefficient) of car shapes using the ShapeNetCar dataset. The benchmark tests the model's ability to process complex 3D mesh structures and extract meaningful physical features.

For detailed instructions on running this benchmark, see the [README](./02-Car-Design-ShapeNetCar/README.md) in the subdirectory.

### 02.1-Car-Design-ShapeNetCar

This directory contains benchmarks to evaluate the standalone Erwin model on the Car Design task. It's included as a direct comparison point since we didn't have results from Erwin on the same evaluations. This allows us to isolate the performance of the Erwin component and compare it with our integrated HAET approach.

Features:
- Same dataset and task as 02-Car-Design-ShapeNetCar
- Focuses solely on Erwin's performance

This benchmark helps us understand how much improvement comes from combining Erwin with HAETransolver versus using Erwin alone.

### 03-Segmentation-Reduced-8

This benchmark evaluates HAET on a 3D shape segmentation task with reduced resolution meshes (1/8 of original size). It tests the model's ability to understand local and global features in a semantic segmentation context. It was included to test the scalability of HAET on larger datasets and meshes.

Features:
- Point cloud segmentation
- Reduced resolution to test scalability
- Part-based semantic segmentation

## Adding New Benchmarks

To add a new benchmark:
1. Create a new directory with a clear naming convention
2. Include a README.md with benchmark description
3. Follow the structure of existing benchmarks for consistency
4. Add data loading, training, and evaluation scripts
5. Update this main README to include your benchmark
