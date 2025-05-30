#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=erwin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

cd $HOME/HAET/benchmarks/04-PDE-Solving-StandardBenchmark
srun python exp_airfoil.py \
--model HAETransolver_Structured_Mesh_2D \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 4 \
--slice_num 128 \
--unified_pos 0 \
--ref 8 \
--eval 0 \
--save_name HAET_Airfoil 