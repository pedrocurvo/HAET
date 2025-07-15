#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=segment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

# Change to the correct directory
cd $HOME/HAET/benchmarks/03-Segmentation-Reduced8

# Create output directory if it doesn't exist
mkdir -p slurm_output
mkdir -p checkpoints
mkdir -p results

# Run the segmentation experiment
srun python exp_pipe.py \
--model Transolver_Irregular_Mesh \
--n-hidden 128 \
--n-heads 2 \
--n-layers 1 \
--mlp_ratio 2 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 1 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--eval 0 \
--grid_size 32 \
--voxel_size 0.1 \
--epochs 500 \
--dropout 0.0 \
--use_amp 1 \
--save_name semantic3d_transolver

