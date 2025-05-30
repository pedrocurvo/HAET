#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=plasticity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/slurm_output_plasticity_training_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

cd $HOME/HAET/benchmarks/04-PDE-Solving-StandardBenchmark || exit 1 # Change to your working directory

# # unninstall torch and torchvision if they exist
# srun pip uninstall -y torch torchvision torchaudio torch-cluster torch-scatter
# # Install the required packages
# srun pip cache purge
# srun pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
# srun pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html --force-reinstall
# srun pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html --force-reinstall

# Check for and handle environment issues
if ! command -v python3 &> /dev/null; then
    echo "Python not found in path. Something is wrong with the environment."
    exit 1
fi

echo "Running experiment on Plasticity dataset"

srun python exp_plas.py \
    --model HAETransolver_Structured_Mesh_2D \
    --n-hidden 128 \
    --n-heads 8 \
    --n-layers 8 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --batch-size 8 \
    --slice_num 64 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --save_name plas_HAETransolver

echo "Experiment completed. Check the output files for results."

