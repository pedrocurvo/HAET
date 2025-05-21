#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=erwintransolver
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

cd $HOME/HAET/benchmarks/04-PDE-Solving-StandardBenchmark || exit 1 # Change to your working directory

# Create environment if it doesn't exist under .conda
if [ ! -d "$HOME/.conda/envs/erwin" ]; then
    echo "Creating fresh conda environment..."
    conda create -y -n erwin python=3.9
    source activate erwin
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -U "huggingface_hub[cli]"
else
    echo "Using existing conda environment..."
    source activate erwin
    pip install -r requirements.txt
fi

# Check for and handle environment issues
if ! command -v python3 &> /dev/null; then
    echo "Python not found in path. Something is wrong with the environment."
    exit 1
fi

srun python exp_elas.py \
--model Transolver_Irregular_Mesh \
--n-hidden 128 \
--n-heads 8 \
--n-layers 8 \
--lr 0.001 \
--max_grad_norm 0.1 \
--batch-size 1 \
--slice_num 64 \
--unified_pos 0 \
--ref 8 \
--eval 0 \
--save_name elas_HAETransolver \