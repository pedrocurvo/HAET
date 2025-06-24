#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=airfrans
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=80:00:00
#SBATCH --output=slurm_output/AIRFRANS/%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

srun python main.py \
    --model HAET -t full \
    --slice_num 64 \
    --my_path ./data/Dataset \
    --score 1