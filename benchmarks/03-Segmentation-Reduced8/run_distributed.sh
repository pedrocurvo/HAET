#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=3
#SBATCH --job-name=segment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output/slurm_output_%A.out
#SBATCH --mem=120G

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

# Change to the correct directory
cd $HOME/HAET/benchmarks/03-Segmentation-Reduced8

# Get the number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found ${NUM_GPUS} GPUs"

# Create a directory for slurm output if it doesn't exist
mkdir -p slurm_output

export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# Run with torch.distributed.run (newer and recommended) instead of launch
srun python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    exp_pipe.py \
    --world_size ${NUM_GPUS} \
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