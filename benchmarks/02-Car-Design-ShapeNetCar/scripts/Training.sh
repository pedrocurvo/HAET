#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=car
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=80:00:00
#SBATCH --output=slurm_output/CAR/%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0


cd $HOME/HAET/benchmarks/02-Car-Design-ShapeNetCar

srun python main.py \
    --cfd_model=ErwinTransolverS64 \
    --data_dir data/shapenet_car/mlcfd_data/training_data \
    --save_dir data/shapenet_car/mlcfd_data/preprocessed_data \
    --batch_size 1 \
    --weight 0.5 \
    --unified_pos 0 \
    --nb_epochs 500 \
    --slice_num 64