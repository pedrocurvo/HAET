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


cd $HOME/HAET/benchmarks/02.1-Car-Design-ShapeNetCar  

srun python main_evaluation.py --cfd_model=ErwinFlashTransformer --data_dir data/shapenet_car/mlcfd_data/training_data --save_dir data/shapenet_car/mlcfd_data/preprocessed_data