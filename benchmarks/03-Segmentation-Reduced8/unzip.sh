#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=unzip
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output_%A.out

# Load module or activate conda env if needed
module load 2024  # or conda activate your_env

# Create data directories
cd $HOME/erwinpp/03-Segmentation-Reduced8/data


# Training files
# 7z x bildstein_station3_xyz_intensity_rgb.7z -otraining
# 7z x bildstein_station5_xyz_intensity_rgb.7z -otraining
# 7z x domfountain_station1_xyz_intensity_rgb.7z -otraining
# 7z x sem8_labels_training.7z -otraining

# Test files
# 7z x domfountain_station2_xyz_intensity_rgb.7z -otest
# 7z x domfountain_station3_xyz_intensity_rgb.7z -otest
7z x MarketplaceFeldkirch_Station4_rgb_intensity-reduced.txt.7z -otest

echo "Extraction complete!"