# Semantic3D Point Cloud Segmentation

This code implements a 3D point cloud segmentation model for the Semantic3D/reduced-8 dataset.

## Dataset

The Semantic3D/reduced-8 dataset consists of labeled 3D point clouds. Each sample is represented as:

- A `.txt` file with point data (x, y, z, intensity, R, G, B)
- A matching `.labels` file with integer labels in [0, 8], where 0 means unlabeled

The data is organized into two folders:
- `data/training`: Contains point clouds with labels for training and validation
- `data/test`: Contains unlabeled point clouds for final testing (predictions only)

## Data Preparation

1. The dataset files should be organized in the appropriate folders:
   - `data/training`: Contains the training point clouds and labels
   - `data/test`: Contains the testing point clouds (without labels)
   
2. You can use the included `unzip.sh` script to extract the data:

```bash
cd erwinpp/03-Segmentation-Reduced8
mkdir -p data/training data/test
# Update the script to extract files to the correct folders
sbatch unzip.sh  # If using Slurm
# OR
bash unzip.sh
```

## Training

The script automatically splits the training data into training and validation sets using the `--val_split` parameter (default: 0.2 or 20% for validation).

To train the model, run:

```bash
python exp_pipe.py --data_path ./data/training --save_name sem3d_model
```

### Important Parameters

- `--model`: Model architecture (default: 'Transolver_Structured_Mesh_3D')
- `--grid_size`: Size of the 3D voxel grid (default: 32)
- `--voxel_size`: Size of each voxel for discretization (default: 0.1)
- `--val_split`: Percentage of training data to use for validation (default: 0.2)
- `--n-hidden`: Hidden dimension (default: 64)
- `--n-layers`: Number of transformer layers (default: 3)
- `--n-heads`: Number of attention heads (default: 4)
- `--batch-size`: Batch size (default: 8)
- `--epochs`: Number of training epochs (default: 500)
- `--lr`: Learning rate (default: 1e-3)
- `--use_wandb`: Enable Weights & Biases logging (default: 1)

## Evaluation

To evaluate a trained model on the validation set:

```bash
python exp_pipe.py --data_path ./data/training --save_name sem3d_model --eval 1
```

To generate predictions for the unlabeled test data:

```bash
python exp_pipe.py --data_path ./data/training --test_path ./data/test --save_name sem3d_model --eval 1 --test_on_unlabeled 1
```

This will:
1. Evaluate the model on the validation set
2. Generate prediction files for all test data in `./results/[save_name]/`

## Model Architecture

The model uses a 3D Transformer architecture (Transolver) to process the voxelized point cloud data. The process involves:

1. Loading and normalizing the point cloud data
2. Voxelizing the points into a regular 3D grid
3. Processing through transformer layers with self-attention
4. Producing per-voxel class predictions

## Classes

The Semantic3D/reduced-8 dataset contains 8 classes (plus class 0 for unlabeled points):

1. Man-made terrain
2. Natural terrain
3. High vegetation
4. Low vegetation
5. Buildings
6. Hard scape
7. Scanning artefacts
8. Cars 