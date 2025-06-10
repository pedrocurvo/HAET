"""
Main training script for ErwinTransolver model on the ShapeNetCar dataset.
This script handles command-line argument parsing, model initialization, and training.
"""
import os
import argparse
import torch

import train
from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
from models.Transolver import Model


def parse_arguments():
    """
    Parse command-line arguments for the training process.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train the ErwinTransolver model on ShapeNetCar dataset")
    
    # Data and paths
    parser.add_argument('--data_dir', default='/data/shapenet_car/mlcfd_data/training_data',
                        help='Directory containing the training data')
    parser.add_argument('--save_dir', default='/data/shapenet_car/mlcfd_data/preprocessed_data',
                        help='Directory to save preprocessed data')
    parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml',
                        help='Path to CFD parameters configuration file')
    
    # Model configuration
    parser.add_argument('--cfd_model', type=str, required=True,
                        help='Model type to use (e.g., ErwinTransolverNoEmbedding)')
    parser.add_argument('--cfd_mesh', action='store_true',
                        help='Use CFD mesh if specified')
    parser.add_argument('--r', default=0.2, type=float,
                        help='Radius parameter for the model')
    
    # Model architecture parameters
    parser.add_argument('--n_hidden', default=256, type=int,
                        help='Number of hidden units in the model')
    parser.add_argument('--n_layers', default=2, type=int,
                        help='Number of layers in the model')
    parser.add_argument('--space_dim', default=3, type=int,
                        help='Dimensionality of the space')
    parser.add_argument('--fun_dim', default=4, type=int,
                        help='Dimensionality of the function space')
    parser.add_argument('--n_head', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', default=2, type=int,
                        help='MLP ratio for transformer')
    parser.add_argument('--out_dim', default=4, type=int,
                        help='Output dimension')
    parser.add_argument('--slice_num', default=128, type=int,
                        help='Number of slices for the model')
    parser.add_argument('--unified_pos', default=0, type=int,
                        help='Whether to use unified position encoding')
    
    # Training parameters
    parser.add_argument('--weight', default=0.5, type=float,
                        help='Weight for regularization')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--nb_epochs', default=200, type=int,
                        help='Number of training epochs')
    parser.add_argument('--max_autotune', action='store_true',
                        help='Use max-autotune mode for model compilation')
    
    # Hardware and run configuration
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--fold_id', default=0, type=int,
                        help='Fold ID for cross-validation')
    parser.add_argument('--val_iter', default=10, type=int,
                        help='Validation frequency (iterations)')
    parser.add_argument('--preprocessed', default=1, type=int,
                        help='Whether to use preprocessed data (1) or not (0)')
    parser.add_argument('--experiment_name', default=None, type=str,
                        help='Name of the experiment for organizing results')
    
    return parser.parse_args()


def setup_device(args):
    """
    Set up and return the appropriate device (CPU/GPU) for training.
    
    Args:
        args: Command-line arguments
        
    Returns:
        torch.device: The device to use for training
    """
    n_gpu = torch.cuda.device_count()
    use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    return device


def load_datasets(args):
    """
    Load and prepare the training and validation datasets.
    
    Args:
        args: Command-line arguments
        
    Returns:
        tuple: Training dataset, validation dataset, and normalization coefficient
    """
    train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
    train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
    val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
    
    return train_ds, val_ds, coef_norm


def create_model(args):
    """
    Create and initialize the model based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        torch.nn.Module: Initialized model
    """
    model = Model(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        space_dim=args.space_dim,
        fun_dim=args.fun_dim,
        n_head=args.n_head,
        mlp_ratio=args.mlp_ratio,
        out_dim=args.out_dim,
        slice_num=args.slice_num,
        radius=args.r,
        unified_pos=args.unified_pos
    ).cuda()
    
    return model


def main():
    """
    Main function to run the training process.
    """
    # Parse command-line arguments
    args = parse_arguments()
    print(args)
    
    # Set up device and hyperparameters
    device = setup_device(args)
    hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}
    
    # Load datasets
    train_ds, val_ds, coef_norm = load_datasets(args)
    
    # Create the model
    model = create_model(args)
    # # Load checkpoint
    # path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
    # checkpoint_path = os.path.join(path, f'checkpoints/best_model.pth')
    # checkpoint = torch.load(checkpoint_path, map_location=device)

    # # Process state dict to remove "_orig_mod." prefix from keys (added by torch.compile)
    # state_dict = checkpoint['model_state_dict']
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     if key.startswith('_orig_mod.'):
    #         new_key = key[len('_orig_mod.'):]
    #         new_state_dict[new_key] = value
    #     else:
    #         new_state_dict[key] = value

    # # Load the processed state dict
    # model.load_state_dict(new_state_dict)
    
    # Create metrics directory
    if args.experiment_name:
        # If experiment_name is provided, use it as the top-level directory
        path = f'metrics/{args.experiment_name}/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
    else:
        # Fallback to original path structure if no experiment_name is provided
        path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created metrics directory: {path}")
    
    # Compile model for better performance if requested
    if hasattr(args, 'max_autotune') and args.max_autotune:
        print("Compiling model with max-autotune...")
        model = torch.compile(model, mode="max-autotune")
    else:
        print("Compiling model with default settings...")
        model = torch.compile(model)
    
    # Train the model
    model = train.main(
        device, 
        train_ds, 
        val_ds, 
        model, 
        hparams, 
        path, 
        val_iter=args.val_iter, 
        reg=args.weight,
        coef_norm=coef_norm
    )
    
    return model


if __name__ == "__main__":
    main()
