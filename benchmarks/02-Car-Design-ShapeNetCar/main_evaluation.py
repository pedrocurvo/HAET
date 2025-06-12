"""
Evaluation script for ErwinTransolver model on the ShapeNetCar dataset.
This script handles model evaluation, metrics calculation, and result reporting.
"""
import os
import torch
import argparse
import yaml
import numpy as np
import time
import wandb
from torch import nn
from torch_geometric.loader import DataLoader
from utils.drag_coefficient import cal_coefficient
from dataset.load_dataset import load_train_val_fold_file
from dataset.dataset import GraphDataset
import scipy as sc
from models.Transolver import Model
from tqdm import tqdm
from torch.amp import autocast
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.tri as mtri
from utils.visualization import visualize_car_and_slices
from matplotlib.colors import Normalize
import matplotlib.tri as mtri


def parse_arguments():
    """
    Parse command-line arguments for the evaluation process.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate the ErwinTransolver model on ShapeNetCar dataset")
    
    # Data and paths
    parser.add_argument('--data_dir', default='/data/shapenet_car/mlcfd_data/training_data',
                        help='Directory containing the training data')
    parser.add_argument('--save_dir', default='/data/shapenet_car/mlcfd_data/preprocessed_data',
                        help='Directory to save preprocessed data')
    
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
    parser.add_argument('--slice_num', default=32, type=int,
                        help='Number of slices for the model')
    parser.add_argument('--unified_pos', default=0, type=int,
                        help='Whether to use unified position encoding')
    
    # Evaluation parameters
    parser.add_argument('--weight', default=0.5, type=float,
                        help='Weight for regularization')
    parser.add_argument('--nb_epochs', default=200, type=int,
                        help='Number of training epochs')
    
    # Hardware and run configuration
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--fold_id', default=0, type=int,
                        help='Fold ID for cross-validation')
    parser.add_argument('--experiment_name', default=None, type=str,
                        help='Name of the experiment for organizing results')
                        
    # WandB configuration
    parser.add_argument('--wandb_project', default='car-design-shapenet-eval', type=str,
                        help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='WandB entity name')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable WandB logging')
                        
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of car mesh and slice weights')
    parser.add_argument('--visualize_samples', default='0', type=str,
                        help='Comma-separated indices of samples to visualize (e.g., "0,1,2")')
    
    return parser.parse_args()


# Get command line arguments
args = parse_arguments()
print(args)

def init_wandb(args):
    """
    Initialize Weights & Biases logging if not disabled.
    
    Args:
        args: Command-line arguments
    """
    if not args.disable_wandb:
        # Create a descriptive run name
        run_name = f"eval_{args.cfd_model}_fold{args.fold_id}_ep{args.nb_epochs}"
        if args.experiment_name:
            run_name = f"{args.experiment_name}_{run_name}"
        
        # Initialize wandb with all relevant config parameters
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "cfd_model": args.cfd_model,
                "fold_id": args.fold_id,
                "r": args.r,
                "weight": args.weight,
                "nb_epochs": args.nb_epochs,
                "cfd_mesh": args.cfd_mesh,
                "n_hidden": args.n_hidden,
                "n_layers": args.n_layers,
                "space_dim": args.space_dim,
                "fun_dim": args.fun_dim,
                "n_head": args.n_head,
                "mlp_ratio": args.mlp_ratio,
                "out_dim": args.out_dim,
                "slice_num": args.slice_num,
                "unified_pos": args.unified_pos,
                "experiment_name": args.experiment_name
            },
            name=run_name
        )


# Initialize wandb if not disabled
init_wandb(args)

def setup_device(args):
    """
    Set up and return the appropriate device (CPU/GPU) for evaluation.
    
    Args:
        args: Command-line arguments
        
    Returns:
        torch.device: The device to use for evaluation
    """
    n_gpu = torch.cuda.device_count()
    use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    return device


def load_dataset(args):
    """
    Load and prepare the validation dataset.
    
    Args:
        args: Command-line arguments
        
    Returns:
        tuple: Validation dataset, normalization coefficient, and validation file list
    """
    train_data, val_data, coef_norm, vallst = load_train_val_fold_file(args, preprocessed=True)
    val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
    return val_ds, coef_norm, vallst


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


# Set up device
device = setup_device(args)

# Load dataset
val_ds, coef_norm, vallst = load_dataset(args)

# Determine path based on experiment name
if args.experiment_name:
    path = f'metrics/{args.experiment_name}/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
else:
    path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'

print(f"Loading model from: {path}")

# Create the model with the specified parameters
model = create_model(args)

# Load checkpoint
checkpoint_path = os.path.join(path, f'model_{args.nb_epochs}.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)

# Process state dict to remove "_orig_mod." prefix from keys (added by torch.compile)
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('_orig_mod.'):
        new_key = key[len('_orig_mod.'):]
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# Load the processed state dict
model.load_state_dict(new_state_dict)

test_loader = DataLoader(val_ds, batch_size=1)

# Create results directory based on experiment name
if args.experiment_name:
    results_dir = f'./results/{args.experiment_name}/{args.cfd_model}/'
else:
    results_dir = f'./results/{args.cfd_model}/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Parse visualization samples if visualization is requested
visualize_samples = []
if args.visualize and args.visualize_samples:
    visualize_samples = [int(idx.strip()) for idx in args.visualize_samples.split(',') if idx.strip()]
    visualize_dir = os.path.join(results_dir, 'visualizations')
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)
        print(f"Created visualizations directory: {visualize_dir}")

with torch.no_grad():
    model.eval()
    criterion_func = nn.MSELoss(reduction='none')
    l2errs_press = []
    l2errs_velo = []
    mses_press = []
    mses_velo_var = []
    times = []
    gt_coef_list = []
    pred_coef_list = []
    coef_error = 0
    index = 0
    for index, (cfd_data, geom) in enumerate(tqdm(test_loader)):
        print(vallst[index])
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        tic = time.time()
        
        with autocast('cuda', dtype=torch.bfloat16):
            out = model((cfd_data, geom))
            
        toc = time.time()
        targets = cfd_data.y

        if coef_norm is not None:
            mean = torch.tensor(coef_norm[2]).to(device)
            std = torch.tensor(coef_norm[3]).to(device)
            pred_press = out[cfd_data.surf, -1] * std[-1] + mean[-1]
            gt_press = targets[cfd_data.surf, -1] * std[-1] + mean[-1]
            pred_surf_velo = out[cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            gt_surf_velo = targets[cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            pred_velo = out[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            gt_velo = targets[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            out_denorm = out * std + mean
            y_denorm = targets * std + mean

        np.save(os.path.join(results_dir, f"{index}_pred.npy"), out_denorm.float().detach().cpu().numpy())
        np.save(os.path.join(results_dir, f"{index}_gt.npy"), y_denorm.float().detach().cpu().numpy())

        pred_coef = cal_coefficient(vallst[index].split('/')[1], pred_press[:, None].float().detach().cpu().numpy(),
                                    pred_surf_velo.float().detach().cpu().numpy(), root=args.data_dir)
        gt_coef = cal_coefficient(vallst[index].split('/')[1], gt_press[:, None].float().detach().cpu().numpy(),
                                  gt_surf_velo.float().detach().cpu().numpy(), root=args.data_dir)

        gt_coef_list.append(gt_coef)
        pred_coef_list.append(pred_coef)
        coef_error_sample = abs(pred_coef - gt_coef) / gt_coef
        coef_error += coef_error_sample
        print(coef_error / (index + 1))

        l2err_press = torch.norm(pred_press - gt_press) / torch.norm(gt_press)
        l2err_velo = torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)

        mse_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        mse_velo_var = criterion_func(out[~cfd_data.surf, :-1], targets[~cfd_data.surf, :-1]).mean(dim=0)

        l2errs_press.append(l2err_press.cpu().numpy())
        l2errs_velo.append(l2err_velo.cpu().numpy())
        mses_press.append(mse_press.cpu().numpy())
        mses_velo_var.append(mse_velo_var.cpu().numpy())
        times.append(toc - tic)
        
        # Log individual sample results to wandb
        if not args.disable_wandb:
            wandb.log({
                f"sample_{index}/l2_error_pressure": l2err_press.cpu().numpy(),
                f"sample_{index}/l2_error_velocity": l2err_velo.cpu().numpy(),
                f"sample_{index}/mse_pressure": mse_press.cpu().numpy(),
                f"sample_{index}/inference_time": toc - tic,
                f"sample_{index}/gt_coefficient": gt_coef,
                f"sample_{index}/pred_coefficient": pred_coef,
                f"sample_{index}/coef_error": coef_error_sample,
                f"sample_{index}/filename": vallst[index]
            })
        
        index += 1

    gt_coef_list = np.array(gt_coef_list)
    pred_coef_list = np.array(pred_coef_list)
    spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
    mean_coef_error = coef_error / index
    print("rho_d: ", spear)
    print("c_d: ", mean_coef_error)
    l2err_press = np.mean(l2errs_press)
    l2err_velo = np.mean(l2errs_velo)
    rmse_press = np.sqrt(np.mean(mses_press))
    rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
    if coef_norm is not None:
        rmse_press *= coef_norm[3][-1]
        rmse_velo_var *= coef_norm[3][:-1]
    mean_inference_time = np.mean(times)
    
    print('relative l2 error press:', l2err_press)
    print('relative l2 error velo:', l2err_velo)
    print('press:', rmse_press)
    print('velo:', rmse_velo_var, np.sqrt(np.mean(np.square(rmse_velo_var))))
    print('time:', mean_inference_time)
    
    # Create a function to log final results to wandb
    def log_final_results_to_wandb():
        """Log final evaluation results to Weights & Biases."""
        if args.disable_wandb:
            return
            
        # Create correlation plot
        if len(gt_coef_list) > 0:
            correlation_data = [[g, p] for g, p in zip(gt_coef_list, pred_coef_list)]
            table = wandb.Table(data=correlation_data, columns=["Ground Truth", "Prediction"])
            wandb.log({
                "coefficient_correlation": wandb.plot.scatter(
                    table, "Ground Truth", "Prediction", 
                    title="Drag Coefficient Correlation"
                )
            })
        
        # Create a results summary dictionary
        result_dict = {
            "model": args.cfd_model,
            "experiment": args.experiment_name if args.experiment_name else "default",
            "fold_id": args.fold_id,
            "epochs": args.nb_epochs,
            "spearman_correlation": spear,
            "mean_coef_error": mean_coef_error,
            "l2_error_pressure": l2err_press,
            "l2_error_velocity": l2err_velo,
            "rmse_pressure": rmse_press,
            "rmse_velocity_mean": np.sqrt(np.mean(np.square(rmse_velo_var))),
            "mean_inference_time": mean_inference_time,
            "total_samples": index
        }
        
        # Log summary metrics
        wandb.log({
            "summary/spearman_correlation": spear,
            "summary/mean_coef_error": mean_coef_error,
            "summary/l2_error_pressure": l2err_press,
            "summary/l2_error_velocity": l2err_velo,
            "summary/rmse_pressure": rmse_press,
            "summary/rmse_velocity_mean": np.sqrt(np.mean(np.square(rmse_velo_var))),
            "summary/mean_inference_time": mean_inference_time,
            "summary/total_samples": index
        })
        
        # Save result summary as artifact
        import json
        with open("evaluation_results.json", "w") as f:
            json.dump(result_dict, f, indent=4)
        
        artifact = wandb.Artifact(
            name=f"eval_results_{args.cfd_model}_{args.fold_id}", 
            type="evaluation_results"
        )
        artifact.add_file("evaluation_results.json")
        wandb.log_artifact(artifact)
        
        # Finalize wandb run
        wandb.finish()
    
    # Log results to wandb
    # log_final_results_to_wandb()
    
    # Visualize car mesh and slice weights for selected samples
    if args.visualize and len(visualize_samples) > 0:
        print(f"\nGenerating visualizations for {len(visualize_samples)} samples...")
        for sample_idx in visualize_samples:
            visualize_car_and_slices(sample_idx, results_dir, model, val_ds, args)
