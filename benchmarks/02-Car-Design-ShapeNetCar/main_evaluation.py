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
from torch.cuda.amp import autocast

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--nb_epochs', default=200, type=float)
parser.add_argument('--wandb_project', default='car-design-shapenet-eval', type=str)
parser.add_argument('--wandb_entity', default=None, type=str)
parser.add_argument('--disable_wandb', action='store_true')
args = parser.parse_args()
print(args)

# Initialize wandb if not disabled
if not args.disable_wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "cfd_model": args.cfd_model,
            "fold_id": args.fold_id,
            "r": args.r,
            "weight": args.weight,
            "nb_epochs": args.nb_epochs,
            "cfd_mesh": args.cfd_mesh
        },
        name=f"eval_{args.cfd_model}_fold{args.fold_id}_ep{args.nb_epochs}"
    )

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

train_data, val_data, coef_norm, vallst = load_train_val_fold_file(args, preprocessed=True)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'

# 1. Reconstruct the model with training-time hyperparameters
model = Model(n_hidden=256, n_layers=1, space_dim=3,
                  fun_dim=4,
                  n_head=8,
                  mlp_ratio=2, out_dim=4,
                  slice_num=32,
                  radius=args.r,
                  unified_pos=0).to(device)

# Load checkpoint
checkpoint_path = os.path.join(path, f'model_{args.nb_epochs}.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load only the model weights
model.load_state_dict(checkpoint['model_state_dict'])

test_loader = DataLoader(val_ds, batch_size=1)

if not os.path.exists('./results/' + args.cfd_model + '/'):
    os.makedirs('./results/' + args.cfd_model + '/')

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
        
        with autocast():
            out = model((cfd_data, geom))
            
        toc = time.time()
        targets = cfd_data.y

        if coef_norm is not None:
            mean = torch.tensor(coef_norm[2]).to(device)
            std = torch.tensor(coef_norm[3]).to(device)
            pred_press = out[cfd_data.surf, -1] * std[-1] + mean[-1]
            gt_press = targets[cfd_data.surf, -1] * std[-1] + mean[-1]
            pred_velo = out[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            gt_velo = targets[~cfd_data.surf, :-1] * std[:-1] + mean[:-1]
            out_denorm = out * std + mean
            y_denorm = targets * std + mean

        np.save('./results/' + args.cfd_model + '/' + str(index) + '_pred.npy', out_denorm.detach().cpu().numpy())
        np.save('./results/' + args.cfd_model + '/' + str(index) + '_gt.npy', y_denorm.detach().cpu().numpy())

        pred_coef = cal_coefficient(vallst[index].split('/')[1], pred_press[:, None].detach().cpu().numpy(),
                                    pred_velo.detach().cpu().numpy(), root=args.data_dir)
        gt_coef = cal_coefficient(vallst[index].split('/')[1], gt_press[:, None].detach().cpu().numpy(),
                                  gt_velo.detach().cpu().numpy(), root=args.data_dir)

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
    
    # Log summary metrics to wandb
    if not args.disable_wandb:
        # Create correlation plot
        if len(gt_coef_list) > 0:
            correlation_data = [[g, p] for g, p in zip(gt_coef_list, pred_coef_list)]
            table = wandb.Table(data=correlation_data, columns=["Ground Truth", "Prediction"])
            wandb.log({
                "coefficient_correlation": wandb.plot.scatter(
                    table, "Ground Truth", "Prediction", title="Drag Coefficient Correlation"
                )
            })
        
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
        
        # Finalize wandb run
        wandb.finish()
