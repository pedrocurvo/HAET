import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time, json
import math
import psutil
import os

import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler

import wandb
from tqdm import tqdm

from pathlib import Path
import os.path as osp


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def is_nan_loss(loss):
    """Check if the loss is NaN or infinite"""
    return not torch.isfinite(loss).all() or math.isnan(loss.item())


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


def train(device, model, train_loader, optimizer, scheduler, criterion='MSE', reg=1):
    model.train()
    torch.cuda.empty_cache()  # Clear GPU memory before training
    
    avg_loss_per_var = torch.zeros(4, device=device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device=device)
    avg_loss_vol_var = torch.zeros(4, device=device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0

    # Initialize gradient scaler for AMP
    scaler = GradScaler('cuda')
    
    # Track batch times for performance monitoring
    batch_times = []
    total_batches = len(train_loader)

    for batch_idx, data in enumerate(train_loader):
        batch_start = time.time()
        
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        optimizer.zero_grad()
        
        # Forward pass with AMP autocast
        forward_start = time.time()
        with autocast('cuda', dtype=torch.bfloat16):
            out = model(data_clone)
            targets = data_clone.y

            if criterion == 'MSE' or criterion == 'MSE_weighted':
                loss_criterion = nn.MSELoss(reduction='none')
            elif criterion == 'MAE':
                loss_criterion = nn.L1Loss(reduction='none')
            
            loss_per_var = loss_criterion(out, targets).mean(dim=0)
            total_loss = loss_per_var.mean()
            loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
            loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
            loss_surf = loss_surf_var.mean()
            loss_vol = loss_vol_var.mean()
            
            if criterion == 'MSE_weighted':
                final_loss = loss_vol + reg * loss_surf
            else:
                final_loss = total_loss
        forward_time = time.time() - forward_start

        # Check for NaN loss
        if is_nan_loss(final_loss) or is_nan_loss(loss_surf) or is_nan_loss(loss_vol):
            print(f"🚨 NaN loss detected at batch {batch_idx}! Skipping batch...")
            wandb.log({
                'nan_event/batch_idx': batch_idx,
                'nan_event/loss_surf': loss_surf.item() if not is_nan_loss(loss_surf) else float('nan'),
                'nan_event/loss_vol': loss_vol.item() if not is_nan_loss(loss_vol) else float('nan'),
            })
            continue

        # Backward pass with AMP scaler
        backward_start = time.time()
        scaler.scale(final_loss).backward()
        
        # Gradient clipping with scaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        backward_time = time.time() - backward_start

        scheduler.step()
        
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol
        iter += 1
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Log detailed batch metrics every 10 batches
        if batch_idx % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            memory_used = get_memory_usage()
            avg_batch_time = np.mean(batch_times[-10:]) if batch_times else 0
            eta = avg_batch_time * (total_batches - batch_idx)
            
            wandb.log({
                'batch/loss_surf': loss_surf.item(),
                'batch/loss_vol': loss_vol.item(),
                'batch/total_loss': total_loss.item(),
                'batch/learning_rate': current_lr,
                'batch/memory_used_mb': memory_used,
                'batch/forward_time': forward_time,
                'batch/backward_time': backward_time,
                'batch/batch_time': batch_time,
                'batch/eta_seconds': eta,
            })

    mean_loss_surf = avg_loss_surf.cpu().data.numpy() / iter if iter > 0 else float('inf')
    mean_loss_vol = avg_loss_vol.cpu().data.numpy() / iter if iter > 0 else float('inf')
    mean_loss_per_var = avg_loss_per_var.cpu().data.numpy() / iter if iter > 0 else np.array([float('inf')] * 4)
    mean_loss_surf_var = avg_loss_surf_var.cpu().data.numpy() / iter if iter > 0 else np.array([float('inf')] * 4)
    mean_loss_vol_var = avg_loss_vol_var.cpu().data.numpy() / iter if iter > 0 else np.array([float('inf')] * 4)
    mean_total_loss = avg_loss.cpu().data.numpy() / iter if iter > 0 else float('inf')

    # Log training metrics
    wandb.log({
        "train/loss_surf": mean_loss_surf,
        "train/loss_vol": mean_loss_vol,
        "train/total_loss": mean_total_loss,
        "train/learning_rate": scheduler.get_last_lr()[0],
        "train/avg_batch_time": np.mean(batch_times) if batch_times else 0,
        "train/memory_used_mb": get_memory_usage(),
    })

    return mean_total_loss, mean_loss_per_var, mean_loss_surf_var, mean_loss_vol_var, mean_loss_surf, mean_loss_vol


@torch.no_grad()
def test(device, model, test_loader, criterion='MSE'):
    model.eval()
    avg_loss_per_var = np.zeros(4)
    avg_loss = 0
    avg_loss_surf_var = np.zeros(4)
    avg_loss_vol_var = np.zeros(4)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iter = 0

    for data in test_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)
        
        # Use autocast for validation as well for consistent precision
        with autocast('cuda', dtype=torch.bfloat16):
            out = model(data_clone)

            targets = data_clone.y
            if criterion == 'MSE' or 'MSE_weighted':
                loss_criterion = nn.MSELoss(reduction='none')
            elif criterion == 'MAE':
                loss_criterion = nn.L1Loss(reduction='none')

            loss_per_var = loss_criterion(out, targets).mean(dim=0)
            loss = loss_per_var.mean()
            loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim=0)
            loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim=0)
            loss_surf = loss_surf_var.mean()
            loss_vol = loss_vol_var.mean()

        avg_loss_per_var += loss_per_var.cpu().numpy()
        avg_loss += loss.cpu().numpy()
        avg_loss_surf_var += loss_surf_var.cpu().numpy()
        avg_loss_vol_var += loss_vol_var.cpu().numpy()
        avg_loss_surf += loss_surf.cpu().numpy()
        avg_loss_vol += loss_vol.cpu().numpy()
        iter += 1

    mean_loss_surf = avg_loss_surf / iter
    mean_loss_vol = avg_loss_vol / iter
    mean_total_loss = avg_loss / iter
    
    # Log validation metrics
    wandb.log({
        "val/loss_surf": mean_loss_surf,
        "val/loss_vol": mean_loss_vol,
        "val/total_loss": mean_total_loss,
    })

    return mean_total_loss, avg_loss_per_var / iter, avg_loss_surf_var / iter, avg_loss_vol_var / iter, mean_loss_surf, mean_loss_vol


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, criterion='MSE', reg=1, val_iter=10,
         name_mod='GraphSAGE', val_sample=True):
    '''
        Args:
        device (str): device on which you want to do the computation.
        train_dataset (list): list of the data in the training set.
        val_dataset (list): list of the data in the validation set.
        Net (class): network to train.
        hparams (dict): hyper parameters of the network.
        path (str): where to save the trained model and the figures.
        criterion (str, optional): chose between 'MSE', 'MAE', and 'MSE_weigthed'. The latter is the volumetric MSE plus the surface MSE computed independently. Default: 'MSE'.
        reg (float, optional): weigth for the surface loss when criterion is 'MSE_weighted'. Default: 1.
        val_iter (int, optional): number of epochs between each validation step. Default: 10.
        name_mod (str, optional): type of model. Default: 'GraphSAGE'.
    '''
    Path(path).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="airfrans-cfd",
        config={
            **hparams,
            "architecture": Net.__class__.__name__,
            "model_type": name_mod,
            "criterion": criterion,
            "regularization": reg,
            "dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "device": device,
            "gradient_clip_norm": 1.0,
            "amp_enabled": True,  # Log AMP usage
            "val_iter": val_iter,
            "val_sample": val_sample,
        },
    )

    model = Net.to(device)
    wandb.watch(model, log="all", log_freq=100)  # Log model gradients and parameters
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    start = time.time()

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []
    val_surf_list = []
    val_vol_list = []
    val_surf_var_list = []
    val_vol_var_list = []
    
    best_val_loss = float('inf')

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
            idx = torch.tensor(idx)

            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]

            if name_mod != 'PointNet' and name_mod != 'MLP':
                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device), r=hparams['r'], loop=True,
                                                           max_num_neighbors=int(hparams['max_neighbors'])).cpu()

            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled,
                                  batch_size=hparams['batch_size'],
                                  shuffle=True, 
                                  num_workers=8,
                                    pin_memory=True, 
                                    persistent_workers=True)
        del (train_dataset_sampled)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train(device, model, train_loader, optimizer,
                                                                                lr_scheduler, criterion, reg=reg)
        print('epoch: ' + str(epoch))
        print('train_loss： ' + str(train_loss))
        print('loss_vol： ' + str(loss_vol))
        print('loss_surf： ' + str(loss_surf))

        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        del (train_loader)

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

        # Initialize validation variables
        val_loss = None
        val_surf = None
        val_vol = None

        if val_iter is not None:
            if epoch % val_iter == val_iter - 1 or epoch == 0:
                if val_sample:
                    val_surf_vars, val_vol_vars, val_surfs, val_vols = [], [], [], []
                    for i in range(20):
                        val_dataset_sampled = []
                        for data in val_dataset:
                            data_sampled = data.clone()
                            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                            idx = torch.tensor(idx)

                            data_sampled.pos = data_sampled.pos[idx]
                            data_sampled.x = data_sampled.x[idx]
                            data_sampled.y = data_sampled.y[idx]
                            data_sampled.surf = data_sampled.surf[idx]

                            if name_mod != 'PointNet' and name_mod != 'MLP':
                                data_sampled.edge_index = nng.radius_graph(x=data_sampled.pos.to(device),
                                                                           r=hparams['r'], loop=True,
                                                                           max_num_neighbors=int(
                                                                               hparams['max_neighbors'])).cpu()

                            val_dataset_sampled.append(data_sampled)
                        val_loader_temp = DataLoader(val_dataset_sampled,
                                                      batch_size=1,
                                                      shuffle=True,
                                                        num_workers=8,
                                                        pin_memory=True,
                                                        persistent_workers=True)
                        del (val_dataset_sampled)

                        val_loss_temp, _, val_surf_var, val_vol_var, val_surf_temp, val_vol_temp = test(device, model, val_loader_temp,
                                                                                         criterion)
                        del (val_loader_temp)
                        val_surf_vars.append(val_surf_var)
                        val_vol_vars.append(val_vol_var)
                        val_surfs.append(val_surf_temp)
                        val_vols.append(val_vol_temp)
                    val_surf_var = np.array(val_surf_vars).mean(axis=0)
                    val_vol_var = np.array(val_vol_vars).mean(axis=0)
                    val_surf = np.array(val_surfs).mean(axis=0)
                    val_vol = np.array(val_vols).mean(axis=0)
                    val_loss = val_surf + val_vol  # Compute total validation loss
                else:
                    val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device, model, val_loader,
                                                                                     criterion)
                print("=====validation=====")
                print('epoch: ' + str(epoch))
                print('val_vol： ' + str(val_vol))
                print('val_surf： ' + str(val_surf))
                if criterion == 'MSE_weighted':
                    val_loss = reg * val_surf + val_vol
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = osp.join(path, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                    }, best_model_path)
                    wandb.save(best_model_path)
                
                val_surf_list.append(val_surf)
                val_vol_list.append(val_vol)
                val_surf_var_list.append(val_surf_var)
                val_vol_var_list.append(val_vol_var)
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf, best_val=best_val_loss)
            else:
                pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf, val_loss=val_loss, val_surf=val_surf)
        else:
            pbar_train.set_postfix(train_loss=train_loss, loss_surf=loss_surf)

        # Log epoch metrics
        epoch_metrics = {
            'epoch/train_loss': train_loss,
            'epoch/train_loss_surf': loss_surf,
            'epoch/train_loss_vol': loss_vol,
            'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
            'epoch/memory_used_mb': get_memory_usage(),
            'epoch/best_val_loss': best_val_loss,
        }
        
        if val_loss is not None:
            epoch_metrics.update({
                'epoch/val_loss': val_loss,
                'epoch/val_loss_surf': val_surf,
                'epoch/val_loss_vol': val_vol,
            })
        
        wandb.log(epoch_metrics)

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)
    val_surf_var_list = np.array(val_surf_var_list)
    val_vol_var_list = np.array(val_vol_var_list)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    
    # Save final model
    final_model_path = osp.join(path, f'model_{hparams["nb_epochs"]}.pth')
    torch.save({
        'epoch': hparams['nb_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss if 'val_loss' in locals() else None,
    }, final_model_path)
    wandb.save(final_model_path)
    
    # Also save the original format for backward compatibility
    torch.save(model, osp.join(path, 'model'))

    sns.set()
    fig_train_surf, ax_train_surf = plt.subplots(figsize=(20, 5))
    ax_train_surf.plot(train_loss_surf_list, label='Mean loss')
    ax_train_surf.plot(loss_surf_var_list[:, 0], label=r'$v_x$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 1], label=r'$v_y$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 2], label=r'$p$ loss')
    ax_train_surf.plot(loss_surf_var_list[:, 3], label=r'$\nu_t$ loss')
    ax_train_surf.set_xlabel('epochs')
    ax_train_surf.set_yscale('log')
    ax_train_surf.set_title('Train losses over the surface')
    ax_train_surf.legend(loc='best')
    fig_train_surf.savefig(osp.join(path, 'train_loss_surf.png'), dpi=150, bbox_inches='tight')
    wandb.log({"plots/train_loss_surf": wandb.Image(fig_train_surf)})

    fig_train_vol, ax_train_vol = plt.subplots(figsize=(20, 5))
    ax_train_vol.plot(train_loss_vol_list, label='Mean loss')
    ax_train_vol.plot(loss_vol_var_list[:, 0], label=r'$v_x$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 1], label=r'$v_y$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 2], label=r'$p$ loss')
    ax_train_vol.plot(loss_vol_var_list[:, 3], label=r'$\nu_t$ loss')
    ax_train_vol.set_xlabel('epochs')
    ax_train_vol.set_yscale('log')
    ax_train_vol.set_title('Train losses over the volume')
    ax_train_vol.legend(loc='best')
    fig_train_vol.savefig(osp.join(path, 'train_loss_vol.png'), dpi=150, bbox_inches='tight')
    wandb.log({"plots/train_loss_vol": wandb.Image(fig_train_vol)})

    if val_iter is not None:
        fig_val_surf, ax_val_surf = plt.subplots(figsize=(20, 5))
        ax_val_surf.plot(val_surf_list, label='Mean loss')
        ax_val_surf.plot(val_surf_var_list[:, 0], label=r'$v_x$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 1], label=r'$v_y$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 2], label=r'$p$ loss')
        ax_val_surf.plot(val_surf_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_val_surf.set_xlabel('epochs')
        ax_val_surf.set_yscale('log')
        ax_val_surf.set_title('Validation losses over the surface')
        ax_val_surf.legend(loc='best')
        fig_val_surf.savefig(osp.join(path, 'val_loss_surf.png'), dpi=150, bbox_inches='tight')
        wandb.log({"plots/val_loss_surf": wandb.Image(fig_val_surf)})

        fig_val_vol, ax_val_vol = plt.subplots(figsize=(20, 5))
        ax_val_vol.plot(val_vol_list, label='Mean loss')
        ax_val_vol.plot(val_vol_var_list[:, 0], label=r'$v_x$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 1], label=r'$v_y$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 2], label=r'$p$ loss')
        ax_val_vol.plot(val_vol_var_list[:, 3], label=r'$\nu_t$ loss')
        ax_val_vol.set_xlabel('epochs')
        ax_val_vol.set_yscale('log')
        ax_val_vol.set_title('Validation losses over the volume')
        ax_val_vol.legend(loc='best')
        fig_val_vol.savefig(osp.join(path, 'val_loss_vol.png'), dpi=150, bbox_inches='tight')
        wandb.log({"plots/val_loss_vol": wandb.Image(fig_val_vol)})

        if val_iter is not None:
            log_data = {
                'regression': 'Total',
                'loss': criterion,
                'nb_parameters': params_model,
                'time_elapsed': time_elapsed,
                'hparams': hparams,
                'train_loss_surf': train_loss_surf_list[-1],
                'train_loss_surf_var': loss_surf_var_list[-1],
                'train_loss_vol': train_loss_vol_list[-1],
                'train_loss_vol_var': loss_vol_var_list[-1],
                'val_loss_surf': val_surf_list[-1] if val_surf_list else None,
                'val_loss_surf_var': val_surf_var_list[-1] if len(val_surf_var_list) > 0 else None,
                'val_loss_vol': val_vol_list[-1] if val_vol_list else None,
                'val_loss_vol_var': val_vol_var_list[-1] if len(val_vol_var_list) > 0 else None,
                'best_val_loss': best_val_loss,
            }
            
            # Log final metrics to wandb
            wandb.log({
                "final/train_loss_surf": train_loss_surf_list[-1],
                "final/train_loss_vol": train_loss_vol_list[-1],
                "final/val_loss_surf": val_surf_list[-1] if val_surf_list else None,
                "final/val_loss_vol": val_vol_list[-1] if val_vol_list else None,
                "final/best_val_loss": best_val_loss,
                "final/time_elapsed": time_elapsed,
                "final/nb_parameters": params_model,
            })
            
            log_path = osp.join(path, name_mod + '_log.json')
            with open(log_path, 'a') as f:
                json.dump(log_data, f, indent=12, cls=NumpyEncoder)
            wandb.save(log_path)

    wandb.finish()
    return model
