import numpy as np
import time, json, os
import torch
import torch.nn as nn
import wandb
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import psutil
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, val_loss):
        if self.mode == 'min':
            if val_loss < self.min_validation_loss - self.min_delta:
                self.min_validation_loss = val_loss
                self.counter = 0
                return True
        else:
            if val_loss > self.min_validation_loss + self.min_delta:
                self.min_validation_loss = val_loss
                self.counter = 0
                return True

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


def train(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()
    torch.cuda.empty_cache()  # Clear GPU memory before training
    
    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    
    # Track batch times for performance monitoring
    batch_times = []
    total_batches = len(train_loader)
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler()
    
    for batch_idx, (cfd_data, geom) in enumerate(train_loader):
        batch_start = time.time()
        
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
        
        # Forward pass with gradient computation timing and AMP autocast
        forward_start = time.time()
        with autocast():
            out = model((cfd_data, geom))
            targets = cfd_data.y

            loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
            loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
            loss_velo = loss_velo_var.mean()
            total_loss = loss_velo + reg * loss_press
        forward_time = time.time() - forward_start

        # Backward pass with gradient computation timing and AMP scaler
        backward_start = time.time()
        scaler.scale(total_loss).backward()
        
        # Gradient clipping with scaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        backward_time = time.time() - backward_start

        scheduler.step()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Log detailed batch metrics every 10 batches
        if batch_idx % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            memory_used = get_memory_usage()
            avg_batch_time = np.mean(batch_times[-10:]) if batch_times else 0
            eta = avg_batch_time * (total_batches - batch_idx)
            
            wandb.log({
                'batch/loss_pressure': loss_press.item(),
                'batch/loss_velocity': loss_velo.item(),
                'batch/total_loss': total_loss.item(),
                'batch/learning_rate': current_lr,
                'batch/memory_used_mb': memory_used,
                'batch/forward_time': forward_time,
                'batch/backward_time': backward_time,
                'batch/batch_time': batch_time,
                'batch/eta_seconds': eta
            })

    mean_loss_press = np.mean(losses_press)
    mean_loss_velo = np.mean(losses_velo)
    
    metrics = {
        "train/loss_pressure": mean_loss_press,
        "train/loss_velocity": mean_loss_velo,
        "train/total_loss": mean_loss_velo + reg * mean_loss_press,
        "train/learning_rate": scheduler.get_last_lr()[0],
        "train/avg_batch_time": np.mean(batch_times),
        "train/memory_used_mb": get_memory_usage()
    }
    wandb.log(metrics)

    return mean_loss_press, mean_loss_velo


@torch.no_grad()
def test(device, model, test_loader):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        
        # Use autocast for validation as well for consistent precision
        with autocast():
            out = model((cfd_data, geom))
            targets = cfd_data.y

            loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
            loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
            loss_velo = loss_velo_var.mean()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

    mean_loss_press = np.mean(losses_press)
    mean_loss_velo = np.mean(losses_velo)
    
    # Log validation metrics
    wandb.log({
        "val/loss_pressure": mean_loss_press,
        "val/loss_velocity": mean_loss_velo,
        "val/total_loss": mean_loss_velo + mean_loss_press,
    })

    return mean_loss_press, mean_loss_velo


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1, val_iter=1, coef_norm=[]):
    # Create checkpoint directory
    checkpoint_dir = Path(path) / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="car-design-shapenet",
        config={
            **hparams,
            "architecture": Net.__class__.__name__,
            "regularization": reg,
            "dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "device": device,
            "early_stopping_patience": 7,
            "gradient_clip_norm": 1.0,
            "amp_enabled": True,  # Log AMP usage
        }
    )
    
    model = Net.to(device)
    wandb.watch(model, log="all", log_freq=100)  # Log model gradients and parameters
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        final_div_factor=1000.,
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-6)
    best_val_loss = float('inf')
    
    start = time.time()
    train_loss, val_loss = 1e5, 1e5
    
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        epoch_start = time.time()
        
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        loss_velo, loss_press = train(device, model, train_loader, optimizer, lr_scheduler, reg=reg)
        train_loss = loss_velo + reg * loss_press
        del train_loader

        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1)
            loss_velo, loss_press = test(device, model, val_loader)
            val_loss = loss_velo + reg * loss_press
            del val_loader

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_dir / f'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, str(best_model_path))
                wandb.save(str(best_model_path))

            # Early stopping check
            if early_stopping(val_loss):
                if early_stopping.early_stop:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            pbar_train.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                'best_val': f'{best_val_loss:.6f}'
            })
        else:
            pbar_train.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start
        wandb.log({
            'epoch/train_loss': train_loss,
            'epoch/val_loss': val_loss if val_iter is not None else None,
            'epoch/learning_rate': lr_scheduler.get_last_lr()[0],
            'epoch/time_seconds': epoch_time,
            'epoch/best_val_loss': best_val_loss,
            'epoch/memory_used_mb': get_memory_usage()
        })

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    
    # Save final model
    final_model_path = path + os.sep + f'model_{hparams["nb_epochs"]}.pth'
    torch.save({
        'epoch': hparams['nb_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
    }, final_model_path)
    wandb.save(final_model_path)

    if val_iter is not None:
        log_data = {
            'nb_parameters': params_model,
            'time_elapsed': time_elapsed,
            'hparams': hparams,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'coef_norm': list(coef_norm),
            'early_stopped': early_stopping.early_stop,
            'final_epoch': epoch + 1
        }
        
        # Log final metrics to wandb
        wandb.log({
            "final/train_loss": train_loss,
            "final/val_loss": val_loss,
            "final/best_val_loss": best_val_loss,
            "final/time_elapsed": time_elapsed,
            "final/nb_parameters": params_model,
            "final/early_stopped": early_stopping.early_stop,
            "final/epochs_trained": epoch + 1
        })
        
        log_path = path + os.sep + f'log_{hparams["nb_epochs"]}.json'
        with open(log_path, 'a') as f:
            json.dump(log_data, f, indent=12, cls=NumpyEncoder)
        wandb.save(log_path)
    
    wandb.finish()
    return model
