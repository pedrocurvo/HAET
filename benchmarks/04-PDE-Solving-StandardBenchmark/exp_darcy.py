import os
import argparse
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
import wandb
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model_dict import get_model
from utils.normalizer import UnitTransformer
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import time
import psutil

parser = argparse.ArgumentParser('Training Transolver')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='darcy_Transolver')
parser.add_argument('--data_path', type=str, default='/data/fno')
parser.add_argument('--use_wandb', type=int, default=1, help='Whether to use Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='PDE-Solving', help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
parser.add_argument('--use_amp', type=int, default=1, help='Whether to use Automatic Mixed Precision')
parser.add_argument('--max_autotune', action='store_true', help='Use max-autotune mode for model compilation')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = args.ntrain
ntest = 200
epochs = args.epochs
eval = args.eval
save_name = args.save_name


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


def central_diff(x: torch.Tensor, h, resolution):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x,
              (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


def main():
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=save_name,
        )
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler('cuda') if args.use_amp else None
    
    r = args.downsample
    h = int(((421 - 1) / r) + 1)
    s = h
    dx = 1.0 / s

    train_data = scio.loadmat(train_path)
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()
    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(test_path)
    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()
    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1)
    y_test = torch.from_numpy(y_test)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print("Dataloading is over.")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=True, 
                                               num_workers=8,
                                               persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                              batch_size=args.batch_size, 
                                              shuffle=False,
                                              pin_memory=True, 
                                              num_workers=8,
                                              persistent_workers=True)

    model = get_model(args).Model(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=1,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=s, W=s).cuda()

    # compile the model, autotuning for performance
    if args.max_autotune:
        print("Compiling model with max-autotune settings.")
        model = torch.compile(model, mode="max-autotune")
    else:
        print("Compiling model with default settings.")
        model = torch.compile(model)
    
    if args.use_wandb and not eval:
        wandb.watch(model, log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    print(model)
    total_params = count_parameters(model)
    
    # Log model parameters to wandb
    if args.use_wandb:
        wandb.log({"model/total_parameters": total_params})

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    de_x = TestLoss(size_average=False)
    de_y = TestLoss(size_average=False)

    if eval:
        print("model evaluation")
        print(s, s)
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        showcase = 10
        id = 0
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        print("Starting evaluation...")
        rel_err = 0.0
        eval_pbar = tqdm(test_loader, desc="Evaluation", position=0)
        
        with torch.no_grad():
            for x, fx, y in eval_pbar:
                id += 1
                x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                
                # Use AMP for evaluation if enabled
                use_auto_cast = True if args.use_amp else False
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_auto_cast):
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                    out = y_normalizer.decode(out)
                
                tl = myloss(out, y).item()
                rel_err += tl
                
                # Update progress bar
                eval_pbar.set_postfix({
                    'rel_err': f'{tl:.6f}',
                    'avg_rel_err': f'{rel_err/id:.6f}',
                    'memory_mb': f'{get_memory_usage():.1f}'
                })
                
                # Log batch-level evaluation metrics
                if args.use_wandb:
                    wandb.log({
                        "eval_batch/rel_error": tl,
                        "eval_batch/avg_rel_error": rel_err/id,
                        "eval_batch/memory_used_mb": get_memory_usage(),
                        "eval_batch/sample_id": id
                    })

                    if id < showcase:
                        print(id)
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(out[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/',
                                         "case_" + str(id) + "_pred.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(y[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_gt.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((y[0, :] - out[0, :]).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.clim(-0.0005, 0.0005)
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_error.pdf"))
                        plt.close()
                        # ============ #
                        plt.figure()
                        plt.axis('off')
                        plt.imshow((fx[0, :].unsqueeze(-1)).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                        plt.colorbar()
                        plt.savefig(
                            os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_input.pdf"))
                        plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))
        
        # Log final evaluation metrics to W&B
        if args.use_wandb:
            wandb.log({
                "test/rel_error": rel_err,
                "test/memory_used_mb": get_memory_usage()
            })
            wandb.finish()
    else:
        print("Starting training...")
        epoch_pbar = tqdm(range(args.epochs), desc="Epochs", position=0)
        
        for ep in epoch_pbar:
            epoch_start_time = time.time()
            model.train()
            train_loss = 0
            reg = 0
            batch_times = []

            # Training loop with progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epochs}", position=1, leave=False)
            
            for batch_idx, (x, fx, y) in enumerate(train_pbar):
                batch_start_time = time.time()
                
                x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                optimizer.zero_grad()

                # Use AMP for training if enabled
                use_auto_cast = True if args.use_amp else False
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_auto_cast):
                    out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)  # B, N , 2, fx: B, N, y: B, N
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                    l2loss = myloss(out, y)

                    out = rearrange(out.unsqueeze(-1), 'b (h w) c -> b c h w', h=s)
                    out = out[..., 1:-1, 1:-1].contiguous()
                    out = F.pad(out, (1, 1, 1, 1), "constant", 0)
                    out = rearrange(out, 'b c h w -> b (h w) c')
                    gt_grad_x, gt_grad_y = central_diff(y.unsqueeze(-1), dx, s)
                    pred_grad_x, pred_grad_y = central_diff(out, dx, s)
                    deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)
                    loss = 0.1 * deriv_loss + l2loss
                
                # Use scaler for backward pass if AMP is enabled
                if args.use_amp:
                    scaler.scale(loss).backward()
                    if args.max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                train_loss += l2loss.item()
                reg += deriv_loss.item()
                scheduler.step()
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # Update training progress bar
                current_lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix({
                    'l2_loss': f'{l2loss.item():.6f}',
                    'reg_loss': f'{deriv_loss.item():.6f}',
                    'lr': f'{current_lr:.2e}',
                    'mem_mb': f'{get_memory_usage():.1f}',
                    'batch_time': f'{batch_time:.3f}s'
                })
                
                # Log batch-level metrics every 10 batches
                if args.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "batch/l2_loss": l2loss.item(),
                        "batch/reg_loss": deriv_loss.item(),
                        "batch/total_loss": loss.item(),
                        "batch/learning_rate": current_lr,
                        "batch/memory_used_mb": get_memory_usage(),
                        "batch/batch_time": batch_time,
                        "batch/epoch": ep,
                        "batch/batch_idx": batch_idx
                    })

            train_loss /= ntrain
            reg /= ntrain
            epoch_time = time.time() - epoch_start_time
            
            # Validation
            model.eval()
            rel_err = 0.0
            id = 0
            val_start_time = time.time()
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc="Validation", position=1, leave=False)
                for x, fx, y in val_pbar:
                    id += 1
                    if id == 2:
                        vis = True
                    else:
                        vis = False
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    
                    # Use AMP for evaluation
                    use_auto_cast = True if args.use_amp else False
                    with autocast('cuda', dtype=torch.bfloat16, enabled=use_auto_cast):
                        out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                        out = y_normalizer.decode(out)
                    
                    tl = myloss(out, y).item()
                    rel_err += tl
                    
                    val_pbar.set_postfix({
                        'val_loss': f'{tl:.6f}',
                        'avg_val_loss': f'{rel_err/id:.6f}'
                    })

            rel_err /= ntest
            val_time = time.time() - val_start_time
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'reg_loss': f'{reg:.6f}',
                'val_loss': f'{rel_err:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'epoch_time': f'{epoch_time:.1f}s'
            })
            
            print("Epoch {} Reg : {:.5f} Train loss : {:.5f}".format(ep, reg, train_loss))
            print("rel_err:{}".format(rel_err))
            
            # Log epoch metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "epoch/train_loss": train_loss,
                    "epoch/reg_loss": reg,
                    "epoch/val_loss": rel_err,
                    "epoch/learning_rate": scheduler.get_last_lr()[0],
                    "epoch/epoch_time": epoch_time,
                    "epoch/val_time": val_time,
                    "epoch/avg_batch_time": np.mean(batch_times) if batch_times else 0,
                    "epoch/memory_used_mb": get_memory_usage(),
                    "epoch/epoch": ep
                })

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                checkpoint_path = os.path.join('./checkpoints', save_name + '.pt')
                torch.save(model.state_dict(), checkpoint_path)
                if args.use_wandb:
                    wandb.save(checkpoint_path)

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        final_checkpoint_path = os.path.join('./checkpoints', save_name + '.pt')
        torch.save(model.state_dict(), final_checkpoint_path)
        
        # Save final model to wandb and finish session
        if args.use_wandb:
            wandb.save(final_checkpoint_path)
            wandb.log({
                "final/train_loss": train_loss,
                "final/reg_loss": reg,
                "final/val_loss": rel_err,
                "final/total_parameters": total_params,
                "final/epochs_completed": args.epochs
            })
            wandb.finish()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time: {:.2f} seconds".format(total_time))
    
    # Log total time to wandb if enabled
    if args.use_wandb:
        wandb.log({"final/total_time_seconds": total_time})
