import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import wandb
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer
from torch.amp import autocast, GradScaler
import time
import psutil

parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_2D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='plas_Transolver')
parser.add_argument('--data_path', type=str, default='./data/plasticity/plas_N987_T20.mat')
parser.add_argument('--use_wandb', type=int, default=1, help='Whether to use Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='PDE-Solving', help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
parser.add_argument('--use_amp', type=int, default=1, help='Whether to use Automatic Mixed Precision')
parser.add_argument('--max_autotune', action='store_true', help='Use max-autotune mode for model compilation')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


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


def random_collate_fn(batch):
    shuffled_batch = []
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]

        num_timesteps = t.size(0)
        permuted_indices = torch.randperm(num_timesteps)

        t = t[permuted_indices]
        u = u[..., permuted_indices]

        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(0)
            shuffled_t = t.unsqueeze(0)
            shuffled_u = u.unsqueeze(0)
            shuffled_a = a.unsqueeze(0)
        else:
            shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
            shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
            shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
            shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)

    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)

    return shuffled_batch


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
    
    DATA_PATH = args.data_path

    N = 987
    ntrain = 900
    ntest = 80

    s1 = 101
    s2 = 31
    T = 20
    Deformation = 4

    r1 = 1
    r2 = 1
    s1 = int(((s1 - 1) / r1) + 1)
    s2 = int(((s2 - 1) / r2) + 1)

    data = scio.loadmat(DATA_PATH)
    input = torch.tensor(data['input'], dtype=torch.float)
    output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)
    print(input.shape, output.shape)
    x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
    x_train = x_train.reshape(ntrain, -1, 1)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = y_train.reshape(ntrain, -1, Deformation, T)
    x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
    x_test = x_test.reshape(ntest, -1, 1)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = y_test.reshape(ntest, -1, Deformation, T)
    print(x_train.shape, y_train.shape)

    x_normalizer = UnitTransformer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    x_normalizer.cuda()

    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print("Dataloading is over.")

    t = np.linspace(0, 1, T)
    t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
    t_train = t.repeat(ntrain, 1)
    t_test = t.repeat(ntest, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
                                               batch_size=args.batch_size, 
                                               shuffle=True, 
                                               collate_fn=random_collate_fn,
                                               drop_last=True,
                                               pin_memory=True, 
                                               num_workers=8,
                                               persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
                                              batch_size=args.batch_size, 
                                              shuffle=False,
                                              pin_memory=True, 
                                              num_workers=8,
                                              persistent_workers=True)

    print("Dataloading is over.")
    model = get_model(args)(space_dim=2,
                            n_hidden=args.n_hidden,
                            n_layers=args.n_layers,
                            Time_Input=True,
                            n_head=args.n_heads,
                            fun_dim=1,
                            out_dim=Deformation,
                            mlp_ratio=args.mlp_ratio,
                            slice_num=args.slice_num,
                            unified_pos=args.unified_pos,
                            H=s1,
                            W=s2).cuda()

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

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')
        test_l2_step = 0
        test_l2_full = 0
        showcase = 10
        id = 0
        
        print("Starting evaluation...")
        eval_pbar = tqdm(test_loader, desc="Evaluation", position=0)
        
        with torch.no_grad():
            for x, tim, fx, yy in eval_pbar:
                id += 1
                loss = 0
                x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                bsz = x.shape[0]

                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    
                    # Use AMP for evaluation if enabled
                    use_auto_cast = True if args.use_amp else False
                    with autocast('cuda', dtype=torch.bfloat16, enabled=use_auto_cast):
                        im = model(x, fx, T=input_T)
                    
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, im.unsqueeze(-1)), -1)

                if id < showcase:
                    print(id)
                    truth = y[0].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                    pred_vis = im[0].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                    truth_du = np.linalg.norm(truth[:, :, 2:], axis=-1)
                    pred_du = np.linalg.norm(pred_vis[:, :, 2:], axis=-1)

                    plt.axis('off')
                    plt.scatter(truth[:, :, 0], truth[:, :, 1], 10, truth_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 6)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.axis('off')
                    plt.scatter(pred_vis[:, :, 0], pred_vis[:, :, 1], 10, pred_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 6)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.axis('off')
                    plt.scatter(truth[:, :, 0], truth[:, :, 1], 10, pred_du[:, :] - truth_du[:, :], cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.2, 0.2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                
                # Update progress bar
                step_loss = loss.item()
                full_loss = myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                
                eval_pbar.set_postfix({
                    'step_loss': f'{step_loss/(T*bsz):.6f}',
                    'full_loss': f'{full_loss/bsz:.6f}',
                    'avg_step': f'{test_l2_step/(id*T*bsz):.6f}',
                    'avg_full': f'{test_l2_full/(id*bsz):.6f}',
                    'memory_mb': f'{get_memory_usage():.1f}'
                })
                
                # Log batch-level evaluation metrics
                if args.use_wandb:
                    wandb.log({
                        "eval_batch/step_loss": step_loss/(T*bsz),
                        "eval_batch/full_loss": full_loss/bsz,
                        "eval_batch/avg_step_loss": test_l2_step/(id*T*bsz),
                        "eval_batch/avg_full_loss": test_l2_full/(id*bsz),
                        "eval_batch/memory_used_mb": get_memory_usage(),
                        "eval_batch/sample_id": id
                    })

        print("test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(test_l2_step / ntest / T, test_l2_full / ntest))
        
        # Log final evaluation metrics to W&B
        if args.use_wandb:
            wandb.log({
                "test/step_loss": test_l2_step / ntest / T,
                "test/full_loss": test_l2_full / ntest,
                "test/memory_used_mb": get_memory_usage()
            })
            wandb.finish()
    else:
        print("Starting training...")
        epoch_pbar = tqdm(range(args.epochs), desc="Epochs", position=0)
        
        for ep in epoch_pbar:
            epoch_start_time = time.time()
            model.train()
            train_l2_step = 0
            batch_times = []

            # Training loop with progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epochs}", position=1, leave=False)
            
            for batch_idx, (x, tim, fx, yy) in enumerate(train_pbar):
                batch_start_time = time.time()
                x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                bsz = x.shape[0]

                use_auto_cast = True if args.use_amp else False
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_auto_cast):
                    for t in range(T):
                        y = yy[..., t:t + 1]
                        input_T = tim[:, t:t + 1].reshape(bsz, 1)  # B,step
                        im = model(x, fx, T=input_T)

                        loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        train_l2_step += loss.item()

                optimizer.zero_grad()
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

                scheduler.step()
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # Update training progress bar
                current_lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{current_lr:.2e}',
                    'mem_mb': f'{get_memory_usage():.1f}',
                    'batch_time': f'{batch_time:.3f}s'
                })
                
                # Log batch-level metrics every 10 batches
                if args.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "batch/loss": loss.item(),
                        "batch/learning_rate": current_lr,
                        "batch/memory_used_mb": get_memory_usage(),
                        "batch/batch_time": batch_time,
                        "batch/epoch": ep,
                        "batch/batch_idx": batch_idx
                    })

            train_loss = train_l2_step / ntrain / T
            epoch_time = time.time() - epoch_start_time
            
            # Evaluation
            model.eval()
            test_l2_step = 0
            test_l2_full = 0
            val_start_time = time.time()
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc="Validation", position=1, leave=False)
                for x, tim, fx, yy in val_pbar:
                    loss = 0
                    x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                    bsz = x.shape[0]

                    for t in range(T):
                        y = yy[..., t:t + 1]
                        input_T = tim[:, t:t + 1].reshape(bsz, 1)
                        
                        # Use AMP for validation
                        use_auto_cast = True if args.use_amp else False
                        with autocast('cuda', dtype=torch.bfloat16, enabled=use_auto_cast):
                            im = model(x, fx, T=input_T)
                        
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im.unsqueeze(-1)
                        else:
                            pred = torch.cat((pred, im.unsqueeze(-1)), -1)

                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                    
                    val_pbar.set_postfix({
                        'val_step_loss': f'{loss.item()/(T*bsz):.6f}',
                        'val_full_loss': f'{myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()/bsz:.6f}'
                    })

            val_time = time.time() - val_start_time
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_step_loss': f'{test_l2_step / ntest / T:.6f}',
                'val_full_loss': f'{test_l2_full / ntest:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'epoch_time': f'{epoch_time:.1f}s'
            })
            
            # Log epoch metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "epoch/train_step_loss": train_loss,
                    "epoch/val_step_loss": test_l2_step / ntest / T,
                    "epoch/val_full_loss": test_l2_full / ntest,
                    "epoch/learning_rate": scheduler.get_last_lr()[0],
                    "epoch/epoch_time": epoch_time,
                    "epoch/val_time": val_time,
                    "epoch/avg_batch_time": np.mean(batch_times) if batch_times else 0,
                    "epoch/memory_used_mb": get_memory_usage(),
                    "epoch/epoch": ep
                })

            print("Epoch {} , train_step_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(ep,
                                                                                                             train_loss,
                                                                                                             test_l2_step / ntest / T,
                                                                                                             test_l2_full / ntest))
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
                "final/train_step_loss": train_loss,
                "final/val_step_loss": test_l2_step / ntest / T,
                "final/val_full_loss": test_l2_full / ntest,
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
