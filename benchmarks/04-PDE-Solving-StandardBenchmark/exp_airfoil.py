import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import *
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer
# Replace deprecated imports
from torch.amp import autocast, GradScaler
import time

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
parser.add_argument('--save_name', type=str, default='airfoil_Transolver')
parser.add_argument('--data_path', type=str, default='./data/airfoil/naca')
parser.add_argument('--use_wandb', type=int, default=1, help='Whether to use Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='PDE-Solving', help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
parser.add_argument('--use_amp', type=int, default=1, help='Whether to use Automatic Mixed Precision')
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


def main():
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=save_name,
        )
    
    # Initialize gradient scaler for AMP with correct API
    scaler = GradScaler('cuda') if args.use_amp else None
    
    INPUT_X = args.data_path + '/NACA_Cylinder_X.npy'
    INPUT_Y = args.data_path + '/NACA_Cylinder_Y.npy'
    OUTPUT_Sigma = args.data_path + '/NACA_Cylinder_Q.npy'

    ntrain = 1000
    ntest = 200

    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 4]
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    x_train = x_train.reshape(ntrain, -1, 2)
    x_test = x_test.reshape(ntest, -1, 2)
    y_train = y_train.reshape(ntrain, -1)
    y_test = y_test.reshape(ntest, -1)

    # Add normalizers similar to exp_pipe.py
    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, y_test),
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print("Dataloading is over.")

    model = get_model(args)(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=0,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  H=s1, W=s2).cuda()

    # compile the model, autotuning for performance
    model = torch.compile(model, mode="max-autotune")
    
    # Add wandb model watching
    if args.use_wandb and not eval:
        wandb.watch(model, log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"))
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        rel_err = 0.0
        showcase = 10
        id = 0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()

                # Use AMP for inference if enabled
                use_auto_cast = True if args.use_amp else False
                with autocast('cuda', enabled=use_auto_cast):
                    out = model(x, None).squeeze(-1)
                    out = y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl
                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   np.zeros([140, 35]),
                                   shading='auto',
                                   edgecolors='black', linewidths=0.1)
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "input_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   out[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1.2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   y[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1.2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   out[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy() - \
                                   y[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.2, 0.2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))

        # Log evaluation metrics to W&B
        if args.use_wandb:
            wandb.log({"test/rel_err": rel_err})
    else:
        for ep in range(args.epochs):

            model.train()
            train_loss = 0

            for pos, fx, y in train_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()  # x:B,N,2  fx:B,N,2  y:B,N
                optimizer.zero_grad()

                # Use AMP for training if enabled
                use_auto_cast = True if args.use_amp else False
                with autocast('cuda', enabled=use_auto_cast):
                    out = model(x, None).squeeze(-1)
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                    loss = myloss(out, y)

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

                train_loss += loss.item()
                scheduler.step()

            train_loss = train_loss / ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            model.eval()
            rel_err = 0.0
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()

                    # Use AMP for evaluation if enabled
                    use_auto_cast = True if args.use_amp else False
                    with autocast('cuda', enabled=use_auto_cast):
                        out = model(x, None).squeeze(-1)
                        out = y_normalizer.decode(out)

                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))

            # Log metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "test/rel_error": rel_err,
                    "epoch": ep,
                    "learning_rate": scheduler.get_last_lr()[0]
                })

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        final_checkpoint_path = os.path.join('./checkpoints', save_name + '.pt')
        torch.save(model.state_dict(), final_checkpoint_path)

        # Save final model to wandb and finish session
        if args.use_wandb:
            wandb.save(final_checkpoint_path)
            wandb.finish()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total time: {:.2f} seconds".format(end_time - start_time))
