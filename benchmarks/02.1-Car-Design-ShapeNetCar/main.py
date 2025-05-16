import train
import os
import torch
import torch.nn as nn
import argparse

from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
from models.components.erwinflash.erwin_flash import ErwinTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/shapenet_car/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/shapenet_car/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=float)
parser.add_argument('--nb_epochs', default=200, type=float)
parser.add_argument('--preprocessed', default=1, type=int)
args = parser.parse_args()
print(args)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

if args.cfd_model == 'ErwinFlashTransformer':
    # Positional Encoder similar to ShapenetCarModel
    class Positional_Encoder(nn.Module):
        def __init__(self, num_features, dimensionality=3, sigma=1.):
            super().__init__()
            self.linear = nn.Linear(dimensionality, num_features // 2, bias=False)
            nn.init.normal_(self.linear.weight, 0.0, 1.0 / sigma)
            
        def forward(self, x):
            proj = self.linear(x)
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            
    # Create a wrapper class to adapt ErwinTransformer to the expected interface
    class ErwinWrapper(torch.nn.Module):
        def __init__(self, hidden_dim=128, input_feature_dim=7, radius=0.2):
            super(ErwinWrapper, self).__init__()
            # Save the radius parameter for building the graph
            self.radius = radius
            
            # The Erwin model will directly use the input features
            self.model = ErwinTransformer(
                c_in=input_feature_dim,  # Use the original input features (7)
                c_hidden=[hidden_dim, hidden_dim],     # Hidden dimensions
                ball_sizes=[64, 32],     # Simplified architecture
                enc_num_heads=[8, 8],    
                enc_depths=[4, 4],       
                dec_num_heads=[8],       
                dec_depths=[4],          
                strides=[2],             
                rotate=45,               
                decode=True,             
                mlp_ratio=4,             
                dimensionality=3,        
                mp_steps=3,              
            )
            
            # Output prediction head
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4)  # Output: velocity(3) + pressure(1)
            )
            
        def forward(self, data):
            cfd_data, geom_data = data
            
            # Extract positions and features
            node_positions = cfd_data.pos
            node_features = cfd_data.x  # [N, 7] (pos(3) + sdf(1) + normal(3))
            
            # Create batch indices (all zeros for single batch)
            batch_idx = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
            
            # Use edge_index if available, otherwise it will be built with radius
            edge_index = cfd_data.edge_index if hasattr(cfd_data, 'edge_index') else None
            
            # Forward pass through ErwinTransformer
            # Pass the radius parameter to build the graph if edge_index is not provided
            output_features = self.model(
                node_features, 
                node_positions, 
                batch_idx, 
                edge_index=edge_index,
                radius=self.radius  # Provide the radius parameter
            )
            
            # Apply prediction head to get final output
            output = self.pred_head(output_features)
            
            return output
    
    # Create the wrapper model with the radius parameter from args
    model = ErwinWrapper(hidden_dim=128, input_feature_dim=7, radius=args.r).cuda()


path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
if not os.path.exists(path):
    os.makedirs(path)

model = train.main(device, train_ds, val_ds, model, hparams, path, val_iter=args.val_iter, reg=args.weight,
                   coef_norm=coef_norm)
