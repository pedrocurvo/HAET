import os
import argparse
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import *
from torch.utils.data import Dataset, DataLoader
from utils.testloss import TestLoss
from model_dict import get_model
from utils.normalizer import UnitTransformer
from torch.amp import autocast, GradScaler
import glob
import torch._dynamo

parser = argparse.ArgumentParser('Training Transformer for 3D Point Cloud Segmentation')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_Irregular_Mesh')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--voxel_size', type=float, default=0.1, help='Voxel size for discretization')
parser.add_argument('--grid_size', type=int, default=32, help='Grid size for 3D voxelization')
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
parser.add_argument('--test_on_unlabeled', type=int, default=0, help='Test on unlabeled test data')
parser.add_argument('--save_name', type=str, default='sem3d_segmentation')
parser.add_argument('--data_path', type=str, default='./data/training')
parser.add_argument('--test_path', type=str, default='./data/test')
parser.add_argument('--use_wandb', type=int, default=1, help='Whether to use Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='Semantic3D-Segmentation', help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
parser.add_argument('--use_amp', type=int, default=1, help='Whether to use Automatic Mixed Precision')
parser.add_argument('--world_size', type=int, default=-1, help='Number of processes for distributed training')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
parser.add_argument('--local-rank', type=int, default=-1, help='Local rank from torch.distributed.launch')
parser.add_argument('--dist_url', type=str, default='env://', help='URL used to set up distributed training')
parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
parser.add_argument('--use_batch_weights', type=int, default=1, help='Whether to use batch weights for loss calculation')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name

# Disable DDP optimizer to fix compatibility with torch.compile
torch._dynamo.config.optimize_ddp = False

# Handle local rank from environment variable (for torch.distributed.run)
if args.local_rank == -1 and 'RANK' in os.environ:
    args.local_rank = int(os.environ['RANK'])
# Handle alternative argument format (for torch.distributed.launch)
elif args.local_rank == -1 and getattr(args, 'local-rank', -1) != -1:
    args.local_rank = getattr(args, 'local-rank')


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


# Set up for distributed training
if args.local_rank != -1:
    rank = args.local_rank
    # Need to ensure world_size is correctly set and initialize process group properly
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = torch.cuda.device_count()
        
    # Initialize printing first so we have proper logging
    setup_for_distributed(rank == 0)
    
    # Set the device before initializing process group
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    # Initialize process group with explicit world_size and rank
    if not dist.is_initialized():
        print(f"Initializing process group: rank={rank}, world_size={world_size}")
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank
        )
    
    # Wait for all processes to reach this point before continuing
    print(f"Rank {rank}: Waiting at barrier")
    dist.barrier()
    print(f"Rank {rank}: Passed barrier")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = 0
    world_size = 1

class UnstructuredSemantic3DDataset(Dataset):
    def __init__(self, data_path, max_points=100000, chunk_size=100000, is_training=True, file_list=None):
        self.data_path = data_path
        self.max_points = max_points  # Maximum points to use per point cloud
        self.chunk_size = chunk_size  # Maximum points per chunk
        self.is_training = is_training
        self.cache_dir = os.path.join(data_path, "cached_data")
        self.num_classes = 8  # 8 classes in Semantic3D
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use provided file list or get all files from the directory
        if file_list is not None:
            self.file_list = file_list
        else:
            # Get list of all files
            self.file_list = []
            for file_path in glob.glob(os.path.join(data_path, "*.txt")):
                if not file_path.endswith('.labels'):
                    label_file = file_path.replace('.txt', '.labels')
                    if is_training and os.path.exists(label_file):
                        self.file_list.append(file_path)
                    elif not is_training:  # For test data, labels may not exist
                        self.file_list.append(file_path)
        
        print(f"Found {len(self.file_list)} {'training' if is_training else 'testing'} files in dataset")
        
        # Initialize class statistics
        self.file_class_counts = []
        self.total_class_counts = torch.zeros(self.num_classes)
        
        # Preprocess and cache files
        self._preprocess_files()
        
        # Calculate global class weights
        self.class_weights = self._calculate_class_weights()
        print(f"Global class weights: {self.class_weights}")
    
    def _calculate_class_weights(self):
        """Calculate class weights based on inverse square root of class frequencies"""
        if torch.sum(self.total_class_counts) == 0:
            # Fallback to equal weights if no statistics available
            return torch.ones(self.num_classes) / self.num_classes
        
        # Add small epsilon to avoid division by zero
        inv_freqs = 1.0 / torch.sqrt(self.total_class_counts + 1e-6)
        weights = inv_freqs / inv_freqs.sum()  # Normalize to sum to 1
        return weights
    
    def _preprocess_files(self):
        """Preprocess and cache point cloud files for faster loading"""
        self.cached_files = []
        
        for file_idx, file_path in enumerate(tqdm(self.file_list, desc="Preprocessing point clouds")):
            cache_file = os.path.join(self.cache_dir, os.path.basename(file_path).replace('.txt', '.npz'))
            self.cached_files.append(cache_file)
            
            # Initialize per-file class counts
            file_class_counts = torch.zeros(self.num_classes)
            
            # Skip preprocessing if cache file already exists, but still compute class statistics
            if os.path.exists(cache_file):
                # Load cached data to get class statistics
                try:
                    data = np.load(cache_file)
                    if 'labels' in data:
                        labels = data['labels']
                        # Count classes in this file (labels are already shifted to 0-7)
                        for c in range(self.num_classes):
                            file_class_counts[c] = np.sum(labels == c)
                    self.file_class_counts.append(file_class_counts)
                    self.total_class_counts += file_class_counts
                    continue
                except Exception as e:
                    print(f"Error loading cached file {cache_file}: {e}. Regenerating...")
            
            # Load point cloud data
            point_data = np.loadtxt(file_path)
            
            # Load labels if training
            labels = None
            if self.is_training:
                label_file = file_path.replace('.txt', '.labels')
                if os.path.exists(label_file):
                    labels = np.loadtxt(label_file).astype(np.int64)
                    # Verify point data and labels have the same length
                    if len(point_data) != len(labels):
                        print(f"Warning: Mismatch between point data ({len(point_data)}) and labels ({len(labels)}) in {file_path}")
                    
                    # Filter out points with label 0 (unlabeled) to reduce dataset size
                    if len(point_data) == len(labels):
                        valid_mask = labels != 0
                        point_data = point_data[valid_mask]
                        labels = labels[valid_mask]
                        print(f"Filtered out {np.sum(~valid_mask)} unlabeled points from {file_path}")
            
            # Extract coordinates and features
            coords = point_data[:, :3]  # x, y, z
            features = point_data[:, 3:]  # intensity, r, g, b
            
            # Normalize coordinates to [0, 1]
            min_coord = np.min(coords, axis=0)
            max_coord = np.max(coords, axis=0)
            coords = (coords - min_coord) / (max_coord - min_coord + 1e-8)
            
            # Normalize features
            # Intensity is already normalized, RGB needs to be divided by 255
            if features.shape[1] == 4:  # intensity, R, G, B
                features[:, 1:] = features[:, 1:] / 255.0  # Only normalize RGB values
            
            # Process labels
            if labels is not None:
                # Labels in Semantic3D: 
                # 0: unlabeled (should be ignored) - already filtered out above
                # 1-8: actual classes
                
                # Shift labels to 0-7 range for model output
                labels = labels - 1
                
                # Count classes in this file
                for c in range(self.num_classes):
                    file_class_counts[c] = np.sum(labels == c)
                
                # Save to cache file
                np.savez_compressed(cache_file, coords=coords, features=features, labels=labels)
            else:
                # Save without labels
                np.savez_compressed(cache_file, coords=coords, features=features)
            
            # Store class statistics
            self.file_class_counts.append(file_class_counts)
            self.total_class_counts += file_class_counts
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load from cached file using memory mapping for large files
        cache_file = self.cached_files[idx]
        data = np.load(cache_file, mmap_mode='r')
        
        # Get data from memory-mapped file
        coords = data['coords']
        features = data['features']
        
        # Get class distribution for this file
        file_class_counts = self.file_class_counts[idx]
        
        # Subsample if the point cloud is very large
        indices = None
        if len(coords) > self.max_points:
            indices = np.random.choice(len(coords), self.max_points, replace=False)
            # Need to copy data from mmap to modify it
            coords = coords[indices].copy()
            features = features[indices].copy()
            
            # Load labels if available
            if 'labels' in data:
                labels = data['labels'][indices].copy()
                
                # Update class counts for the subsampled data
                subsampled_class_counts = torch.zeros(self.num_classes)
                for c in range(self.num_classes):
                    subsampled_class_counts[c] = np.sum(labels == c)
                
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                
                # Convert to tensors
                coords_tensor = torch.tensor(coords, dtype=torch.float16)
                features_tensor = torch.tensor(features, dtype=torch.float16)
                
                return coords_tensor, features_tensor, labels_tensor, subsampled_class_counts
        else:
            # If not subsampling, convert directly to tensors
            coords_tensor = torch.tensor(coords, dtype=torch.float16)
            features_tensor = torch.tensor(features, dtype=torch.float16)
            
            # Load labels if available
            if 'labels' in data:
                labels = data['labels']
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                return coords_tensor, features_tensor, labels_tensor, file_class_counts
        
        # For test data or if labels don't exist
        return coords_tensor, features_tensor, torch.tensor([-1], dtype=torch.long), torch.zeros(self.num_classes)


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    # Initialize wandb if enabled (only on the main process)
    if args.use_wandb and is_main_process():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=save_name,
        )
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if args.use_amp else None
    
    # Get all training files
    training_files = []
    for file_path in glob.glob(os.path.join(args.data_path, "*.txt")):
        if not file_path.endswith('.labels'):
            label_file = file_path.replace('.txt', '.labels')
            if os.path.exists(label_file):
                training_files.append(file_path)
    
    # Split training files into train and validation sets
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(training_files)
    val_size = int(len(training_files) * args.val_split)
    train_files = training_files[val_size:]
    val_files = training_files[:val_size]
    
    if is_main_process():
        print(f"Total training files: {len(training_files)}")
        print(f"Training set: {len(train_files)} files")
        print(f"Validation set: {len(val_files)} files")
    
    # Create datasets
    train_dataset = UnstructuredSemantic3DDataset(args.data_path, max_points=6000000, chunk_size=1000000, is_training=True, file_list=train_files)
    val_dataset = UnstructuredSemantic3DDataset(args.data_path, max_points=6000000, chunk_size=1000000, is_training=True, file_list=val_files)
    
    # For final testing on unlabeled data (if requested)
    if args.test_on_unlabeled:
        test_dataset = UnstructuredSemantic3DDataset(args.test_path, max_points=5000000, chunk_size=100000, is_training=False)
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if args.local_rank != -1 else None
    
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False,
        drop_last=False
    ) if args.local_rank != -1 else None
    
    # Create dataloaders with samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,  # Reduced worker count to avoid resource contention
        pin_memory=True,
        prefetch_factor=2,
        sampler=train_sampler,
        drop_last=True,  # Important for DDP to have equal batch sizes
        persistent_workers=True if args.local_rank != -1 else False  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,  # Reduced worker count
        sampler=val_sampler,
        drop_last=False,
        persistent_workers=True if args.local_rank != -1 else False
    )
    
    if args.test_on_unlabeled:
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if args.local_rank != -1 else None
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4,
            sampler=test_sampler
        )
    
    if is_main_process():
        print("Dataloading is over.")

    # Initialize model (remove H, W, D parameters which are for structured grids)
    if is_main_process():
        print("Creating model...")
    
    # Ensure consistent initialization across all processes
    if args.local_rank != -1:
        # Set same seed for all processes
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Wait for all processes to reach this point
        dist.barrier()
    
    # Create the model
    model = get_model(args).Model(space_dim=3,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=4,  # intensity + RGB
                                  out_dim=8,  # 8 classes (excluding unlabeled)
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  chunk_size=2000000,  # Set chunk size for large point clouds
                                  unified_pos=args.unified_pos).to(device)
    
    # Compile model 
    model = torch.compile(model)
    
    # Count parameters (only on main process)
    if is_main_process():
        count_parameters(model)
        print("Model created successfully")
    
    # Wrap model with DDP for distributed training
    if args.local_rank != -1:
        try:
            # Wait for all processes to finish model initialization
            dist.barrier()
            
            if is_main_process():
                print("Wrapping model with DDP...")
                
            # Verify model is properly initialized on this rank
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Rank {rank}: Model has {param_count} parameters before DDP")
            
            # Then wrap with DDP
            model = DDP(model, 
                        device_ids=[args.local_rank], 
                        output_device=args.local_rank, 
                        find_unused_parameters=True,
                        broadcast_buffers=True)  # Add broadcast_buffers to ensure all buffers are synced
            
            # Synchronize after DDP creation
            dist.barrier()
            
            if is_main_process():
                print("DDP model created successfully")
                
        except Exception as e:
            print(f"Error in DDP initialization on rank {rank}: {e}")
            # Attempt to clean up
            if dist.is_initialized():
                dist.destroy_process_group()
            raise e
    
    if args.use_wandb and not eval and is_main_process():
        wandb.watch(model, log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Get global class weights from the dataset
    if is_main_process():
        print(f"Global class weights: {train_dataset.class_weights}")
    
    if is_main_process():
        print(args)
        print(model)
        count_parameters(model)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))

    if eval:
        # Load checkpoint
        if args.local_rank != -1:
            # Load model for DDP
            if is_main_process():
                model_state_dict = torch.load("./checkpoints/" + save_name + ".pt")
                torch.save(model_state_dict, os.path.join('./checkpoints', save_name + '_resave' + '.pt'))
            dist.barrier()  # Make sure the master process saves first
            # Load on all processes
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
            model.module.load_state_dict(torch.load("./checkpoints/" + save_name + "_resave.pt", map_location=map_location))
        else:
            model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
            torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '_resave' + '.pt'))
        
        model.eval()
        if is_main_process() and not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        if is_main_process():
            print("Evaluating on validation set...")
        
        correct = 0
        total = 0
        class_correct = [0] * 8
        class_total = [0] * 8
        confusion_matrix = np.zeros((8, 8), dtype=np.int64)

        with torch.no_grad():
            for coords, features, labels, class_counts in val_loader:
                coords, features, labels, class_counts = coords.to(device), features.to(device), labels.to(device), class_counts.to(device)
                
                # Use AMP for inference if enabled
                use_auto_cast = True if args.use_amp else False
                with autocast(device_type='cuda', enabled=use_auto_cast):
                    outputs = model(coords, features)  # Shape: [batch_size, num_points, 8]
                    # Reshape outputs and labels for cross entropy
                    B, N, C = outputs.shape
                    outputs = outputs.view(B*N, C)  # Reshape to [batch_size*num_points, 8]
                    
                    labels_flat = labels.view(-1)   # Reshape to [batch_size*num_points]
                    
                    # Compute batch-specific weights
                    valid_labels = labels_flat[labels_flat != -1]  # Exclude ignored labels
                    if len(valid_labels) > 0:
                        batch_weights = class_counts[valid_labels] / class_counts[valid_labels].sum()
                        
                        # Compute per-element loss
                        element_loss = criterion(outputs, labels_flat)
                        
                        # Apply weights based on class
                        weighted_loss = element_loss.clone()
                        for c in range(C):
                            class_mask = (labels_flat == c)
                            weighted_loss[class_mask] = element_loss[class_mask] * batch_weights[c]
                        
                        # Average only over valid elements
                        loss = weighted_loss[labels_flat != -1].mean()
                    else:
                        # Fallback to standard loss with global weights if no valid labels
                        criterion_fallback = torch.nn.CrossEntropyLoss(weight=global_weights, ignore_index=-1)
                        loss = criterion_fallback(outputs, labels_flat)
                
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.view(B, N)  # Reshape back to [batch_size, num_points]
                
                # Calculate accuracy (only on non-ignored indices)
                mask = (labels != -1)
                predicted_masked = predicted[mask]
                labels_masked = labels[mask]
                
                batch_total = labels_masked.size(0)
                batch_correct = (predicted_masked == labels_masked).sum().item()
                
                # Gather results from all processes
                if args.local_rank != -1:
                    batch_total_tensor = torch.tensor([batch_total]).to(device)
                    batch_correct_tensor = torch.tensor([batch_correct]).to(device)
                    
                    dist.all_reduce(batch_total_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_correct_tensor, op=dist.ReduceOp.SUM)
                    
                    batch_total = batch_total_tensor.item()
                    batch_correct = batch_correct_tensor.item()
                
                total += batch_total
                correct += batch_correct
                
                # Per-class accuracy
                for i in range(8):
                    class_mask = (labels_masked == i)
                    batch_class_total = class_mask.sum().item()
                    batch_class_correct = ((predicted_masked == i) & class_mask).sum().item()
                    
                    # Gather results from all processes
                    if args.local_rank != -1:
                        batch_class_total_tensor = torch.tensor([batch_class_total]).to(device)
                        batch_class_correct_tensor = torch.tensor([batch_class_correct]).to(device)
                        
                        dist.all_reduce(batch_class_total_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(batch_class_correct_tensor, op=dist.ReduceOp.SUM)
                        
                        batch_class_total = batch_class_total_tensor.item()
                        batch_class_correct = batch_class_correct_tensor.item()
                    
                    class_total[i] += batch_class_total
                    class_correct[i] += batch_class_correct
                    
                    # Update confusion matrix
                    for j in range(8):
                        batch_confusion = ((predicted_masked == j) & class_mask).sum().item()
                        
                        # Gather results from all processes
                        if args.local_rank != -1:
                            batch_confusion_tensor = torch.tensor([batch_confusion]).to(device)
                            dist.all_reduce(batch_confusion_tensor, op=dist.ReduceOp.SUM)
                            batch_confusion = batch_confusion_tensor.item()
                        
                        confusion_matrix[i, j] += batch_confusion

        # Calculate metrics
        accuracy = 100 * correct / total
        class_accuracies = [100 * class_correct[i] / max(1, class_total[i]) for i in range(8)]
        mean_class_accuracy = sum(class_accuracies) / 8
        
        if is_main_process():
            print(f"Validation Overall Accuracy: {accuracy:.2f}%")
            print(f"Validation Mean Class Accuracy: {mean_class_accuracy:.2f}%")
            print("Per-class accuracies:")
            for i in range(8):
                print(f"Class {i+1}: {class_accuracies[i]:.2f}%")
            
            # Log metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "val/overall_accuracy": accuracy,
                    "val/mean_class_accuracy": mean_class_accuracy,
                    "val/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=list(range(8)),
                        preds=list(range(8)),
                        class_names=["man-made terrain", "natural terrain", "high vegetation", 
                                    "low vegetation", "buildings", "hard scape", "scanning artefacts", "cars"]
                    )
                })
                
                # Log per-class accuracies
                for i in range(8):
                    wandb.log({f"val/class_{i+1}_accuracy": class_accuracies[i]})
        
        # If requested, run prediction on unlabeled test data
        if args.test_on_unlabeled and os.path.exists(args.test_path):
            if is_main_process():
                print("\nRunning prediction on unlabeled test data...")
            
            test_predictions = []
            test_file_paths = []
            
            with torch.no_grad():
                for batch_idx, (coords, features, labels, class_counts) in enumerate(test_loader):
                    coords, features, labels, class_counts = coords.to(device), features.to(device), labels.to(device), class_counts.to(device)
                    
                    # Use AMP for inference if enabled
                    use_auto_cast = True if args.use_amp else False
                    with autocast(device_type='cuda', enabled=use_auto_cast):
                        outputs = model(coords, features)  # Shape: [batch_size, num_points, 8]
                        # Reshape outputs and labels for cross entropy
                        B, N, C = outputs.shape
                        outputs = outputs.view(B*N, C)  # Reshape to [batch_size*num_points, 8]
                        _, predicted = torch.max(outputs, 1)
                        predicted = predicted.view(B, N)  # Reshape back to [batch_size, num_points]
                    
                    # Store predictions (add 1 to convert back to 1-8 range)
                    for i in range(coords.size(0)):
                        test_predictions.append(predicted[i].cpu().numpy() + 1)
                        if batch_idx * args.batch_size + i < len(test_dataset.file_list):
                            test_file_paths.append(test_dataset.file_list[batch_idx * args.batch_size + i])
            
            # Save predictions (main process only)
            if is_main_process():
                for file_path, pred in zip(test_file_paths, test_predictions):
                    output_file = os.path.join('./results/' + save_name + '/', 
                                               os.path.basename(file_path).replace('.txt', '_pred.txt'))
                    np.savetxt(output_file, pred, fmt='%d')
                    print(f"Saved predictions to {output_file}")
    else:
        for ep in range(args.epochs):
            # Set epoch for samplers
            if args.local_rank != -1:
                train_sampler.set_epoch(ep)
                
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            step = 0

            for coords, features, labels, class_counts in train_loader:
                coords, features, labels, class_counts = coords.to(device), features.to(device), labels.to(device), class_counts.to(device)
                
                # Optionally use batch-specific weights
                if args.use_batch_weights:
                    # Sum class counts across the batch
                    batch_class_counts = class_counts.sum(dim=0)
                    # Compute batch-specific weights
                    batch_inv_freqs = 1.0 / torch.sqrt(batch_class_counts + 1e-6)
                    batch_weights = batch_inv_freqs / batch_inv_freqs.sum()
                    # Clamp weights for stability
                    batch_weights = torch.clamp(batch_weights, 0.1, 10.0)
                    # Create a new criterion with batch-specific weights
                    batch_criterion = torch.nn.CrossEntropyLoss(weight=batch_weights, ignore_index=-1)
                    criterion_to_use = batch_criterion
                else:
                    criterion_to_use = criterion
                
                optimizer.zero_grad()
                
                # Use AMP for forward pass
                use_auto_cast = True if args.use_amp else False
                with autocast(device_type='cuda', enabled=use_auto_cast):
                    outputs = model(coords, features)  # Shape: [batch_size, num_points, 8]
                    # Reshape outputs and labels for cross entropy
                    B, N, C = outputs.shape
                    outputs = outputs.view(B*N, C)  # Reshape to [batch_size*num_points, 8]
                    
                    labels_flat = labels.view(-1)   # Reshape to [batch_size*num_points]
                    
                    # Compute per-element loss
                    loss = criterion_to_use(outputs, labels_flat)
                
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
                
                # Calculate accuracy (only on non-ignored indices)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.view(B, N)  # Reshape back to [batch_size, num_points]
                mask = (labels != -1)
                predicted_masked = predicted[mask]
                labels_masked = labels[mask]
                
                batch_total = labels_masked.size(0)
                batch_correct = (predicted_masked == labels_masked).sum().item()
                
                # For distributed training, synchronize metrics
                if args.local_rank != -1:
                    batch_loss = loss.item()
                    batch_loss_tensor = torch.tensor([batch_loss]).to(device)
                    batch_total_tensor = torch.tensor([batch_total]).to(device)
                    batch_correct_tensor = torch.tensor([batch_correct]).to(device)
                    
                    dist.all_reduce(batch_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_total_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_correct_tensor, op=dist.ReduceOp.SUM)
                    
                    batch_loss = batch_loss_tensor.item() / dist.get_world_size()
                    batch_total = batch_total_tensor.item()
                    batch_correct = batch_correct_tensor.item()
                else:
                    batch_loss = loss.item()
                
                total += batch_total
                correct += batch_correct
                
                # Log step-wise metrics
                step += 1
                step_acc = 100 * batch_correct / (batch_total + 1e-8)
                
                if args.use_wandb and is_main_process():
                    wandb.log({
                        "train/step_loss": batch_loss,
                        "train/step_acc": step_acc,
                        "train/step": ep * len(train_loader) + step,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                    
                if step % 10 == 0 and is_main_process():
                    print(f"Epoch {ep}, Step {step}/{len(train_loader)}, Train loss: {batch_loss:.5f}, Train acc: {step_acc:.2f}%")

            # Compute epoch metrics
            if args.local_rank != -1:
                train_loss_tensor = torch.tensor([train_loss]).to(device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_loss = train_loss_tensor.item() / (dist.get_world_size() * len(train_loader))
            else:
                train_loss = train_loss / len(train_loader)
                
            train_acc = 100 * correct / (total + 1e-8)
            
            if is_main_process():
                print(f"Epoch {ep} Train loss: {train_loss:.5f}, Train acc: {train_acc:.2f}%")

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            class_correct = [0] * 8
            class_total = [0] * 8
            val_step = 0
            
            with torch.no_grad():
                for coords, features, labels, class_counts in val_loader:
                    coords, features, labels, class_counts = coords.to(device), features.to(device), labels.to(device), class_counts.to(device)
                    
                    # Use AMP for evaluation
                    use_auto_cast = True if args.use_amp else False
                    with autocast(device_type='cuda', enabled=use_auto_cast):
                        outputs = model(coords, features)  # Shape: [batch_size, num_points, 8]
                        # Reshape outputs and labels for cross entropy
                        B, N, C = outputs.shape
                        outputs = outputs.view(B*N, C)  # Reshape to [batch_size*num_points, 8]
                        
                        labels_flat = labels.view(-1)   # Reshape to [batch_size*num_points]
                        
                        # Compute per-element loss
                        loss = criterion_to_use(outputs, labels_flat)
                    
                    batch_loss = loss.item()
                    val_loss += batch_loss
                    
                    # Calculate accuracy (only on non-ignored indices)
                    mask = (labels != -1)
                    predicted_masked = predicted[mask]
                    labels_masked = labels[mask]
                    
                    batch_total = labels_masked.size(0)
                    batch_correct = (predicted_masked == labels_masked).sum().item()
                    
                    # Synchronize validation metrics for distributed training
                    if args.local_rank != -1:
                        batch_loss_tensor = torch.tensor([batch_loss]).to(device)
                        batch_total_tensor = torch.tensor([batch_total]).to(device)
                        batch_correct_tensor = torch.tensor([batch_correct]).to(device)
                        
                        dist.all_reduce(batch_loss_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(batch_total_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(batch_correct_tensor, op=dist.ReduceOp.SUM)
                        
                        batch_loss = batch_loss_tensor.item() / dist.get_world_size()
                        batch_total = batch_total_tensor.item()
                        batch_correct = batch_correct_tensor.item()
                    
                    total += batch_total
                    correct += batch_correct
                    
                    # Per-class accuracy
                    for i in range(8):
                        class_mask = (labels_masked == i)
                        batch_class_total = class_mask.sum().item()
                        batch_class_correct = ((predicted_masked == i) & class_mask).sum().item()
                        
                        # Synchronize class metrics for distributed training
                        if args.local_rank != -1:
                            batch_class_total_tensor = torch.tensor([batch_class_total]).to(device)
                            batch_class_correct_tensor = torch.tensor([batch_class_correct]).to(device)
                            
                            dist.all_reduce(batch_class_total_tensor, op=dist.ReduceOp.SUM)
                            dist.all_reduce(batch_class_correct_tensor, op=dist.ReduceOp.SUM)
                            
                            batch_class_total = batch_class_total_tensor.item()
                            batch_class_correct = batch_class_correct_tensor.item()
                        
                        class_total[i] += batch_class_total
                        class_correct[i] += batch_class_correct
                        
                    # Log step-wise validation metrics
                    val_step += 1
                    step_val_acc = 100 * batch_correct / (batch_total + 1e-8)
                    
                    # Calculate per-class accuracies for this batch
                    step_class_accs = []
                    for i in range(8):
                        if batch_class_total > 0:
                            class_acc = 100 * batch_class_correct / batch_class_total
                            step_class_accs.append(class_acc)
                    
                    step_mean_class_acc = sum(step_class_accs) / max(1, len(step_class_accs))
                    
                    if args.use_wandb and is_main_process():
                        wandb.log({
                            "val/step_loss": batch_loss,
                            "val/step_acc": step_val_acc,
                            "val/step_mean_class_acc": step_mean_class_acc,
                            "val/step": ep * len(val_loader) + val_step
                        })
                        
                    if val_step % 10 == 0 and is_main_process():
                        print(f"Epoch {ep}, Val Step {val_step}/{len(val_loader)}, Val loss: {batch_loss:.5f}, Val acc: {step_val_acc:.2f}%")

            # Compute epoch validation metrics
            if args.local_rank != -1:
                val_loss_tensor = torch.tensor([val_loss]).to(device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                val_loss = val_loss_tensor.item() / (dist.get_world_size() * len(val_loader))
            else:
                val_loss = val_loss / len(val_loader)
                
            val_acc = 100 * correct / (total + 1e-8)
            class_accuracies = [100 * class_correct[i] / max(1, class_total[i]) for i in range(8)]
            mean_class_accuracy = sum(class_accuracies) / 8
            
            if is_main_process():
                print(f"Validation loss: {val_loss:.5f}, Accuracy: {val_acc:.2f}%, mCA: {mean_class_accuracy:.2f}%")
                
                # Log metrics to wandb
                if args.use_wandb:
                    wandb.log({
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "val/mean_class_accuracy": mean_class_accuracy,
                        "epoch": ep,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
                    
                    # Log per-class accuracies
                    for i in range(8):
                        wandb.log({f"val/class_{i+1}_accuracy": class_accuracies[i]})

            if ep % 100 == 0 and is_main_process():
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                checkpoint_path = os.path.join('./checkpoints', save_name + '.pt')
                # When using DDP, save the model without the DDP wrapper
                if args.local_rank != -1:
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                
                # Log checkpoint to wandb
                if args.use_wandb:
                    wandb.save(checkpoint_path)

        if is_main_process():
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            print('save model')
            final_checkpoint_path = os.path.join('./checkpoints', save_name + '.pt')
            # When using DDP, save the model without the DDP wrapper
            if args.local_rank != -1:
                torch.save(model.module.state_dict(), final_checkpoint_path)
            else:
                torch.save(model.state_dict(), final_checkpoint_path)
            
            # Save final model to wandb
            if args.use_wandb:
                wandb.save(final_checkpoint_path)
                wandb.finish()


if __name__ == "__main__":
    try:
        if args.local_rank == -1 and args.world_size > 1:
            # Use torch.multiprocessing to launch distributed processes
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            print(f"Launching {args.world_size} processes via mp.spawn")
            mp.spawn(
                main,
                args=(),
                nprocs=args.world_size,
            )
        else:
            main()
    except Exception as e:
        print(f"Error in main process: {e}")
        # Clean up process group if initialized
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
