import torch
import trimesh
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from abupt_collator import AbuptCollator
from drivaerml_dataset import DrivAerMLDataset
from model import AnchoredBranchedUPT
from preprocessors import MomentNormalizationPreprocessor, PositionNormalizationPreprocessor
from utils import plot_pointcloud_single, plot_pointcloud_double, set_seed

set_seed(0)

vtp = pv.read("../data/run_11/boundary_11.vtp")

surface_position = torch.from_numpy(vtp.cell_centers().points)
print(f"{len(surface_position) / 1000 / 1000:.1f}M points in 3D space")
print(f"surface_position.shape: {surface_position.shape}")
torch.save(surface_position, "../data/run_11/surface_position_vtp.pt")
plot_pointcloud_single(
    surface_position,
    # increase this for higher resolution or larger plot
    num_points=10000,
    figsize=(6, 6)
)

surface_pressure = torch.from_numpy(vtp.get_array("pMeanTrim"))
print(f"{len(surface_position) / 1000 / 1000:.1f}M points with a scalar pressure value")
print(f"surface_pressure.shape: {surface_pressure.shape}")
torch.save(surface_pressure, "../data/run_11/surface_pressure.pt")
plot_pointcloud_single(
    surface_position,
    # adjust clamp values for different color scale (original range is roughly -2500 to 700)
    color=surface_pressure.clamp(-1500, 500),
    title="surface pressure",
    # increas this for more fidelity/larger plot
    num_points=20000,
    figsize=(6, 6),
)

wallshearstress = torch.from_numpy(vtp.get_array("wallShearStressMeanTrim"))
print(f"{len(surface_position) / 1000 / 1000:.1f}M points with a 3D wallshearstress vector")
print(f"wallshearstress.shape: {wallshearstress.shape}")
torch.save(wallshearstress, "../data/run_11/surface_wallshearstress.pt")
plot_pointcloud_single(
    surface_position,
    # adjust clamp values for different color scale (original range is roughly 0 to 30)
    color=wallshearstress.norm(dim=1).clamp(0, 5),
    title="wallshearstress",
    # increas this for more fidelity/larger plot
    num_points=20000,
    figsize=(6, 6),
)

num_surface_anchors = 16384
surface_perm = torch.randperm(len(surface_position))
surface_anchor_idxs = surface_perm[:num_surface_anchors]
surface_query_idxs = surface_perm[num_surface_anchors:]
print(f"num_surface_anchors: {len(surface_anchor_idxs)}")
print(f"num_surface_queries: {len(surface_query_idxs)}")

# select anchors
surface_anchor_position = surface_position[surface_anchor_idxs]
surface_anchor_pressure = surface_pressure[surface_anchor_idxs]
surface_anchor_wallshearstress = wallshearstress[surface_anchor_idxs]
# select queries
surface_query_position = surface_position[surface_query_idxs]
surface_query_pressure = surface_pressure[surface_query_idxs]
surface_query_wallshearstress = wallshearstress[surface_query_idxs]
# print shapes
print(f"surface_anchor_position.shape: {surface_anchor_position.shape}")
print(f"surface_anchor_pressure.shape: {surface_anchor_pressure.shape}")
print(f"surface_anchor_wallshearstress.shape: {surface_anchor_wallshearstress.shape}")
print(f"surface_query_position.shape: {surface_query_position.shape}")
print(f"surface_query_pressure.shape: {surface_query_pressure.shape}")
print(f"surface_query_wallshearstress.shape: {surface_query_wallshearstress.shape}")

# use CFD mesh for geometry representation
geometry_position = vtp.cell_centers().points
print(f"geometry_position.shape: {geometry_position.shape}")

num_geometry_points = 65536
geometry_perm = torch.randperm(len(geometry_position))
geometry_idxs = geometry_perm[:num_geometry_points]
print(f"geometry_idxs.shape: {geometry_idxs.shape}")

# select positions
geometry_position = torch.from_numpy(geometry_position[geometry_idxs])

# select supernodes
num_geometry_supernodes = 16384
supernode_idxs = torch.randperm(num_geometry_points)
print(f"supernode_idxs.shape: {supernode_idxs.shape}")

# the dataset loads a raw (i.e., unprocessed sample)
dataset = DrivAerMLDataset(root="../data", split="test")
raw_sample = dict(
    surface_position_vtp=dataset.getitem_surface_position_vtp(0),
    surface_pressure=dataset.getitem_surface_pressure(0),
    surface_wallshearstress=dataset.getitem_surface_wallshearstress(0),
    volume_position=dataset.getitem_volume_position(0),
    volume_totalpcoeff=dataset.getitem_volume_totalpcoeff(0),
    volume_velocity=dataset.getitem_volume_velocity(0),
    volume_vorticity=dataset.getitem_volume_vorticity(0),
)

for key, value in raw_sample.items():
  print(f"{key}: {value.shape}")

collator = AbuptCollator(
    num_geometry_points=65536,
    num_surface_anchor_points=16384,
    num_volume_anchor_points=16384,
    num_geometry_supernodes=16384,
    use_query_positions=True,
    dataset=dataset,
)
# convert a list of samples to a preprocessed batch
batch = collator([raw_sample])

for key, value in batch.items():
  print(f"{key}: {value.shape}")

abupt = AnchoredBranchedUPT().to("cuda").eval()

# Print number of parameters
num_params = sum(p.numel() for p in abupt.parameters())
print(f"Number of parameters: {num_params}")

# move batch to gpu
batch = {key: value.to("cuda") for key, value in batch.items()}

# extract target variables for anchor
target_surface_anchor_pressure = batch.pop("surface_anchor_pressure")
target_surface_anchor_wallshearstress = batch.pop("surface_anchor_wallshearstress")
target_volume_anchor_totalpcoeff = batch.pop("volume_anchor_totalpcoeff")
target_volume_anchor_velocity = batch.pop("volume_anchor_velocity")
target_volume_anchor_vorticity = batch.pop("volume_anchor_vorticity")
# extract target variables for queries
target_surface_query_pressure = batch.pop("surface_query_pressure")
target_surface_query_wallshearstress = batch.pop("surface_query_wallshearstress")
target_volume_query_totalpcoeff = batch.pop("volume_query_totalpcoeff")
target_volume_query_velocity = batch.pop("volume_query_velocity")
target_volume_query_vorticity = batch.pop("volume_query_vorticity")

# we dont need all 8M surface points for now, so we use a subset to make this section run fast
# later on, we will use all 8M points, dont worry :)
num_surface_queries = 16384
batch["surface_query_position"] = batch["surface_query_position"][:, :num_surface_queries]
target_surface_query_pressure = target_surface_query_pressure[:num_surface_queries]
target_surface_query_wallshearstress = target_surface_query_wallshearstress[:num_surface_queries]

for key, value in batch.items():
  print(f"{key}: {value.shape}")

# inference forward passes
with torch.no_grad():
  # Don't use autocast for now until we properly handle mixed precision
  # with torch.autocast(device_type="cuda", dtype=torch.float16):
  prediction = abupt(**batch)

for key, value in prediction.items():
  print(f"{key}: {value.shape}")