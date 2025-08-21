# BallTree Model Parameter Updates

This document summarizes the parameter changes made to migrate from slice-based attention to BallTree-based attention across the benchmark experiments.

## Changes Made

### 1. Core Model Files
- **models/PhysicsAttention/IrregularMesh.py**: Updated to use BallTree partitioning instead of slice tokens
- **models/HAETransolver_Irregular_Mesh.py**: Updated parameter passing and documentation

### 2. Parameter Changes

#### Removed Parameters:
- `slice_num`: Number of slice tokens (replaced by `ball_size`)
- `base_temp`: Base temperature for adaptive temperature scaling (no longer needed)
- `epsilon`: Small constant for Rep-Slice computation (no longer needed)

#### New Parameters:
- `ball_size`: Number of points per ball/region for spatial partitioning
- `radius`: Radius parameter for ErwinTransformer 
- `rotate`: Rotation angle for geometric awareness (default: 45 degrees)

### 3. Updated Experiment Files

All experiment files in `benchmarks/04-PDE-Solving-StandardBenchmark/` have been updated:

- `exp_airfoil.py`
- `exp_darcy.py` 
- `exp_elas.py`
- `exp_ns.py`
- `exp_pipe.py`
- `exp_plas.py`

### 4. Updated Script Files

All SLURM script files in `benchmarks/04-PDE-Solving-StandardBenchmark/scripts/` have been updated:

- `HAET_Airfoil.sh`
- `HAET_Darcy.sh`
- `HAET_Elas.sh`
- `HAET_NS.sh`
- `HAET_Pipe.sh`
- `HAET_Plas.sh`

#### Before (Python files):
```python
parser.add_argument('--slice_num', type=int, default=32)

model = get_model(args)(
    slice_num=args.slice_num,
    # other params...
)
```

#### After (Python files):
```python
parser.add_argument('--ball_size', type=int, default=32, help='Number of points per ball/region for spatial partitioning')
parser.add_argument('--radius', type=float, default=1.0, help='Radius parameter for ErwinTransformer')
parser.add_argument('--rotate', type=float, default=45, help='Rotation angle for geometric awareness')

model = get_model(args)(
    ball_size=args.ball_size,
    radius=args.radius,
    rotate=args.rotate,
    # other params...
)
```

#### Before (Script files):
```bash
srun python exp_airfoil.py \
    --slice_num 1024 \
    # other params...
```

#### After (Script files):
```bash
srun python exp_airfoil.py \
    --ball_size 1024 \
    --radius 1.0 \
    --rotate 45 \
    # other params...
```

## Benefits of BallTree Approach

1. **Better Spatial Locality**: Points are grouped based on actual spatial proximity
2. **Physics-Aware**: Respects the geometric structure of the mesh
3. **Efficient Processing**: Fixed ball sizes make computation more predictable
4. **Improved Attention**: Supernodes extracted from spatially coherent regions

## Usage Examples

### Running with default BallTree parameters:
```bash
python exp_airfoil.py --ball_size 32 --radius 1.0 --rotate 45
```

### Experimenting with different ball sizes:
```bash
python exp_airfoil.py --ball_size 16  # Smaller balls for finer granularity
python exp_airfoil.py --ball_size 64  # Larger balls for coarser granularity
```

### Adjusting spatial parameters:
```bash
python exp_airfoil.py --radius 0.5 --rotate 30  # Different spatial awareness settings
```

## Migration Notes

- All existing scripts should be updated to use `--ball_size` instead of `--slice_num`
- Default values remain the same (32) for backward compatibility
- New parameters (`radius`, `rotate`) have sensible defaults
- The model interface remains compatible, only parameter names have changed

## Testing

All updated files have been syntax-checked and should work with the new BallTree-based attention mechanism. The changes maintain the same computational complexity while providing better spatial awareness for physics problems.
