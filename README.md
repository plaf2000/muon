# Muon Optimizer Research Project

Research project on the Muon optimizer with experiments tracking sharpness, ablations, Edge-of-Stability analysis and more

## Quick Setup

```bash
# 1. Create conda environment
bash setup.sh

# 2. Activate environment
conda activate muon

# 3. (macOS only) Set OpenMP fix (add to ~/.zshrc for permanent)
export KMP_DUPLICATE_LIB_OK=TRUE

# 4. Verify installation
python verify_setup.py


# 5. login in wandb
wandb login 
```


## Project Structure

```
muon/
├── src/
│   ├── models/          # MLP, TinyViT
│   ├── optimizers/      # Muon optimizer
│   ├── geometry/        # Hessian/curvature computation
│   ├── utils/           # Data loading, training, visualization
│   └── experiments/     # Training scripts
├── configs/             # YAML config files
├── results/             # Training results (organized by run name)
└── environment.yml      # Conda environment
```

## Results Organization

Each run creates its own directory:
```
results/basic_training/
  {run_name}/
    ├── {run_name}_results.pt          # Model + history
    ├── {run_name}_config.yaml         # Config used
    ├── {run_name}_lambda_max.csv      # Curvature data
    ├── {run_name}_subset_indices.txt  # Subset indices (if used)
    ├── checkpoint_epoch_*.pt          # Checkpoints
    └── visualizations/
        ├── {run_name}_training_history.png
        ├── {run_name}_lambda_max.png
        ├── {run_name}_predictions.png
        └── {run_name}_confusion_matrix.png
```

## Wandb Integration

- **Project**: All runs go to `"muon"` project
- **Run names**: `{prefix}_{model}_{dataset}_{optimizer}_{timestamp}`
  - Example: `basic_training_mlp_mnist_muon_20251205_162201`
  - Filter by prefix: `task1_sharpness_*`, `task2_*`, etc.
- **Logged**: Loss, accuracy, λ_max (when enabled), visualizations

## Key Features

- ✅ **Full-batch training** (on subset if specified)
- ✅ **Muon optimizer** with automatic parameter separation
- ✅ **Curvature tracking** (λ_max via power iteration) - optional
- ✅ **Reproducible subsets** (same samples across runs)
- ✅ **Automatic visualizations** (predictions, history, confusion matrix)
- ✅ **Config-driven** (all hyperparameters in YAML)

## Troubleshooting

### OpenMP Error on macOS
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```
Add to `~/.zshrc` to make permanent.

### Out of Memory (CIFAR-10)
Reduce `train_subset_size` in config (e.g., 2000, 1000, 500).

### Verify Setup
```bash
python verify_setup.py
```
## Running Experiments

All hyperparameters are specified via YAML config files:

```bash
# Train MLP on MNIST (basic training - no curvature tracking)
python -m src.experiments.basic_training --config configs/basic_training_mlp_mnist.yaml

# Train TinyViT on CIFAR-10 (basic training - no curvature tracking)
python -m src.experiments.basic_training --config configs/basic_training_vit_cifar10.yaml
```

## Configuration

### Key Config Options

```yaml
experiment:
  name: "basic_training_mlp_mnist"  # Used for run naming and filtering
  seed: 42
  device: "auto"  # Auto-detects CUDA/MPS/CPU

dataset:
  name: "mnist"  # or "cifar10"
  train_subset_size: 2000  # Use subset for CIFAR-10 (same subset across runs)

model:
  type: "mlp"  # or "tiny_vit"
  config:
    # Model-specific parameters

optimizer:
  type: "muon"  # or "sgd", "adamw"
  config:
    lr: 0.02
    momentum: 0.95
    ns_depth: 5
```

### Memory Management for CIFAR-10

For TinyViT on CIFAR-10, use a subset to avoid OOM errors:

```yaml
dataset:
  train_subset_size: 2000  # Adjust based on available memory
```

The subset is **deterministic** - same samples used across all runs (ensures comparability).

## Task 1: Sharpness Tracking

Compare λ_max evolution for Muon, SGD, and AdamW:

```bash
python -m src.experiments.task1_sharpness --config configs/task1_sharpness.yaml
```

This will:
1. Run training with Muon optimizer (tracks λ_max)
2. Run training with SGD optimizer (tracks λ_max)
3. Run training with AdamW optimizer (tracks λ_max)
4. All results logged to wandb project "muon" with prefix `task1_sharpness_*`

**Note**: Task 1 uses `basic_training.py` internally - it just runs it multiple times with different optimizers. The curvature tracking is enabled in the Task 1 config.

## Future Experiments

- **Task 2**: Ablation studies (Muon components)
- **Task 3**: Edge-of-Stability analysis

All use the same `basic_training.py` script with different configs.
