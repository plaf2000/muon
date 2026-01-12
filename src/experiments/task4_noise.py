"""
Task 4: Track loss for Muon, Muon Noise, SGD, AdamW
during full-batch training.

This script runs the same training with different optimizers and compares
the loss. It uses basic_training.py internally.
"""

import yaml
import argparse
from pathlib import Path
import subprocess
import sys
from typing import Dict


def load_config(config_path: str) -> Dict:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_training_with_optimizer(base_config: Dict, optimizer_type: str, optimizer_config: Dict):
    """
    Run training with a specific optimizer by calling basic_training.py.
    
    Args:
        base_config: Base configuration
        optimizer_type: Optimizer type ("muon", "sgd", "adamw")
        optimizer_config: Optimizer-specific configuration
        
    Returns:
        True if successful, False otherwise
    """
    # Create a temporary config for this optimizer
    config = base_config.copy()
    config["optimizer"] = {
        "type": optimizer_type,
        "config": optimizer_config
    }
    
    # Update experiment name to include optimizer
    original_name = config["experiment"]["name"]
    config["experiment"]["name"] = f"{original_name}_{optimizer_type}"
    
    # Save temporary config
    temp_config_path = Path(f"configs/temp_task4_{optimizer_type}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Run basic_training.py with this config
    print(f"\n{'='*60}")
    print(f"Running training with {optimizer_type.upper()} optimizer")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.experiments.basic_training", "--config", str(temp_config_path)],
            check=True
        )
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running training with {optimizer_type}: {e}")
        success = False
    finally:
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    return success


def main():
    """
    Main entry point for Task 4 experiment.
    Runs training with Muon, SGD, and AdamW, tracking λ_max for each.
    """
    parser = argparse.ArgumentParser(description="Task 4: Noise Ablation Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task4_noise.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get optimizers to test from config
    optimizers_config = config.get("optimizers", {})
    
    # Default optimizers if not specified
    if not optimizers_config:
        optimizers_config = {
            "muon": {
                "lr": 0.02,
                "momentum": 0.95,
                "ns_depth": 5,
                "use_rms": False,
                "use_orthogonalization": True,
                "weight_decay": 0.0,
                "adamw_lr": 0.001
            },
            "sgd": {
                "lr": 0.01,
                "momentum": 0.0,
                "weight_decay": 0.0
            },
            "adamw": {
                "lr": 0.001,
                "weight_decay": 0.01
            }
        }
    
    print("="*60)
    print("Task 4: Noise Ablation Experiment")
    print("="*60)
    print(f"Model: {config['model']['type']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Optimizers to test: {', '.join(optimizers_config.keys())}")
    print("="*60)
    
    # Run training for each optimizer
    results = {}
    for opt_name, opt_config in optimizers_config.items():
        success = run_training_with_optimizer(config, opt_name, opt_config)
        results[opt_name] = "success" if success else "failed"
    
    # Summary
    print("\n" + "="*60)
    print("Task 1 Summary")
    print("="*60)
    for opt_name, status in results.items():
        print(f"  {opt_name.upper()}: {status}")
    print("\nAll runs logged to wandb project 'muon'")
    print("Filter by: task4_noise_* to see all Task 4 runs")
    print("="*60)


if __name__ == "__main__":
    main()
