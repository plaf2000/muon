"""
Task 2: Ablate Muon's components (NS depth, RMS, orthogonalization, weight decay).
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, List
import argparse


def load_config(config_path: str) -> Dict:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # TODO: Implement config loading
    pass


def run_ablation_study(
    config: Dict
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run ablation study on Muon components.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with results for each ablation variant
    """
    # TODO: Implement ablation study
    pass


def main():
    """
    Main entry point for Task 2 experiment.
    """
    parser = argparse.ArgumentParser(description="Task 2: Muon Ablation Study")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2_ablation.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # TODO: Load config and run experiment
    pass


if __name__ == "__main__":
    main()
