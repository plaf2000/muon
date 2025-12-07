#!/usr/bin/env python
"""
Verification script to check that the Muon environment is set up correctly.
"""

import sys
import os
import torch

def verify_setup():
    """Verify that the environment is set up correctly."""
    print("=" * 60)
    print("Muon Environment Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check PyTorch
    print("PyTorch:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    # Check MPS (Mac GPU)
    has_mps = hasattr(torch.backends, "mps")
    if has_mps:
        mps_available = torch.backends.mps.is_available()
        print(f"  MPS available: {mps_available}")
        if mps_available:
            print("  ✓ Mac GPU (Metal) support is enabled")
    else:
        print("  MPS available: False (not supported on this system)")
    
    print()
    
    # Check other dependencies
    print("Other dependencies:")
    try:
        import numpy
        print(f"  ✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("  ✗ NumPy: NOT INSTALLED")
    
    try:
        import yaml
        print(f"  ✓ PyYAML: {yaml.__version__}")
    except ImportError:
        print("  ✗ PyYAML: NOT INSTALLED")
    
    try:
        import wandb
        print(f"  ✓ Wandb: {wandb.__version__}")
    except ImportError:
        print("  ✗ Wandb: NOT INSTALLED")
    
    try:
        from curvlinops import HessianLinearOperator
        print("  ✓ Curvlinops: INSTALLED (optional)")
    except ImportError:
        print("  ⚠ Curvlinops: NOT INSTALLED (optional, will use PyTorch fallback)")
    
    print()
    
    # Test device selection
    print("Device Selection Test:")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Selected: CUDA ({torch.cuda.get_device_name(0)})")
    elif has_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Selected: MPS (Mac GPU)")
    else:
        device = torch.device("cpu")
        print(f"  Selected: CPU")
    
    # Test tensor creation on selected device
    try:
        x = torch.randn(5, 5, device=device)
        print(f"  ✓ Successfully created tensor on {device}")
    except Exception as e:
        print(f"  ✗ Failed to create tensor on {device}: {e}")
        return False
    
    print()
    print("=" * 60)
    print("Verification complete!")
    print("=" * 60)
    
    # Warnings
    if sys.platform == "darwin":  # macOS
        kmp_set = "KMP_DUPLICATE_LIB_OK" in os.environ
        if not kmp_set:
            print()
            print("⚠ WARNING: KMP_DUPLICATE_LIB_OK not set")
            print("  Add this to your ~/.zshrc or ~/.bash_profile:")
            print("    export KMP_DUPLICATE_LIB_OK=TRUE")
            print("  Or run in current shell:")
            print("    export KMP_DUPLICATE_LIB_OK=TRUE")
    
    return True

if __name__ == "__main__":
    import os
    verify_setup()
