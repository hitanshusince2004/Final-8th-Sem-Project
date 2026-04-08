"""
Glacier Melting Detection — Training Utilities
==============================================
Shared helper functions for the training pipeline.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path to import project modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.ml_config import N_INPUT_CH, PATCH_SIZE, N_CLASSES
from utils.dataset import get_dataloaders, SyntheticGlacierDataset

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    """Get the best available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  Running on CPU (GPU not available)")
    return device

def get_image_loaders(patch_dir, batch_size, use_synthetic=False,
                      task="binary", n_synthetic=500, num_workers=0):
    """
    Return (train_loader, val_loader, test_loader).
    Uses synthetic data if real patches are not found.
    """
    if use_synthetic or not os.path.isdir(patch_dir):
        print(f"  Using synthetic data (n={n_synthetic})")
        train_ds = SyntheticGlacierDataset(int(n_synthetic*0.7), PATCH_SIZE, N_INPUT_CH, N_CLASSES, task)
        val_ds   = SyntheticGlacierDataset(int(n_synthetic*0.15), PATCH_SIZE, N_INPUT_CH, N_CLASSES, task)
        test_ds  = SyntheticGlacierDataset(int(n_synthetic*0.15), PATCH_SIZE, N_INPUT_CH, N_CLASSES, task)
        kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
        return (DataLoader(train_ds, shuffle=True, **kw),
                DataLoader(val_ds,   shuffle=False, **kw),
                DataLoader(test_ds,  shuffle=False, **kw))
    else:
        return get_dataloaders(patch_dir, task=task, batch_size=batch_size, num_workers=num_workers)
