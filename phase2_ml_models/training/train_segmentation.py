"""
Glacier Melting Detection — Advanced Segmentation Training Script
==================================================================
Orchestrates training of the two advanced DL models:
  1. U-Net
  2. DeepLabv3+

This script is used to finish the training pipeline after ML and basic CNN models are done.
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.ml_config import PATCH_DIR, SEED
from training.train_utils import set_seed, get_device
from training.train_unet import run_unet
from training.train_deeplab import run_deeplabv3plus

def main():
    parser = argparse.ArgumentParser(description="Train U-Net and DeepLabv3+ models.")
    parser.add_argument("--patch_dir", default=PATCH_DIR, help="Path to image patch directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    # Setup
    set_seed(SEED)
    device = get_device()
    
    print("\n" + "═"*60)
    print("  PHASE 3: Advanced Segmentation Models (U-Net & DeepLabv3+)")
    print("═"*60)

    # 1. Train U-Net
    print("\n--- Starting U-Net Training ---")
    unet_results = run_unet(
        patch_dir     = args.patch_dir,
        use_synthetic = args.synthetic or not os.path.isdir(args.patch_dir),
    )

    # 2. Train DeepLabv3+
    print("\n--- Starting DeepLabv3+ Training ---")
    deeplab_results = run_deeplabv3plus(
        patch_dir     = args.patch_dir,
        use_synthetic = args.synthetic or not os.path.isdir(args.patch_dir),
    )

    print("\n" + "═"*60)
    print("  Advanced Segmentation Training Complete")
    print("═"*60)

if __name__ == "__main__":
    main()
