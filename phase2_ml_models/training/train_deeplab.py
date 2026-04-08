"""
Glacier Melting Detection — DeepLabv3+ Model Training Script
============================================================
Trains the DeepLabv3+ segmentation model.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.ml_config import DEEPLAB_CONFIG, DEEPLAB_TRAIN, PATCH_DIR
from models.deeplab_model import build_deeplabv3plus
from training.trainer import GlacierSegTrainer
from training.train_utils import get_image_loaders

def run_deeplabv3plus(patch_dir=PATCH_DIR, use_synthetic=False):
    """Train DeepLabv3+ — two-stage: freeze backbone first, then fine-tune."""
    print("\n" + "═"*60)
    print("  PHASE 3c: DeepLabv3+ — Advanced Segmentation")
    print("═"*60)

    bs = DEEPLAB_TRAIN["batch_size"]
    train_loader, val_loader, test_loader = get_image_loaders(
        patch_dir, bs, use_synthetic, task="multiclass"
    )

    model = build_deeplabv3plus(DEEPLAB_CONFIG)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  DeepLabv3+ parameters: {params:,}")

    # Stage 1: Train head only (freeze backbone) — faster convergence
    print("\n  Stage 1: Training head with frozen backbone…")
    model.freeze_backbone()

    stage1_config = {**DEEPLAB_TRAIN, "epochs": 1, "lr": 5e-4, 
                     "lr_scheduler": "cosine", "early_stop": 10}

    trainer1 = GlacierSegTrainer(
        model, train_loader, val_loader,
        config     = stage1_config,
        model_name = "DeepLabV3Plus_stage1",
    )
    trainer1.train()

    # Stage 2: Fine-tune entire model with lower LR
    print("\n  Stage 2: Fine-tuning entire model…")
    model.unfreeze_backbone()

    stage2_config = {**DEEPLAB_TRAIN, "epochs": DEEPLAB_TRAIN["epochs"],
                     "lr": DEEPLAB_TRAIN["lr"], "lr_scheduler": "poly"}

    trainer2 = GlacierSegTrainer(
        model, train_loader, val_loader,
        config     = stage2_config,
        model_name = "DeepLabV3Plus",
    )
    trainer2.train()
    results = trainer2.evaluate_test(test_loader)
    trainer2.plot_comprehensive_results(test_loader)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", default=PATCH_DIR, help="Path to image patch directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()
    
    run_deeplabv3plus(args.patch_dir, args.synthetic)
