"""
Glacier Melting Detection — U-Net Model Training Script
=======================================================
Trains the U-Net segmentation model.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.ml_config import UNET_CONFIG, UNET_TRAIN, PATCH_DIR
from models.unet_model import UNetLite
from training.trainer import GlacierSegTrainer
from training.train_utils import get_image_loaders

def run_unet(patch_dir=PATCH_DIR, use_synthetic=False):
    """Train the U-Net segmentation model."""
    print("\n" + "═"*60)
    print("  PHASE 3b: U-Net — Encoder-Decoder Segmentation")
    print("═"*60)

    bs = UNET_TRAIN["batch_size"]
    train_loader, val_loader, test_loader = get_image_loaders(
        patch_dir, bs, use_synthetic, task="multiclass"
    )

    # Use UNetLite with low base filters for CPU stability
    from models.unet_model import UNetLite
    model = UNetLite(in_channels=UNET_CONFIG["in_channels"], 
                     num_classes=UNET_CONFIG["num_classes"],
                     base_filters=8)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  U-Net Lite parameters: {params:,}")

    trainer = GlacierSegTrainer(
        model, train_loader, val_loader,
        config     = UNET_TRAIN,
        model_name = "UNet",
    )
    trainer.train()
    results = trainer.evaluate_test(test_loader)
    trainer.plot_comprehensive_results(test_loader)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", default=PATCH_DIR, help="Path to image patch directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()
    
    run_unet(args.patch_dir, args.synthetic)
