"""
Glacier Melting Detection — CNN Models Training Script
======================================================
Trains CNN variants (Basic, ResNet18, GlacierFCN).
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.ml_config import CNN_CONFIG, CNN_TRAIN, PATCH_DIR
from models.cnn_model import build_cnn
from training.trainer import GlacierSegTrainer
from training.train_utils import get_image_loaders

def run_cnn_variants(patch_dir=PATCH_DIR, use_synthetic=False):
    """Train multiple CNN variants."""
    print("\n" + "═"*60)
    print("  PHASE 3a: CNN Variants — Multi-spectral Classification/Segmentation")
    print("═"*60)

    bs = CNN_TRAIN["batch_size"]
    results = {}

    # 1. Basic CNN (Classification)
    print("\n--- Training Basic CNN (Patch-level) ---")
    train_loader_clf, val_loader_clf, test_loader_clf = get_image_loaders(patch_dir, bs, use_synthetic, task="binary")
    model_basic = build_cnn(CNN_CONFIG, segmentation=False, model_type="basic")
    trainer_basic = GlacierSegTrainer(model_basic, train_loader_clf, val_loader_clf, config=CNN_TRAIN, model_name="CNN_Basic")
    trainer_basic.train()
    results["CNN_Basic"] = trainer_basic.evaluate_test(test_loader_clf)
    trainer_basic.plot_comprehensive_results(test_loader_clf)

    # 2. ResNet18 (Classification)
    print("\n--- Training ResNet18 (Patch-level) ---")
    model_resnet = build_cnn(CNN_CONFIG, segmentation=False, model_type="resnet18")
    trainer_resnet = GlacierSegTrainer(model_resnet, train_loader_clf, val_loader_clf, config=CNN_TRAIN, model_name="CNN_ResNet18")
    trainer_resnet.train()
    results["CNN_ResNet18"] = trainer_resnet.evaluate_test(test_loader_clf)
    trainer_resnet.plot_comprehensive_results(test_loader_clf)

    # 3. GlacierFCN (Segmentation)
    print("\n--- Training GlacierFCN (Pixel-level) ---")
    train_loader_seg, val_loader_seg, test_loader_seg = get_image_loaders(patch_dir, bs, use_synthetic, task="multiclass")
    model_fcn = build_cnn(CNN_CONFIG, segmentation=True, model_type="fcn")
    trainer_fcn = GlacierSegTrainer(model_fcn, train_loader_seg, val_loader_seg, config=CNN_TRAIN, model_name="CNN_FCN")
    trainer_fcn.train()
    results["CNN_FCN"] = trainer_fcn.evaluate_test(test_loader_seg)
    trainer_fcn.plot_comprehensive_results(test_loader_seg)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", default=PATCH_DIR, help="Path to image patch directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()
    
    run_cnn_variants(args.patch_dir, args.synthetic)
