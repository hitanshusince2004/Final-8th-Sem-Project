"""
Glacier Melting Detection — Main Training Script
=================================================
Orchestrates training of all 5 models by calling individual scripts:
  1. Linear Regression & Random Forest (train_ml.py)
  2. CNN (GlacierFCN) (train_cnn.py)
  3. U-Net (train_unet.py)
  4. DeepLabv3+ (train_deeplab.py)

Usage:
    # Train all models
    python training/train_all.py

    # Train specific models
    python training/train_all.py --models ml dl_cnn dl_unet

    # Use synthetic data (no real patches needed)
    python training/train_all.py --synthetic

    # Train only DL models with custom patch dir
    python training/train_all.py --models dl --patch_dir ./processed_dataset/patches
"""

print("Initializing Glacier Melting Detection Training Pipeline...")
print("   (Loading configuration and basic modules...)\n")

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.ml_config import PATCH_DIR, RESULTS_DIR, CHECKPOINTS_DIR, SEED
from training.train_utils import set_seed, get_device

# Import individual runners
from training.train_ml import run_ml_models
from training.train_cnn import run_cnn_variants
from training.train_unet import run_unet
from training.train_deeplab import run_deeplabv3plus

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON & SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compile_summary(all_results):
    """Print and save a model comparison table."""
    print("\n" + "═"*70)
    print("  ALL MODELS — FINAL COMPARISON")
    print("═"*70)

    # Determine all metric keys across all results
    metric_keys = set()
    for r in all_results.values():
        metric_keys.update(k for k in r.keys()
                           if k not in ("split", "model", "IoU_per_class",
                                        "Dice_per_class", "confusion_matrix",
                                        "test_loss"))
    metric_keys = sorted(metric_keys)

    header = f"  {'Model':<35} " + " ".join(f"{k[:8]:>10}" for k in metric_keys)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for model_name, metrics in all_results.items():
        row = f"  {model_name:<35} "
        for k in metric_keys:
            v = metrics.get(k, "—")
            row += f"{str(v):>10} "
        print(row)

    print("═"*70)

    # Save
    path = os.path.join(RESULTS_DIR, "all_models_summary.json")
    
    # Enrichment: Add task and type to results if missing
    enriched_results = {}
    for name, metrics in all_results.items():
        enriched = metrics.copy()
        
        # Determine task
        if "MAE" in enriched or "R2" in enriched:
            enriched["task"] = "regression"
        else:
            enriched["task"] = "segmentation"
            
        # Determine type
        if "CNN" in name or "U-Net" in name or "DeepLab" in name:
            enriched["type"] = "DL"
        else:
            enriched["type"] = "ML"
            
        enriched_results[name] = enriched

    with open(path, "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump({k: {mk: convert(mv) for mk, mv in v.items()}
                   for k, v in enriched_results.items()}, f, indent=2)
    print(f"\n  Summary saved → {path}")
    return enriched_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",    nargs="+",
                        choices=["ml", "dl_cnn", "dl_unet", "dl_deeplab", "dl", "all"],
                        default=["all"],
                        help="Which models to train")
    parser.add_argument("--patch_dir", default=PATCH_DIR,
                        help="Path to image patch directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no real patches needed)")
    parser.add_argument("--n_synthetic", type=int, default=500,
                        help="Number of synthetic samples (default: 500)")
    args = parser.parse_args()

    run_all = "all" in args.models
    run_dl  = "dl"  in args.models or run_all

    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    set_seed(SEED)
    device = get_device()
    all_results = {}

    # ── ML Models ─────────────────────────────────────────────────────────────
    if run_all or "ml" in args.models:
        ml_results = run_ml_models()
        all_results.update(ml_results)

    # ── CNN ───────────────────────────────────────────────────────────────────
    if run_all or run_dl or "dl_cnn" in args.models:
        cnn_results = run_cnn_variants(
            patch_dir     = args.patch_dir,
            use_synthetic = args.synthetic or not os.path.isdir(args.patch_dir),
        )
        all_results.update(cnn_results)

    # ── U-Net ─────────────────────────────────────────────────────────────────
    if run_all or run_dl or "dl_unet" in args.models:
        unet_results = run_unet(
            patch_dir     = args.patch_dir,
            use_synthetic = args.synthetic or not os.path.isdir(args.patch_dir),
        )
        all_results["U-Net"] = unet_results

    # ── DeepLabv3+ ────────────────────────────────────────────────────────────
    if run_all or run_dl or "dl_deeplab" in args.models:
        deeplab_results = run_deeplabv3plus(
            patch_dir     = args.patch_dir,
            use_synthetic = args.synthetic or not os.path.isdir(args.patch_dir),
        )
        all_results["DeepLabv3+"] = deeplab_results

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        compile_summary(all_results)
        
        # ── Research Plots ────────────────────────────────────────────────────
        print("\n" + "═"*60)
        print("  GENERATING RESEARCH-GRADE PLOTS")
        print("═"*60)
        from training.generate_research_plots import (
            generate_fig2, generate_fig2b, generate_fig3, 
            generate_fig4, generate_fig5, generate_fig6, generate_fig7
        )
        try: generate_fig2()
        except Exception as e: print(f"Error generating Fig 2: {e}")
        try: generate_fig2b()
        except Exception as e: print(f"Error generating Fig 2b: {e}")
        try: generate_fig3()
        except Exception as e: print(f"Error generating Fig 3: {e}")
        try: generate_fig4()
        except Exception as e: print(f"Error generating Fig 4: {e}")
        try: generate_fig5()
        except Exception as e: print(f"Error generating Fig 5: {e}")
        try: generate_fig6()
        except Exception as e: print(f"Error generating Fig 6: {e}")
        try: generate_fig7()
        except Exception as e: print(f"Error generating Fig 7: {e}")
        print(f"\n  All research figures generated in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
