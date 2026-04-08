"""
Glacier Melting Detection — Inference & Output Map Generator
============================================================
Loads trained checkpoints and generates all required output maps:

  1. Binary segmentation mask (glacier vs non-glacier)
  2. Multi-class segmentation map (snow/ice/water/debris/land)
  3. Probability map (per-pixel glacier probability)
  4. Glacier area change map (2018–2025 diff)
  5. Melt rate map (temporal NDSI change)
  6. Surface temperature map (from LST)
  7. Water body expansion map

Usage:
    python training/inference.py \
        --model unet \
        --checkpoint checkpoints/unet_best.pth \
        --input_dir  processed_dataset/patches/test \
        --output_dir results/maps
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.ml_config import (
    N_INPUT_CH, N_CLASSES, INPUT_CHANNELS, LABEL_CH_BINARY,
    UNET_CONFIG, CNN_CONFIG, DEEPLAB_CONFIG, RESULTS_DIR,
)
from models.unet_model    import build_unet
from models.cnn_model     import build_cnn
from models.deeplab_model import build_deeplabv3plus


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR MAPS
# ══════════════════════════════════════════════════════════════════════════════

MULTICLASS_CMAP = mcolors.ListedColormap([
    "#888780",   # 0 = Land
    "#B5D4F4",   # 1 = Snow/Ice
    "#3B8BD4",   # 2 = Water
    "#FAC775",   # 3 = Debris ice
])
MULTICLASS_LABELS = ["Land", "Snow/Ice", "Water", "Debris ice"]

PROB_CMAP  = plt.cm.YlGnBu
MELT_CMAP  = plt.cm.RdYlBu_r
TEMP_CMAP  = plt.cm.coolwarm
WATER_CMAP = plt.cm.Blues


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_type, checkpoint_path, device):
    """Load a trained model from a checkpoint file."""
    builders = {
        "unet":    (build_unet,           UNET_CONFIG),
        "cnn":     (lambda cfg: build_cnn(cfg, segmentation=True), CNN_CONFIG),
        "deeplab": (build_deeplabv3plus,  DEEPLAB_CONFIG),
    }
    build_fn, config = builders[model_type]
    model = build_fn(config).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded {model_type} checkpoint (epoch {epoch})")
    else:
        print(f"  ⚠ Checkpoint not found: {checkpoint_path} — using untrained weights")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ON A SINGLE PATCH
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_patch(model, image_tensor, device, tta=True):
    """
    Run inference on a single (1, C, H, W) tensor.
    tta=True applies test-time augmentation (h-flip, v-flip) and averages.

    Returns:
        prob_map:  (H, W, C) numpy array — class probabilities
        pred_mask: (H, W)    numpy array — argmax class label
        glacier_prob: (H, W) numpy array — P(glacier) = P(class 1)
    """
    image = image_tensor.to(device)

    if tta:
        preds = []
        # Original
        logits = model(image)
        preds.append(F.softmax(logits, dim=1))
        # H-flip
        logits_hf = model(torch.flip(image, dims=[3]))
        preds.append(torch.flip(F.softmax(logits_hf, dim=1), dims=[3]))
        # V-flip
        logits_vf = model(torch.flip(image, dims=[2]))
        preds.append(torch.flip(F.softmax(logits_vf, dim=1), dims=[2]))

        probs = torch.stack(preds, dim=0).mean(dim=0)   # (1, C, H, W)
    else:
        logits = model(image)
        probs  = F.softmax(logits, dim=1)

    probs_np     = probs.squeeze(0).cpu().numpy()         # (C, H, W)
    prob_map     = probs_np.transpose(1, 2, 0)            # (H, W, C)
    pred_mask    = probs_np.argmax(axis=0)                # (H, W)
    glacier_prob = probs_np[1]                            # (H, W) — P(snow/ice)

    return prob_map, pred_mask, glacier_prob


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT MAP GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def save_geotiff(array, path, profile=None):
    """Save a 2D numpy array as a single-band GeoTIFF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = array.shape
    default_profile = {
        "driver": "GTiff", "dtype": "float32",
        "count": 1, "height": h, "width": w,
        "crs": "EPSG:4326",
        "transform": from_bounds(0, 0, 1, 1, w, h),
        "compress": "lzw",
    }
    if profile:
        default_profile.update(profile)
    with rasterio.open(path, "w", **default_profile) as dst:
        dst.write(array.astype(np.float32), 1)


def save_multiband_geotiff(array, path, n_bands, profile=None):
    """Save (H, W, C) array as C-band GeoTIFF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = array.shape[:2]
    default_profile = {
        "driver": "GTiff", "dtype": "float32",
        "count": n_bands, "height": h, "width": w,
        "crs": "EPSG:4326",
        "transform": from_bounds(0, 0, 1, 1, w, h),
        "compress": "lzw",
    }
    if profile:
        default_profile.update(profile)
    with rasterio.open(path, "w", **default_profile) as dst:
        for b in range(n_bands):
            dst.write(array[:, :, b].astype(np.float32), b + 1)


def visualise_segmentation(pred_mask, rgb=None, save_path=None, title=""):
    """Visualise multi-class segmentation mask with optional RGB overlay."""
    fig, axes = plt.subplots(1, 2 if rgb is not None else 1, figsize=(12, 5))
    axes = [axes] if not isinstance(axes, np.ndarray) else axes

    im = axes[0].imshow(pred_mask, cmap=MULTICLASS_CMAP, vmin=0, vmax=N_CLASSES-1)
    axes[0].set_title(title or "Segmentation mask")
    axes[0].axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=MULTICLASS_CMAP.colors[i], label=MULTICLASS_LABELS[i])
                       for i in range(N_CLASSES)]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=8)

    if rgb is not None and len(axes) > 1:
        axes[1].imshow(np.clip(rgb, 0, 1))
        axes[1].set_title("RGB composite")
        axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def generate_area_change_map(mask_2018, mask_2025, save_path=None):
    """
    Compute glacier area change between two binary masks.
    Values: +1 = new glacier, -1 = lost glacier, 0 = unchanged.
    """
    glacier_2018 = (mask_2018 == 1).astype(int)
    glacier_2025 = (mask_2025 == 1).astype(int)
    change_map   = glacier_2025 - glacier_2018    # -1, 0, +1

    if save_path:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(change_map, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Change (-1=loss, +1=gain)")
        ax.set_title("Glacier area change (2018 → 2025)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return change_map


def compute_melt_rate_proxy(ndsi_series):
    """
    Compute per-pixel melt rate proxy from a time series of NDSI values.
    Melt rate = linear slope of NDSI over time (negative = melting).
    ndsi_series: list of (H, W) NDSI arrays ordered by time.
    Returns (H, W) slope array.
    """
    n   = len(ndsi_series)
    X   = np.arange(n, dtype=np.float32)
    X_c = X - X.mean()
    denom = (X_c ** 2).sum()
    stacked = np.stack(ndsi_series, axis=0)   # (T, H, W)
    stacked_c = stacked - stacked.mean(axis=0, keepdims=True)
    slope = (X_c[:, None, None] * stacked_c).sum(axis=0) / (denom + 1e-8)
    return slope


# ══════════════════════════════════════════════════════════════════════════════
# BATCH INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model_type, checkpoint_path, input_dir, output_dir,
                  use_tta=True, max_patches=None):
    """
    Run inference over all patches in input_dir and generate output maps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(model_type, checkpoint_path, device)

    patch_files = sorted(Path(input_dir).glob("**/*.tif"))
    if max_patches:
        patch_files = patch_files[:max_patches]

    print(f"\n  Running inference on {len(patch_files)} patches…")
    os.makedirs(output_dir, exist_ok=True)

    all_masks    = []
    all_probs    = []
    all_glacier  = []

    for i, pf in enumerate(patch_files):
        # Load patch
        with rasterio.open(pf) as src:
            data    = src.read().astype(np.float32)
            profile = src.profile

        data = np.nan_to_num(data, nan=0.0)

        # Select input channels
        image = data[INPUT_CHANNELS]   # (11, H, W)

        # Per-channel normalise
        for c in range(image.shape[0]):
            mn, mx = image[c].min(), image[c].max()
            if mx - mn > 1e-6:
                image[c] = (image[c] - mn) / (mx - mn)

        image_t = torch.tensor(image).unsqueeze(0)   # (1, 11, H, W)

        prob_map, pred_mask, glacier_prob = predict_patch(model, image_t, device, tta=use_tta)

        all_masks.append(pred_mask)
        all_probs.append(prob_map)
        all_glacier.append(glacier_prob)

        # Save per-patch outputs
        stem = pf.stem
        save_geotiff(pred_mask.astype(np.float32),
                     os.path.join(output_dir, "masks",   f"{stem}_mask.tif"), profile)
        save_geotiff(glacier_prob,
                     os.path.join(output_dir, "prob",    f"{stem}_prob.tif"),  profile)
        save_multiband_geotiff(prob_map,
                     os.path.join(output_dir, "prob_mc", f"{stem}_prob_mc.tif"), N_CLASSES, profile)

        # Visualisation for first 10 patches
        if i < 10:
            # Try to get RGB from patch (bands 0,1,2 = B4,B3,B2)
            rgb = data[:3].transpose(1, 2, 0)
            rgb = np.clip(rgb / (rgb.max() + 1e-6), 0, 1)
            visualise_segmentation(
                pred_mask, rgb,
                save_path=os.path.join(output_dir, "viz", f"{stem}_seg.png"),
                title=f"{model_type.upper()} — {stem}",
            )

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(patch_files)}")

    print(f"\n  ✓ Inference complete. Outputs saved to: {output_dir}/")

    # ── Summary statistics ────────────────────────────────────────────────────
    all_masks_arr = np.stack(all_masks, axis=0)    # (N, H, W)
    class_dist    = {
        MULTICLASS_LABELS[c]: float((all_masks_arr == c).mean()) * 100
        for c in range(N_CLASSES)
    }
    print("\n  Class distribution across all patches:")
    for cls, pct in class_dist.items():
        print(f"    {cls:<15} {pct:.1f}%")

    import json
    with open(os.path.join(output_dir, "inference_summary.json"), "w") as f:
        json.dump({
            "model":          model_type,
            "n_patches":      len(patch_files),
            "class_dist_pct": class_dist,
            "tta":            use_tta,
        }, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="unet",
                        choices=["unet", "cnn", "deeplab"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (default: auto-detect)")
    parser.add_argument("--input_dir",  default="processed_dataset/patches/test")
    parser.add_argument("--output_dir", default="results/maps")
    parser.add_argument("--no_tta",     action="store_true")
    parser.add_argument("--max",        type=int, default=None,
                        help="Max number of patches to process")
    args = parser.parse_args()

    # Auto-detect checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        auto = {
            "unet":    "checkpoints/unet_best.pth",
            "cnn":     "checkpoints/cnn_fcn_best.pth",
            "deeplab": "checkpoints/deeplabv3plus_best.pth",
        }
        ckpt = auto[args.model]

    run_inference(
        model_type      = args.model,
        checkpoint_path = ckpt,
        input_dir       = args.input_dir,
        output_dir      = args.output_dir,
        use_tta         = not args.no_tta,
        max_patches     = args.max,
    )


if __name__ == "__main__":
    main()
