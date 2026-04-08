"""
Glacier Melting Detection — Dataset Utilities
=============================================
PyTorch Dataset classes and DataLoader factories for:
  - NumericalGlacierDataset  → tabular CSV for ML/LR/RF
  - GlacierPatchDataset      → GeoTIFF patches for CNN / U-Net / DeepLabv3+
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import random
from pathlib import Path

from config.ml_config import (
    INPUT_CHANNELS, LABEL_CH_BINARY, LABEL_CH_MULTI,
    PATCH_SIZE, AUG_CONFIG, SEED, N_CLASSES,
)


# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL DATASET
# ══════════════════════════════════════════════════════════════════════════════

class NumericalGlacierDataset(Dataset):
    """
    PyTorch Dataset wrapping the tabular CSV.
    Used for CNN-on-tabular or simple MLP baselines.
    For sklearn models, use .to_numpy() directly.
    """

    def __init__(self, csv_path, feature_cols, target_col, scaler=None):
        df = pd.read_csv(csv_path)
        available_features = [c for c in feature_cols if c in df.columns]
        self.X = df[available_features].fillna(0).values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.feature_names = available_features
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    def to_numpy(self):
        """Return (X, y) numpy arrays — used by sklearn models."""
        return self.X, self.y


def load_numerical_splits(train_csv, val_csv, test_csv, feature_cols, target_col):
    """
    Load all three splits as NumericalGlacierDataset objects.
    Returns (train_ds, val_ds, test_ds).
    """
    train = NumericalGlacierDataset(train_csv, feature_cols, target_col)
    val   = NumericalGlacierDataset(val_csv,   feature_cols, target_col)
    test  = NumericalGlacierDataset(test_csv,  feature_cols, target_col)
    print(f"  Numerical splits: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class GlacierAugment:
    """
    Deterministic-seed augmentations for multi-band satellite patches.
    All transforms are applied identically to image and mask.
    """

    def __init__(self, cfg=AUG_CONFIG, training=True):
        self.cfg      = cfg
        self.training = training

    def __call__(self, image, mask):
        """
        image: (C, H, W) float32 tensor
        mask:  (H, W)    int64 tensor
        Returns: augmented (image, mask)
        """
        if not self.training:
            return image, mask

        # Random horizontal flip
        if self.cfg.get("horizontal_flip") and random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            mask  = torch.flip(mask,  dims=[1])

        # Random vertical flip
        if self.cfg.get("vertical_flip") and random.random() > 0.5:
            image = torch.flip(image, dims=[1])
            mask  = torch.flip(mask,  dims=[0])

        # Random 90-degree rotations
        if self.cfg.get("rotate_90"):
            k = random.randint(0, 3)
            if k > 0:
                image = torch.rot90(image, k, dims=[1, 2])
                mask  = torch.rot90(mask,  k, dims=[0, 1])

        # Random crop
        crop_size = self.cfg.get("random_crop")
        if crop_size and image.shape[1] > crop_size:
            h, w  = image.shape[1], image.shape[2]
            top   = random.randint(0, h - crop_size)
            left  = random.randint(0, w - crop_size)
            image = image[:, top:top+crop_size, left:left+crop_size]
            mask  = mask[top:top+crop_size, left:left+crop_size]

        # Brightness / contrast jitter on optical channels only (first 5 bands)
        brightness = self.cfg.get("brightness", 0)
        contrast   = self.cfg.get("contrast", 0)
        if brightness > 0:
            delta = random.uniform(-brightness, brightness)
            image[:5] = image[:5] + delta
        if contrast > 0:
            factor = random.uniform(1 - contrast, 1 + contrast)
            mean   = image[:5].mean(dim=[1, 2], keepdim=True)
            image[:5] = (image[:5] - mean) * factor + mean

        # Gaussian noise
        noise_std = self.cfg.get("gaussian_noise", 0)
        if noise_std > 0:
            image = image + torch.randn_like(image) * noise_std

        return image, mask


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PATCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class GlacierPatchDataset(Dataset):
    """
    PyTorch Dataset for multi-band GeoTIFF patches.

    Each patch file contains 13 bands:
      [0..8]  = B4,B3,B2,B8,B11,NDSI,NDWI,glacier_mask,multiclass_mask
      [9..12] = VV,VH,LST_Celsius,elevation

    INPUT_CHANNELS selects the 11 model input bands (excludes mask bands 7,8).
    Label channels are 7 (binary) and 8 (multi-class).
    """

    def __init__(self, patch_dir, split="train", task="binary",
                 augment=True, patch_size=PATCH_SIZE,
                 input_channels=INPUT_CHANNELS,
                 label_ch_binary=LABEL_CH_BINARY,
                 label_ch_multi=LABEL_CH_MULTI):

        self.patch_size     = patch_size
        self.input_channels = input_channels
        self.label_ch       = label_ch_binary if task == "binary" else label_ch_multi
        self.task           = task

        # Find all GeoTIFF files for this split
        split_dir = os.path.join(patch_dir, split) if os.path.isdir(os.path.join(patch_dir, split)) else patch_dir
        self.files = sorted(glob.glob(os.path.join(split_dir, "**", "*.tif"), recursive=True))

        if len(self.files) == 0:
            # Fallback: use all patches and split by index
            all_files = sorted(glob.glob(os.path.join(patch_dir, "**", "*.tif"), recursive=True))
            n = len(all_files)
            rng = np.random.RandomState(SEED)
            idx = rng.permutation(n)
            train_end = int(0.70 * n)
            val_end   = int(0.85 * n)
            if split == "train":
                self.files = [all_files[i] for i in idx[:train_end]]
            elif split == "val":
                self.files = [all_files[i] for i in idx[train_end:val_end]]
            else:
                self.files = [all_files[i] for i in idx[val_end:]]

        # Per-channel statistics for normalisation (computed lazily)
        self._mean = None
        self._std  = None

        self.augment = GlacierAugment(training=(augment and split == "train"))
        print(f"  GlacierPatchDataset [{split}]: {len(self.files):,} patches, task={task}")

    def __len__(self):
        return len(self.files)

    def _load_patch(self, path):
        """Read GeoTIFF, return (C, H, W) float32 array."""
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)   # (C, H, W)
        return data

    def _normalise(self, image):
        """Per-channel min-max normalisation to [0, 1]."""
        for c in range(image.shape[0]):
            cmin = image[c].min()
            cmax = image[c].max()
            if cmax - cmin > 1e-6:
                image[c] = (image[c] - cmin) / (cmax - cmin)
        return image

    def _resize(self, arr, target):
        """Simple nearest-neighbour resize to (C, target, target) or (target, target)."""
        from torch.nn.functional import interpolate
        if arr.ndim == 2:
            t = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float()
            t = interpolate(t, size=(target, target), mode="nearest")
            return t.squeeze(0).squeeze(0)
        else:
            t = torch.tensor(arr).unsqueeze(0)
            t = interpolate(t, size=(target, target), mode="bilinear", align_corners=False)
            return t.squeeze(0)

    def __getitem__(self, idx):
        path = self.files[idx]

        try:
            data = self._load_patch(path)
        except Exception:
            # Return zeros on corrupt file
            data = np.zeros((13, self.patch_size, self.patch_size), dtype=np.float32)

        # Replace NaN / Inf
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

        # Extract input bands and label
        image = data[self.input_channels]    # (11, H, W)
        label = data[self.label_ch]          # (H, W)

        # Normalise image channels
        image = self._normalise(image)

        # Convert to tensors
        image_t = torch.tensor(image)
        label_t = torch.tensor(label).long()

        # Resize if patch is wrong size
        if image_t.shape[1] != self.patch_size or image_t.shape[2] != self.patch_size:
            image_t = self._resize(image_t.numpy(), self.patch_size)
            label_t = self._resize(label_t.numpy(), self.patch_size).long()

        # Apply augmentations
        image_t, label_t = self.augment(image_t, label_t)

        return image_t, label_t


# ══════════════════════════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(patch_dir, task="binary", batch_size=8,
                    num_workers=0, pin_memory=True):
    """
    Returns (train_loader, val_loader, test_loader) for image segmentation.
    """
    train_ds = GlacierPatchDataset(patch_dir, split="train", task=task, augment=True)
    val_ds   = GlacierPatchDataset(patch_dir, split="val",   task=task, augment=False)
    test_ds  = GlacierPatchDataset(patch_dir, split="test",  task=task, augment=False)

    kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    print(f"  DataLoaders ready: train={len(train_loader)} batches, "
          f"val={len(val_loader)}, test={len(test_loader)}")

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# DEMO SYNTHETIC DATA (when real patches not yet available)
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticGlacierDataset(Dataset):
    """
    Generates synthetic patches for testing the pipeline before
    real GEE data is downloaded.
    Simulates realistic value ranges per band.
    """

    def __init__(self, n_samples=500, patch_size=256, n_channels=11,
                 n_classes=4, task="binary"):
        self.n          = n_samples
        self.ps         = patch_size
        self.nc         = n_channels
        self.n_classes  = n_classes
        self.task       = task
        rng = np.random.RandomState(SEED)

        self.images = []
        self.labels = []

        for _ in range(n_samples):
            # Simulate spectral bands [0,1]
            img = rng.rand(n_channels, patch_size, patch_size).astype(np.float32)

            # Simulate spatially coherent glacier patch (random ellipse)
            mask = np.zeros((patch_size, patch_size), dtype=np.int64)
            cy, cx = rng.randint(50, patch_size-50, 2)
            ry, rx = rng.randint(20, 80, 2)
            Y, X   = np.ogrid[:patch_size, :patch_size]
            ellipse = ((Y - cy)/ry)**2 + ((X - cx)/rx)**2 <= 1
            if self.task == "binary":
                mask[ellipse] = 1
            else:
                mask[ellipse] = 1
                # Add some water pixels at glacier edge
                edge = np.zeros_like(mask, dtype=bool)
                edge[max(0,cy-ry-5):cy+ry+5, max(0,cx-rx-5):cx+rx+5] = True
                edge = edge & ~ellipse
                mask[edge & (rng.rand(patch_size, patch_size) > 0.8)] = 2
                # Debris-covered ice
                mask[ellipse & (rng.rand(patch_size, patch_size) > 0.85)] = 3

            self.images.append(torch.tensor(img))
            self.labels.append(torch.tensor(mask))

        print(f"  SyntheticGlacierDataset: {n_samples} samples, patch={patch_size}px, ch={n_channels}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
