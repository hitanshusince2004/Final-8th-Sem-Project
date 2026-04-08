"""
Glacier Melting Detection - ML/DL Configuration
================================================
Central config for all model training, evaluation, and paths.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Correct DATA_DIR to point to phase1 output
DATA_DIR        = os.path.join(os.path.dirname(BASE_DIR), "phase1_data_generation", "local_dataset", "processed")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

for d in [CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Numerical Dataset ─────────────────────────────────────────────────────────
TRAIN_CSV = os.path.join(DATA_DIR, "numerical_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "numerical_val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "numerical_test.csv")

# Features to use for ML models (subset - drop metadata cols)
ML_FEATURE_COLS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "NDSI", "NDWI", "NDVI", "BAI",
    "VV", "VH", "VV_lin", "VH_lin", "VV_VH_ratio",
    "SAR_contrast", "SAR_entropy",
    "LS_B2", "LS_B3", "LS_B4", "LS_B5", "LS_B6", "LS_B7",
    "LST_Celsius", "Emissivity", "LS_NDSI", "LS_NDWI", "LS_NDVI",
    "elevation", "slope", "aspect",
    "is_melt_season",
    "lst_anomaly", "terrain_ruggedness", "sar_span",
]

# Regression target (melt rate proxy = NDSI change is computed externally;
# here we use LST_Celsius as a continuous regression target)
REGRESSION_TARGET = "LST_Celsius"

# Classification target
BINARY_TARGET     = "glacier_mask"
MULTICLASS_TARGET = "multiclass_mask"
N_CLASSES         = 4

# ── Image Dataset ─────────────────────────────────────────────────────────────
PATCH_DIR   = os.path.join(os.path.dirname(DATA_DIR), "patches")
PATCH_SIZE  = 256
N_CHANNELS  = 13    # B4,B3,B2,B8,B11,NDSI,NDWI,glacier_mask,multiclass_mask,VV,VH,LST,elevation

# Input channels for DL (exclude label channels 7,8)
INPUT_CHANNELS  = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12]   # 11 bands
N_INPUT_CH      = len(INPUT_CHANNELS)
LABEL_CH_BINARY = 7    # glacier_mask band index in patch
LABEL_CH_MULTI  = 8    # multiclass_mask band index in patch

# ── Random Forest ─────────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":      500,
    "max_depth":         None,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "n_jobs":            -1,
    "random_state":      42,
    "oob_score":         True,
}

RF_REGRESSION_PARAMS = {
    "n_estimators":      500,
    "max_depth":         None,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "n_jobs":            -1,
    "random_state":      42,
    "oob_score":         True,
}

# ── Linear Regression ─────────────────────────────────────────────────────────
LR_PARAMS = {
    "fit_intercept": True,
    "n_jobs":        -1,
}

# Logistic Regression for classification
LR_CLF_PARAMS = {
    "max_iter":      1000,
    "class_weight":  "balanced",
    "solver":        "lbfgs",
    "C":             1.0,
    "random_state":  42,
    "n_jobs":        -1,
}

# ── CNN ───────────────────────────────────────────────────────────────────────
CNN_CONFIG = {
    "in_channels":     N_INPUT_CH,
    "num_classes":     N_CLASSES,
    "base_filters":    32,
    "dropout":         0.3,
}

CNN_TRAIN = {
    "epochs":          1,
    "batch_size":      4,
    "lr":              1e-3,
    "weight_decay":    1e-4,
    "lr_scheduler":    "cosine",
    "warmup_epochs":   1,
    "early_stop":      5,
    "mixed_precision": False,
}

# ── U-Net ─────────────────────────────────────────────────────────────────────
UNET_CONFIG = {
    "in_channels":     N_INPUT_CH,
    "num_classes":     N_CLASSES,
    "base_filters":    8,
    "depth":           3,
    "dropout":         0.1,
    "bilinear":        True,
    "attention":       False,
}

UNET_TRAIN = {
    "epochs":          1,
    "batch_size":      1,
    "lr":              3e-4,
    "weight_decay":    1e-5,
    "lr_scheduler":    "cosine",
    "early_stop":      5,
    "mixed_precision": False,
}

# ── DeepLabv3+ ────────────────────────────────────────────────────────────────
DEEPLAB_CONFIG = {
    "in_channels":     N_INPUT_CH,
    "num_classes":     N_CLASSES,
    "backbone":        "mobilenet_v2",
    "output_stride":   16,
    "aspp_dilations":  [6, 12, 18],
    "dropout":         0.1,
}

DEEPLAB_TRAIN = {
    "epochs":          1,
    "batch_size":      1,
    "lr":              1e-4,
    "weight_decay":    5e-5,
    "lr_scheduler":    "poly",
    "early_stop":      5,
    "mixed_precision": False,
}

# ── Data Augmentation ─────────────────────────────────────────────────────────
AUG_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip":   True,
    "rotate_90":       True,
    "random_crop":     224,     # crop to 224x224 from 256x256
    "brightness":      0.1,
    "contrast":        0.1,
    "gaussian_noise":  0.01,
}

# ── Class weights for imbalanced segmentation ─────────────────────────────────
# Approximate: land=55%, snow/ice=30%, water=8%, debris=7%
CLASS_WEIGHTS = [0.5, 1.5, 2.5, 2.5]

SEED = 42
