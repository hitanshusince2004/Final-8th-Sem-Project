"""
Glacier Melting Detection - Dataset Processor
==============================================
Project : Final Year Project  |  GEE ID : final-year-project-485804

Reads ALL local CSVs from local_dataset/numerical/,
merges, cleans, splits, normalises, and generates EDA plots.

Run (from phase1_data_generation/):
    python scripts/process_downloaded_data.py

    # Custom paths:
    python scripts/process_downloaded_data.py \
        --csv_dir  ./local_dataset/numerical \
        --img_dir  ./local_dataset/patches \
        --out_dir  ./local_dataset/processed
"""

import os
import glob
import sys
import argparse
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import NUMERICAL_DIR, PATCHES_DIR, PROCESSED_DIR


# ── Column definitions ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
    "NDSI","NDWI","NDVI","BAI",
    "VV","VH","VV_lin","VH_lin","VV_VH_ratio","SAR_contrast","SAR_entropy",
    "LS_B2","LS_B3","LS_B4","LS_B5","LS_B6","LS_B7",
    "LS_Thermal_K","LST_Celsius","Emissivity","LS_NDSI","LS_NDWI","LS_NDVI",
    "elevation","slope","aspect",
    "longitude","latitude","year","is_melt_season",
]
LABEL_COLS   = ["glacier_mask", "multiclass_mask"]
INDEX_COLS   = ["NDSI","NDWI","NDVI","LST_Celsius"]


# ── 1. Merge ──────────────────────────────────────────────────────────────────
def merge_csvs(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    print(f"  Found {len(files)} CSV files in {csv_dir}")
    dfs = []
    for f in tqdm(files, desc="  Reading"):
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ Skipping {f}: {e}")
    if not dfs:
        raise RuntimeError(f"No CSV files found in {csv_dir}. Run generate_numerical_dataset.py first.")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Merged: {len(merged):,} rows x {merged.shape[1]} columns")
    return merged


# ── 2. Clean ──────────────────────────────────────────────────────────────────
def clean(df):
    print("  Cleaning...")
    drop = [c for c in df.columns if c.startswith(".") or c == "system:index"]
    df   = df.drop(columns=drop, errors="ignore")

    optical = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
               "LS_B2","LS_B3","LS_B4","LS_B5","LS_B6","LS_B7"]
    for c in optical:
        if c in df.columns: df[c] = df[c].clip(0, 1)

    for c in ["VV","VH"]:
        if c in df.columns: df[c] = df[c].clip(-35, 5)
    if "LST_Celsius" in df.columns:
        df["LST_Celsius"] = df["LST_Celsius"].clip(-60, 60)
    for c in ["NDSI","NDWI","NDVI","LS_NDSI","LS_NDWI","LS_NDVI","BAI"]:
        if c in df.columns: df[c] = df[c].clip(-1, 1)
    if "elevation" in df.columns: df["elevation"] = df["elevation"].clip(0, 8850)
    if "slope"     in df.columns: df["slope"]     = df["slope"].clip(0, 90)

    num_cols  = df.select_dtypes(include=[np.number]).columns
    na_counts = df[num_cols].isna().sum()
    for c in na_counts[na_counts > 0].index:
        df[c] = df[c].fillna(df[c].median())

    if "season" in df.columns and "is_melt_season" not in df.columns:
        df["is_melt_season"] = (df["season"] == "melt").astype(int)

    print(f"  Clean: {len(df):,} rows, {df.isna().sum().sum()} remaining NaN")
    return df


# ── 3. Derived features ───────────────────────────────────────────────────────
def add_features(df):
    if "LST_Celsius" in df.columns and "year" in df.columns:
        df["lst_anomaly"] = df["LST_Celsius"] - df.groupby("year")["LST_Celsius"].transform("mean")
    if "slope" in df.columns and "elevation" in df.columns:
        df["terrain_ruggedness"] = df["slope"] * (df["elevation"] / 8850.0)
    if "VV_lin" in df.columns and "VH_lin" in df.columns:
        df["sar_span"] = df["VV_lin"] + df["VH_lin"]
    return df


# ── 4. Split ──────────────────────────────────────────────────────────────────
def split(df):
    df["_strat"] = df.get("year", 0).astype(str) + "_" + df.get("glacier_mask", 0).astype(str)
    tr_val, test = train_test_split(df, test_size=0.15, stratify=df["_strat"], random_state=42)
    train, val   = train_test_split(tr_val, test_size=0.15/0.85, stratify=tr_val["_strat"], random_state=42)
    return train.drop(columns=["_strat"]), val.drop(columns=["_strat"]), test.drop(columns=["_strat"])


# ── 5. Normalise ──────────────────────────────────────────────────────────────
def normalise(df, feature_cols):
    avail  = [c for c in feature_cols if c in df.columns and c not in LABEL_COLS]
    scaler = StandardScaler()
    scaled = df.copy()
    scaled[avail] = scaler.fit_transform(df[avail].fillna(0))
    return scaled, scaler, avail


# ── 6. EDA plots ──────────────────────────────────────────────────────────────
def eda(df, out_dir):
    plots = os.path.join(out_dir, "eda_plots")
    os.makedirs(plots, exist_ok=True)

    # Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "glacier_mask" in df.columns:
        cnt = df["glacier_mask"].value_counts()
        axes[0].bar(["Non-glacier","Glacier"], [cnt.get(0,0), cnt.get(1,0)],
                    color=["#3B8BD4","#1D9E75"])
        axes[0].set_title("Binary class distribution"); axes[0].set_ylabel("Count")
    if "multiclass_mask" in df.columns:
        mc = df["multiclass_mask"].value_counts().sort_index()
        axes[1].bar(["Land","Snow/Ice","Water","Debris"], [mc.get(i,0) for i in range(4)],
                    color=["#888780","#B5D4F4","#3B8BD4","#FAC775"])
        axes[1].set_title("Multi-class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots,"class_distribution.png"), dpi=150)
    plt.close()

    # Temporal trend
    if "glacier_mask" in df.columns and "year" in df.columns:
        trend = df.groupby(["year","is_melt_season"])["glacier_mask"].mean().unstack()
        fig, ax = plt.subplots(figsize=(10,4))
        if 1 in trend: ax.plot(trend.index, trend[1]*100, "o-", color="#E8593C", label="Melt")
        if 0 in trend: ax.plot(trend.index, trend[0]*100, "s--", color="#3B8BD4", label="Winter")
        ax.set_xlabel("Year"); ax.set_ylabel("Glacier %"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Glacier coverage 2018-2025")
        plt.tight_layout(); plt.savefig(os.path.join(plots,"temporal_trend.png"), dpi=150); plt.close()

    # Index distributions
    idxcols = [c for c in INDEX_COLS if c in df.columns]
    if idxcols:
        fig, axes = plt.subplots(1, len(idxcols), figsize=(4*len(idxcols), 3))
        if len(idxcols) == 1: axes = [axes]
        for ax, col in zip(axes, idxcols):
            subset_g = df[df.get("glacier_mask",pd.Series(0))==1][col].dropna()
            subset_n = df[df.get("glacier_mask",pd.Series(0))==0][col].dropna()
            
            if len(subset_g) > 0:
                g = subset_g.sample(min(3000, len(subset_g)))
                ax.hist(g, bins=60, alpha=.6, color="#1D9E75", label="Glacier")
            
            if len(subset_n) > 0:
                n = subset_n.sample(min(3000, len(subset_n)))
                ax.hist(n, bins=60, alpha=.6, color="#3B8BD4", label="Non-glacier")
                
            ax.set_title(col); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(plots,"index_distributions.png"), dpi=150); plt.close()

    print(f"  EDA plots -> {plots}/")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir",  default=NUMERICAL_DIR)
    parser.add_argument("--img_dir",  default=PATCHES_DIR)
    parser.add_argument("--out_dir",  default=PROCESSED_DIR)
    parser.add_argument("--skip_img", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n[1/5] Merging CSVs...")
    df = merge_csvs(args.csv_dir)

    print("\n[2/5] Cleaning...")
    df = clean(df)
    df = add_features(df)

    print("\n[3/5] Normalising...")
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_scaled, scaler, used = normalise(df, feat_cols)

    print("\n[4/5] Splitting...")
    train, val, test = split(df_scaled)
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    df.to_csv(os.path.join(args.out_dir, "numerical_full.csv"),  index=False)
    train.to_csv(os.path.join(args.out_dir, "numerical_train.csv"), index=False)
    val.to_csv(os.path.join(args.out_dir, "numerical_val.csv"),   index=False)
    test.to_csv(os.path.join(args.out_dir, "numerical_test.csv"),  index=False)
    with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "total_rows": len(df), "train_rows": len(train),
        "val_rows": len(val), "test_rows": len(test),
        "feature_count": len(used), "features": used,
        "label_cols": LABEL_COLS,
        "years": sorted(df["year"].unique().tolist()) if "year" in df.columns else [],
        "glacier_positive_pct": round(float(df["glacier_mask"].mean()*100), 2) if "glacier_mask" in df.columns else None,
    }
    with open(os.path.join(args.out_dir, "dataset_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[5/5] EDA plots...")
    eda(df, args.out_dir)

    print(f"\n{'='*60}")
    print(f"  PROCESSING COMPLETE")
    print(f"  Output: {args.out_dir}")
    print(f"  Rows:   {len(df):,}  |  Features: {len(used)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
