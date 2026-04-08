"""
Glacier Melting Detection - Numerical Dataset Generator
========================================================
Project : Final Year Project  |  GEE ID : final-year-project-485804
Exports : *** DIRECTLY TO LOCAL DISK *** - no Google Drive, no GCS

Strategy
--------
Instead of GEE batch Export (which needs Drive/GCS), this script:
  1. Builds a seasonal composite per year
  2. Calls ee.Image.sampleRegions() with stratified sampling
  3. Pulls data via .getInfo() in safe batches of ≤5000 rows
  4. Saves each batch immediately as a CSV chunk
  5. Merges all chunks into numerical_YEAR_SEASON.csv

Run
---
    cd phase1_data_generation
    python scripts/generate_numerical_dataset.py

    # Single year:
    python scripts/generate_numerical_dataset.py --year 2022 --season melt

Output (local_dataset/numerical/)
---
    numerical_2018_melt.csv
    numerical_2018_winter.csv
    ...
    numerical_2025_winter.csv          (14 files × ~1500 rows = ~21000 rows)
"""

import ee
import os
import sys
import csv
import time
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    GEE_PROJECT, AOI_COORDS, YEARS,
    MELT_SEASON, WINTER_SEASON,
    NUMERICAL_DIR, CSV_SCALE, SAMPLES_PER_YEAR, SAMPLE_SEED,
)
from utils.gee_utils import (
    build_s2_composite, build_s1_composite,
    build_landsat_composite, get_dem,
)


# ── Initialise GEE ────────────────────────────────────────────────────────────
def init_ee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"Earth Engine initialised  (project: {GEE_PROJECT})")
    except Exception:
        print("  Authenticating...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        print(f"Earth Engine initialised  (project: {GEE_PROJECT})")


# ── Build full stacked image ──────────────────────────────────────────────────
def build_stack(aoi, year, season_dict):
    s2  = build_s2_composite(aoi, year, season_dict)
    s1  = build_s1_composite(aoi, year, season_dict)
    ls  = build_landsat_composite(aoi, year, season_dict)
    dem = get_dem(aoi)

    s2_sel = s2.select([
        "B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
        "NDSI","NDWI","NDVI","BAI",
    ])
    
    # Cast masks back to integer after median() operation
    glacier_mask = s2.select("glacier_mask").gt(0.5).uint8().rename("glacier_mask")
    multi_mask   = s2.select("multiclass_mask").round().uint8().rename("multiclass_mask")
    s2_sel = s2_sel.addBands([glacier_mask, multi_mask])
    s1_sel = s1.select(["VV","VH","VV_lin","VH_lin","VV_VH_ratio","SAR_contrast","SAR_entropy"])
    ls_sel = ls.select([
        "SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7",
        "ST_B10","LST_Celsius","Emissivity","NDSI","NDWI","NDVI",
    ]).rename([
        "LS_B2","LS_B3","LS_B4","LS_B5","LS_B6","LS_B7",
        "LS_Thermal_K","LST_Celsius","Emissivity","LS_NDSI","LS_NDWI","LS_NDVI",
    ])
    dem_sel = dem.select(["elevation","slope","aspect"])

    year_band   = ee.Image.constant(year).int16().rename("year")
    season_band = ee.Image.constant(1).uint8().rename("is_melt_season")

    return s2_sel.addBands(s1_sel).addBands(ls_sel).addBands(dem_sel).addBands([year_band, season_band]).clip(aoi)


# ── Pull samples via getInfo() in batches ─────────────────────────────────────
def pull_samples_local(stack, aoi, n_samples, year, season_name, seed):
    """
    Sample points -> pull via getInfo() -> return list of dicts.
    Using randomPoints + sampleRegions for maximum reliability on large AOIs.
    """
    # Sample more points than needed because some will hit 'no-data' areas
    oversample = int(n_samples * 2.5)
    print(f"    Generating {oversample} random points across AOI...")
    
    # Generate random points - this is very fast in GEE
    points = ee.FeatureCollection.randomPoints(region=aoi, points=oversample, seed=seed)
    
    # Sample the image stack at these points - also relatively fast
    fc = stack.sampleRegions(
        collection = points,
        scale      = CSV_SCALE,
        geometries = True,
        tileScale  = 16
    )

    # Add metadata
    def add_meta(f):
        return f.set({
            "longitude": f.geometry().coordinates().get(0),
            "latitude":  f.geometry().coordinates().get(1),
            "year":      year,
            "season":    season_name,
        })
    fc = fc.map(add_meta)

    print(f"    Pulling data via getInfo()...")
    
    max_retries = 3
    features = []
    
    for attempt in range(max_retries):
        try:
            # Get all at once if under 5000
            data = fc.getInfo()
            features = data.get("features", [])
            if features: break
        except Exception as e:
            print(f"    Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in smaller batches...")
            try:
                # Try smaller batches if the full pull fails
                total = fc.size().getInfo()
                features = []
                batch = 400
                offset = 0
                while offset < total:
                    chunk = fc.toList(batch, offset).getInfo()
                    features.extend(chunk)
                    offset += batch
                    print(f"      Pulled {len(features)}/{total} rows...")
                    time.sleep(1)
                if features: break
            except Exception as e2:
                print(f"      Batch pull failed: {e2}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue

    all_rows = []
    for feat in features:
        props = feat.get("properties", {})
        # Filter out features that have mostly NaNs (where sampling hit no data)
        # We need at least basic bands to be valid
        if props.get("B2") is not None and props.get("glacier_mask") is not None:
            row = {k: v for k, v in props.items() if not k.startswith(".")}
            all_rows.append(row)
            if len(all_rows) >= n_samples:
                break

    print(f"    GEE returned {len(all_rows)} valid samples (target: {n_samples}).")
    return all_rows


# ── Save rows to CSV ──────────────────────────────────────────────────────────
def save_csv(rows, path):
    if not rows:
        print(f"    ⚠ No rows to save for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Saved {len(rows):,} rows -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",   type=int, help="Single year (2018–2025)")
    parser.add_argument("--season", type=str, choices=["melt","winter"], help="Single season")
    args = parser.parse_args()

    init_ee()
    aoi = ee.Geometry.Polygon(AOI_COORDS)

    years   = [args.year]  if args.year   else YEARS
    seasons = {}
    if args.season == "melt":
        seasons = {"melt": MELT_SEASON}
    elif args.season == "winter":
        seasons = {"winter": WINTER_SEASON}
    else:
        seasons = {"melt": MELT_SEASON, "winter": WINTER_SEASON}

    total_rows  = 0
    total_files = 0

    for year in years:
        for season_name, season_dict in seasons.items():
            print(f"\n{'-'*60}")
            print(f"  Year: {year}   Season: {season_name}")
            print(f"{'-'*60}")

            out_path = os.path.join(NUMERICAL_DIR, f"numerical_{year}_{season_name}.csv")

            if os.path.exists(out_path):
                print(f"  Already exists - skipping: {out_path}")
                total_files += 1
                continue

            try:
                print(f"  Building spectral stack...")
                stack = build_stack(aoi, year, season_dict)

                rows = pull_samples_local(
                    stack, aoi,
                    n_samples   = SAMPLES_PER_YEAR,
                    year        = year,
                    season_name = season_name,
                    seed        = SAMPLE_SEED + year,
                )

                save_csv(rows, out_path)
                total_rows  += len(rows)
                total_files += 1

            except Exception as e:
                print(f"  Error for {year}/{season_name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  NUMERICAL DATASET COMPLETE")
    print(f"  Files saved : {total_files}")
    print(f"  Total rows  : {total_rows:,}")
    print(f"  Output dir  : {NUMERICAL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
