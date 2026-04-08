"""
Glacier Melting Detection — Image Dataset Generator
=====================================================
Project : Final Year Project  |  GEE ID : final-year-project-485804
Exports : *** DIRECTLY TO LOCAL DISK *** — no Google Drive, no GCS

Uses geemap.download_ee_image() to pull GeoTIFF patches directly
to local disk without needing Google Drive or Cloud Storage.

Image types (per year × season):
  RGB · Grayscale · glacier_mask · multi_mask
  NDSI · NDWI · NIR · SWIR · LST · DEM · SAR_VV · SAR_VH

Install requirement (one-time):
    pip install geemap

Run
---
    python scripts/generate_image_dataset.py                       # all years
    python scripts/generate_image_dataset.py --year 2022           # one year
    python scripts/generate_image_dataset.py --year 2022 --season melt --patches 50
"""

import ee
import os
import sys
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    GEE_PROJECT, AOI_COORDS, YEARS,
    MELT_SEASON, WINTER_SEASON,
    PATCHES_DIR, FULL_COMPOSITES_DIR,
    PATCH_SIZE, PATCH_SCALE, PATCHES_PER_YEAR, SAMPLE_SEED,
)
from utils.gee_utils import (
    build_s2_composite, build_s1_composite,
    build_landsat_composite, get_dem,
)


# ── Initialise GEE ────────────────────────────────────────────────────────────
def init_ee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ Earth Engine initialised  (project: {GEE_PROJECT})")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ Earth Engine initialised  (project: {GEE_PROJECT})")


# ── geemap import (with helpful error) ───────────────────────────────────────
def get_geemap():
    try:
        import geemap
        return geemap
    except ImportError:
        print("\n  ✗ geemap not installed.")
        print("  Run:  pip install geemap")
        print("  Then re-run this script.\n")
        sys.exit(1)


# ── Image type definitions ────────────────────────────────────────────────────
IMAGE_TYPES = {
    "rgb":           {"source": "s2",      "bands": ["B4","B3","B2"],         "scale": PATCH_SCALE},
    "grayscale":     {"source": "s2",      "bands": ["B3"],                   "scale": PATCH_SCALE},
    "glacier_mask":  {"source": "s2",      "bands": ["glacier_mask"],         "scale": PATCH_SCALE},
    "multi_mask":    {"source": "s2",      "bands": ["multiclass_mask"],      "scale": PATCH_SCALE},
    "ndsi":          {"source": "s2",      "bands": ["NDSI"],                 "scale": PATCH_SCALE},
    "ndwi":          {"source": "s2",      "bands": ["NDWI"],                 "scale": PATCH_SCALE},
    "nir":           {"source": "s2",      "bands": ["B8"],                   "scale": PATCH_SCALE},
    "swir":          {"source": "s2",      "bands": ["B11"],                  "scale": PATCH_SCALE},
    "lst":           {"source": "landsat", "bands": ["LST_Celsius"],          "scale": 30},
    "dem":           {"source": "dem",     "bands": ["elevation"],            "scale": 30},
    "sar_vv":        {"source": "s1",      "bands": ["VV"],                   "scale": PATCH_SCALE},
    "sar_vh":        {"source": "s1",      "bands": ["VH"],                   "scale": PATCH_SCALE},
}


# ── Build all composites for a year/season ────────────────────────────────────
def build_composites(aoi, year, season_dict):
    return {
        "s2":      build_s2_composite(aoi, year, season_dict),
        "s1":      build_s1_composite(aoi, year, season_dict),
        "landsat": build_landsat_composite(aoi, year, season_dict),
        "dem":     get_dem(aoi),
    }


# ── Download full-AOI composite ───────────────────────────────────────────────
def download_full_aoi(geemap, image, bands, aoi, out_path, scale=100):
    """Download a full-AOI composite at reduced resolution (100 m) to local GeoTIFF."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return  # skip already downloaded

    try:
        img = image.select(bands)
        geemap.download_ee_image(
            image    = img,
            filename = out_path,
            region   = aoi,
            scale    = scale,
            crs      = "EPSG:4326",
        )
        print(f"      ✓ {os.path.basename(out_path)}")
    except Exception as e:
        print(f"      ✗ {os.path.basename(out_path)} — {e}")


# ── Download patch stacks ─────────────────────────────────────────────────────
def download_patches(geemap, composites, aoi, year, season_name, n_patches):
    """
    Sample n_patches random centres from AOI,
    clip stacked multi-band image to each patch bounding box,
    download each patch as a GeoTIFF.
    """
    print(f"  Downloading {n_patches} patches to local disk…")

    out_dir = os.path.join(PATCHES_DIR, str(year), season_name)
    os.makedirs(out_dir, exist_ok=True)

    # Check how many already exist
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".tif")])
    if existing >= n_patches:
        print(f"  ⏩ {existing} patches already exist — skipping.")
        return existing

    # Build stacked patch image (11 input bands + 2 label bands + LST + elevation)
    s2  = composites["s2"]
    s1  = composites["s1"]
    ls  = composites["landsat"]
    dem = composites["dem"]

    patch_stack = (
        s2.select(["B4","B3","B2","B8","B11","NDSI","NDWI","glacier_mask","multiclass_mask"])
        .addBands(s1.select(["VV","VH"]))
        .addBands(ls.select(["LST_Celsius"]))
        .addBands(dem.select(["elevation"]))
    )

    # Random centres
    centres = ee.FeatureCollection.randomPoints(
        region = aoi,
        points = n_patches,
        seed   = SAMPLE_SEED + year,
    )
    centres_list = centres.toList(n_patches).getInfo()

    downloaded = 0
    for i, feat in enumerate(centres_list):
        patch_name = f"glacier_patch_{year}_{season_name}_{i:04d}.tif"
        out_path   = os.path.join(out_dir, patch_name)

        if os.path.exists(out_path):
            downloaded += 1
            continue

        coords = feat["geometry"]["coordinates"]
        lng, lat = coords[0], coords[1]

        # Buffer to square patch
        half   = (PATCH_SIZE * PATCH_SCALE) / 2
        region = ee.Geometry.BBox(lng - half/111320, lat - half/111320,
                                   lng + half/111320, lat + half/111320)

        try:
            geemap.download_ee_image(
                image    = patch_stack.clip(region),
                filename = out_path,
                region   = region,
                scale    = PATCH_SCALE,
                crs      = "EPSG:4326",
            )
            downloaded += 1
            if (i + 1) % 25 == 0 or i == 0:
                print(f"    [{i+1:4d}/{n_patches}] saved {patch_name}")
        except Exception as e:
            print(f"    ✗ patch {i:04d}: {e}")

        time.sleep(0.3)   # avoid hammering GEE

    return downloaded


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int, help="Single year (2018–2025)")
    parser.add_argument("--season",  type=str, choices=["melt","winter"])
    parser.add_argument("--patches", type=int, default=PATCHES_PER_YEAR,
                        help=f"Patches per year/season (default: {PATCHES_PER_YEAR})")
    parser.add_argument("--no_full_aoi", action="store_true",
                        help="Skip full-AOI composite downloads (faster)")
    args = parser.parse_args()

    init_ee()
    gm  = get_geemap()
    aoi = ee.Geometry.Polygon(AOI_COORDS)

    years   = [args.year]   if args.year   else YEARS
    seasons = {}
    if args.season == "melt":
        seasons = {"melt": MELT_SEASON}
    elif args.season == "winter":
        seasons = {"winter": WINTER_SEASON}
    else:
        seasons = {"melt": MELT_SEASON, "winter": WINTER_SEASON}

    total_patches = 0

    for year in years:
        for season_name, season_dict in seasons.items():
            print(f"\n{'═'*60}")
            print(f"  Year: {year}   Season: {season_name}")
            print(f"{'═'*60}")

            try:
                composites = build_composites(aoi, year, season_dict)

                # ── A) Full-AOI composites (one per image type) ───────────────
                if not args.no_full_aoi:
                    print(f"  Downloading full-AOI composites (12 types)…")
                    for img_type, cfg in IMAGE_TYPES.items():
                        comp     = composites[cfg["source"]]
                        out_path = os.path.join(
                            FULL_COMPOSITES_DIR, img_type,
                            f"glacier_{img_type}_{year}_{season_name}.tif"
                        )
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        download_full_aoi(gm, comp, cfg["bands"], aoi, out_path, scale=100)

                # ── B) Patch-level downloads ──────────────────────────────────
                n = download_patches(gm, composites, aoi, year, season_name, args.patches)
                total_patches += n

            except Exception as e:
                print(f"  ✗ Error: {e}")
                traceback.print_exc()

    print(f"\n{'═'*60}")
    print(f"  IMAGE DATASET COMPLETE")
    print(f"  Patches downloaded : {total_patches:,}")
    print(f"  Patches dir        : {PATCHES_DIR}")
    print(f"  Full composites    : {FULL_COMPOSITES_DIR}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
