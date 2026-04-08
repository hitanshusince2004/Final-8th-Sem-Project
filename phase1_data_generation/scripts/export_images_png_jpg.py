"""
Glacier Melting Detection — PNG/JPG Image Exporter
====================================================
Project : Final Year Project  |  GEE ID : final-year-project-485804

Downloads ALL 12 glacier image types directly from GEE and saves them
as beautiful colourised PNG / JPG files (like real satellite imagery).

Image Types Generated:
  01_RGB.jpg            → True-colour satellite (like Google Earth)
  02_False_Color.jpg    → NIR false-colour (vegetation red, ice white)
  03_Grayscale.jpg      → Panchromatic intensity
  04_NDSI.png           → Snow index (blue→white scale)
  05_NDWI.png           → Water index (blue scale)
  06_NDVI.png           → Vegetation index (green scale)
  07_Glacier_Mask.png   → Binary mask (teal=glacier, dark=land)
  08_Multiclass_Mask.png→ 4-class (white=snow, blue=water, orange=debris)
  09_LST.png            → Temperature (blue=cold → red=hot)
  10_DEM.png            → Elevation (green→white mountain scale)
  11_SAR_VV.png         → Radar backscatter (grayscale)
  12_SAR_VH.png         → Radar VH polarisation
  13_SWIR.png           → Shortwave infrared
  14_NIR.png            → Near infrared

Run:
    cd phase1_data_generation
    python scripts/export_images_png_jpg.py

    # Single year:
    python scripts/export_images_png_jpg.py --year 2022 --season melt

    # Specific area (zoom in):
    python scripts/export_images_png_jpg.py --year 2023 --lat 34.5 --lng 76.5 --size 1.0

Output:
    local_dataset/images_png_jpg/
        2022/
            melt/
                01_RGB.jpg
                02_False_Color.jpg
                03_Grayscale.jpg
                04_NDSI.png
                ... (14 images per year/season)
        2023/
            melt/
                01_RGB.jpg
                ...
"""

import ee
import os
import sys
import io
import math
import argparse
import numpy as np
import traceback

# PIL for saving PNG/JPG
try:
    from PIL import Image
    import PIL
except ImportError:
    print("Installing Pillow…")
    os.system("pip install Pillow -q")
    from PIL import Image

# requests for downloading GEE thumbnail URLs
try:
    import requests
except ImportError:
    os.system("pip install requests -q")
    import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    GEE_PROJECT, AOI_COORDS, YEARS,
    MELT_SEASON, WINTER_SEASON,
)

# ── Output directory ──────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "local_dataset", "images_png_jpg"
)

# ── Image size (pixels) ───────────────────────────────────────────────────────
IMG_WIDTH  = 1024
IMG_HEIGHT = 1024

# ── Colour palettes for index images ─────────────────────────────────────────
PALETTES = {
    "NDSI":       ["#001f4d", "#003d99", "#0066ff", "#66b3ff", "#cce5ff", "#ffffff"],
    "NDWI":       ["#7f3300", "#cc6600", "#ffcc00", "#66ccff", "#0066cc", "#003380"],
    "NDVI":       ["#7f3300", "#cc8800", "#ffff00", "#66cc00", "#009900", "#003300"],
    "LST":        ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c", "#7b0000"],
    "DEM":        ["#004d00", "#339900", "#99cc33", "#cccc00", "#996600", "#ffffff"],
    "SAR_VV":     ["#000000", "#404040", "#808080", "#c0c0c0", "#ffffff"],
    "SAR_VH":     ["#000000", "#303060", "#606090", "#9090c0", "#c0c0ff"],
    "SWIR":       ["#000033", "#003366", "#0066cc", "#66b3ff", "#ffffff"],
    "NIR":        ["#000000", "#003300", "#006600", "#33cc33", "#99ff99", "#ffffff"],
    "glacier_mask":   ["#2C2C2A", "#1D9E75"],
    "multiclass_mask":["#888780", "#B5D4F4", "#3B8BD4", "#FAC775"],
    "false_color":    ["#000000", "#003300", "#336600", "#cc0000", "#ff6666", "#ffffff"],
}

# ── Visualisation parameters ──────────────────────────────────────────────────
VIS_PARAMS = {
    "RGB":            {"bands": ["B4","B3","B2"],      "min": 0.0,  "max": 0.30,  "gamma": 1.4},
    "False_Color":    {"bands": ["B8","B4","B3"],      "min": 0.0,  "max": 0.40,  "gamma": 1.2},
    "Grayscale":      {"bands": ["B3"],                "min": 0.0,  "max": 0.30,  "palette": ["#000000","#808080","#ffffff"]},
    "NDSI":           {"bands": ["NDSI"],              "min": -0.2, "max": 1.0,   "palette": PALETTES["NDSI"]},
    "NDWI":           {"bands": ["NDWI"],              "min": -0.5, "max": 0.5,   "palette": PALETTES["NDWI"]},
    "NDVI":           {"bands": ["NDVI"],              "min": -0.2, "max": 0.6,   "palette": PALETTES["NDVI"]},
    "Glacier_Mask":   {"bands": ["glacier_mask"],      "min": 0,    "max": 1,     "palette": PALETTES["glacier_mask"]},
    "Multiclass_Mask":{"bands": ["multiclass_mask"],   "min": 0,    "max": 3,     "palette": PALETTES["multiclass_mask"]},
    "LST":            {"bands": ["LST_Celsius"],       "min": -25,  "max": 15,    "palette": PALETTES["LST"]},
    "DEM":            {"bands": ["elevation"],         "min": 1000, "max": 8000,  "palette": PALETTES["DEM"]},
    "SAR_VV":         {"bands": ["VV"],                "min": -25,  "max": 0,     "palette": PALETTES["SAR_VV"]},
    "SAR_VH":         {"bands": ["VH"],                "min": -30,  "max": -5,    "palette": PALETTES["SAR_VH"]},
    "SWIR":           {"bands": ["B11"],               "min": 0.0,  "max": 0.5,   "palette": PALETTES["SWIR"]},
    "NIR":            {"bands": ["B8"],                "min": 0.0,  "max": 0.6,   "palette": PALETTES["NIR"]},
}

FILE_FORMAT = {
    "RGB":             ("01_RGB.jpg",             "JPEG"),
    "False_Color":     ("02_False_Color.jpg",     "JPEG"),
    "Grayscale":       ("03_Grayscale.jpg",       "JPEG"),
    "NDSI":            ("04_NDSI.png",            "PNG"),
    "NDWI":            ("05_NDWI.png",            "PNG"),
    "NDVI":            ("06_NDVI.png",            "PNG"),
    "Glacier_Mask":    ("07_Glacier_Mask.png",    "PNG"),
    "Multiclass_Mask": ("08_Multiclass_Mask.png", "PNG"),
    "LST":             ("09_LST.png",             "PNG"),
    "DEM":             ("10_DEM.png",             "PNG"),
    "SAR_VV":          ("11_SAR_VV.png",          "PNG"),
    "SAR_VH":          ("12_SAR_VH.png",          "PNG"),
    "SWIR":            ("13_SWIR.png",            "PNG"),
    "NIR":             ("14_NIR.png",             "PNG"),
}


# ══════════════════════════════════════════════════════════════════════════════
# GEE INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_ee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ Earth Engine ready  (project: {GEE_PROJECT})")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ Earth Engine ready  (project: {GEE_PROJECT})")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD COMPOSITES
# ══════════════════════════════════════════════════════════════════════════════

def mask_s2(img):
    scl  = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).divide(10000).copyProperties(img, ["system:time_start"])

def mask_landsat(img):
    qa   = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    opt  = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    thm  = img.select("ST_B10").multiply(0.00341802).add(149.0)
    return img.addBands(opt, None, True).addBands(thm, None, True).updateMask(mask)

def add_indices(img):
    ndsi = img.normalizedDifference(["B3","B11"]).rename("NDSI")
    ndwi = img.normalizedDifference(["B3","B8"]).rename("NDWI")
    ndvi = img.normalizedDifference(["B8","B4"]).rename("NDVI")
    return img.addBands([ndsi, ndwi, ndvi])

def add_masks(img):
    glacier = (img.select("NDSI").gt(0.40)
               .And(img.select("NDWI").lt(0.30))).rename("glacier_mask")
    mc = (ee.Image(0)
          .where(img.select("NDSI").gt(0.40), 1)
          .where(img.select("NDWI").gt(0.30).And(img.select("NDSI").lte(0.40)), 2)
          .where(img.select("NDSI").gt(0.10).And(img.select("NDSI").lte(0.40)), 3)
          .rename("multiclass_mask"))
    return img.addBands([glacier, mc])

def build_full_composite(aoi, year, season_dict):
    """Build a single merged image with ALL bands needed for all visualisations."""
    start = f"{year}-{season_dict['start']}"
    end   = f"{year}-{season_dict['end']}"

    # Sentinel-2
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .map(mask_s2).map(add_indices).map(add_masks)
          .median().clip(aoi))

    # Sentinel-1 SAR
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
          .median().clip(aoi))

    # Landsat LST
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUD_COVER", 20))
          .map(mask_landsat))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUD_COVER", 20))
          .map(mask_landsat))
    ls_lst = l8.merge(l9).map(
        lambda img: img.select("ST_B10").subtract(273.15).rename("LST_Celsius")
    ).median().clip(aoi)

    # DEM
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi).rename("elevation")

    # Merge everything
    return (s2.addBands(s1.select(["VV","VH"]))
              .addBands(ls_lst)
              .addBands(dem))


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD IMAGE AS PNG/JPG via GEE getThumbURL
# ══════════════════════════════════════════════════════════════════════════════

def download_as_image(ee_image, vis_params, region, out_path, fmt="JPEG",
                      width=IMG_WIDTH, height=IMG_HEIGHT):
    """
    Uses ee.Image.getThumbURL() to get a direct PNG/JPG download link from GEE,
    then saves it locally. No Drive, no GCS needed.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"    ⏩ Already exists: {os.path.basename(out_path)}")
        return True

    # Build thumb params
    thumb_params = {
        "bands":   vis_params["bands"],
        "min":     vis_params["min"],
        "max":     vis_params["max"],
        "region":  region,
        "dimensions": f"{width}x{height}",
        "format":  "jpg" if fmt == "JPEG" else "png",
    }
    if "palette" in vis_params:
        thumb_params["palette"] = vis_params["palette"]
    if "gamma" in vis_params:
        thumb_params["gamma"] = vis_params["gamma"]

    try:
        url  = ee_image.getThumbURL(thumb_params)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        img  = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img  = img.resize((width, height), Image.LANCZOS)

        if fmt == "JPEG":
            img.save(out_path, "JPEG", quality=95, optimize=True)
        else:
            img.save(out_path, "PNG", optimize=True)

        size_kb = os.path.getsize(out_path) // 1024
        print(f"    ✓ {os.path.basename(out_path):40s}  {size_kb:5d} KB")
        return True

    except Exception as e:
        print(f"    ✗ {os.path.basename(out_path)}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE LEGEND IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def save_legend(image_type, vis_params, out_path):
    """Create a small legend PNG showing the colour scale for index images."""
    try:
        W, H = 400, 60
        img  = Image.new("RGB", (W, H), (30, 30, 28))
        pix  = img.load()

        palette = vis_params.get("palette", [])
        if not palette or len(vis_params["bands"]) > 1:
            return   # no legend for RGB

        # Parse hex colours
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        colors = [hex_to_rgb(c) for c in palette]
        n = len(colors) - 1

        bar_y0, bar_y1 = 10, 40
        for x in range(W):
            t   = x / (W - 1)
            idx = t * n
            lo  = int(idx)
            hi  = min(lo + 1, n)
            f   = idx - lo
            r   = int(colors[lo][0] * (1-f) + colors[hi][0] * f)
            g   = int(colors[lo][1] * (1-f) + colors[hi][1] * f)
            b   = int(colors[lo][2] * (1-f) + colors[hi][2] * f)
            for y in range(bar_y0, bar_y1):
                pix[x, y] = (r, g, b)

        img.save(out_path, "PNG")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE A CONTACT SHEET (all 14 images in one grid PNG)
# ══════════════════════════════════════════════════════════════════════════════

def make_contact_sheet(image_dir, out_path):
    """Combine all 14 images into one 4×4 overview grid PNG."""
    try:
        files = sorted([f for f in os.listdir(image_dir)
                        if f.endswith((".jpg",".png")) and not f.startswith("contact")])
        if not files:
            return

        cols, rows = 4, math.ceil(len(files) / 4)
        THUMB = 300
        LABEL_H = 26
        W_total = cols * THUMB
        H_total = rows * (THUMB + LABEL_H)

        sheet = Image.new("RGB", (W_total, H_total), (20, 20, 18))

        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default()
        except Exception:
            draw = None

        for i, fname in enumerate(files):
            col = i % cols
            row = i // cols
            x   = col * THUMB
            y   = row * (THUMB + LABEL_H)

            try:
                img  = Image.open(os.path.join(image_dir, fname)).convert("RGB")
                img  = img.resize((THUMB, THUMB), Image.LANCZOS)
                sheet.paste(img, (x, y))
            except Exception:
                pass

            if draw:
                label = fname.replace(".jpg","").replace(".png","")
                draw.rectangle([x, y+THUMB, x+THUMB, y+THUMB+LABEL_H], fill=(30,30,28))
                draw.text((x+6, y+THUMB+5), label, fill=(200,200,195), font=font)

        sheet.save(out_path, "PNG", optimize=True)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"\n  ✓ Contact sheet saved: {os.path.basename(out_path)}  ({size_kb} KB)")
    except Exception as e:
        print(f"  ✗ Contact sheet error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXPORT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def export_year_season(year, season_name, season_dict, region, dims):
    """Download all 14 image types for one year/season."""
    print(f"\n{'═'*65}")
    print(f"  Year: {year}   Season: {season_name.upper()}")
    print(f"{'═'*65}")

    out_dir = os.path.join(BASE_OUTPUT_DIR, str(year), season_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Building composite…")
    try:
        composite = build_full_composite(region, year, season_dict)
    except Exception as e:
        print(f"  ✗ Composite failed: {e}")
        return 0

    print(f"  Downloading images as PNG/JPG…")
    success = 0

    for img_type, vis in VIS_PARAMS.items():
        fname, fmt = FILE_FORMAT[img_type]
        out_path   = os.path.join(out_dir, fname)

        # Select the right bands from composite
        try:
            bands = vis["bands"]
            img   = composite.select(bands)
            ok    = download_as_image(img, vis, region, out_path, fmt,
                                      width=dims, height=dims)
            if ok:
                success += 1
        except Exception as e:
            print(f"    ✗ {fname}: {e}")

    # Contact sheet
    sheet_path = os.path.join(out_dir, f"00_contact_sheet_{year}_{season_name}.png")
    make_contact_sheet(out_dir, sheet_path)

    print(f"\n  Downloaded {success}/{len(VIS_PARAMS)} images")
    print(f"  Saved to: {out_dir}")
    return success


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export all glacier image types as PNG/JPG"
    )
    parser.add_argument("--year",   type=int,   help="Single year (2018–2025)")
    parser.add_argument("--season", type=str,   choices=["melt","winter"])
    parser.add_argument("--lat",    type=float, default=34.5,  help="Centre latitude")
    parser.add_argument("--lng",    type=float, default=75.5,  help="Centre longitude")
    parser.add_argument("--size",   type=float, default=None,
                        help="Degree-side of square AOI (e.g. 2.0 = 2°×2°). Default=full AOI")
    parser.add_argument("--dims",   type=int,   default=1024,
                        help="Output image size in pixels (default: 1024)")
    args = parser.parse_args()

    init_ee()

    # Build region
    if args.size:
        half   = args.size / 2
        region = ee.Geometry.BBox(
            args.lng - half, args.lat - half,
            args.lng + half, args.lat + half,
        )
        print(f"  Custom AOI: {args.lat}°N, {args.lng}°E  ±{half}°")
    else:
        region = ee.Geometry.Polygon(AOI_COORDS)
        print(f"  Full AOI: Hindu Kush → Himalayas")

    years   = [args.year]   if args.year   else YEARS
    seasons = {}
    if args.season == "melt":
        seasons = {"melt": MELT_SEASON}
    elif args.season == "winter":
        seasons = {"winter": WINTER_SEASON}
    else:
        seasons = {"melt": MELT_SEASON, "winter": WINTER_SEASON}

    total = 0
    for year in years:
        for season_name, season_dict in seasons.items():
            n     = export_year_season(year, season_name, season_dict, region, args.dims)
            total += n

    print(f"\n{'═'*65}")
    print(f"  ALL DONE")
    print(f"  Total images saved : {total}")
    print(f"  Output directory   : {BASE_OUTPUT_DIR}")
    print(f"\n  Folder structure:")
    for year in years:
        for season_name in seasons:
            d = os.path.join(BASE_OUTPUT_DIR, str(year), season_name)
            if os.path.isdir(d):
                n_files = len([f for f in os.listdir(d) if f.endswith((".jpg",".png"))])
                print(f"    {d}   ({n_files} images)")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
