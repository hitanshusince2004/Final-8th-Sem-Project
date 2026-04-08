"""
Glacier Melting Detection — GEE Configuration
=============================================
Project  : Final Year Project
GCP ID   : final-year-project-485804
Exports  : All data downloads directly to LOCAL DISK (no Drive, no GCS)
"""

import os

# ── GCP / GEE Project ─────────────────────────────────────────────────────────
GEE_PROJECT = "final-year-project-485804"

# ── Local Output Directories ──────────────────────────────────────────────────
BASE_OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_dataset")
NUMERICAL_DIR       = os.path.join(BASE_OUTPUT_DIR, "numerical")
PATCHES_DIR         = os.path.join(BASE_OUTPUT_DIR, "patches")
FULL_COMPOSITES_DIR = os.path.join(BASE_OUTPUT_DIR, "full_composites")
PROCESSED_DIR       = os.path.join(BASE_OUTPUT_DIR, "processed")
LOGS_DIR            = os.path.join(BASE_OUTPUT_DIR, "logs")

for _d in [NUMERICAL_DIR, PATCHES_DIR, FULL_COMPOSITES_DIR, PROCESSED_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Area of Interest ──────────────────────────────────────────────────────────
# Exact Research ROI: Central & South Asia
AOI_COORDS = [
    [66.64933259644972, 40.42698565388099],
    [68.57657342678199, 32.63830714754258],
    [81.60720658357097, 27.069612897524202],
    [94.49620257494624, 24.551726708192586],
    [102.53949449696086, 30.434557243831026],
    [86.68369974510316, 47.51013816642729],
    [66.64933259644972, 40.42698565388099]
]

# ── Time Range ────────────────────────────────────────────────────────────────
START_YEAR = 2018
END_YEAR   = 2025
YEARS      = list(range(START_YEAR, END_YEAR + 1))

MELT_SEASON   = {"start": "05-01", "end": "09-30"}
WINTER_SEASON = {"start": "11-01", "end": "03-31"}

# ── Sentinel-2 ────────────────────────────────────────────────────────────────
S2_COLLECTION  = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_COVER = 20
S2_BANDS = {
    "B2": "Blue",  "B3": "Green", "B4": "Red",
    "B5": "RedEdge1", "B6": "RedEdge2", "B7": "RedEdge3",
    "B8": "NIR",   "B8A": "RedEdge4",
    "B11": "SWIR1","B12": "SWIR2",
}
S2_SCALE = 10

# ── Sentinel-1 SAR ────────────────────────────────────────────────────────────
S1_COLLECTION = "COPERNICUS/S1_GRD"
S1_BANDS      = ["VV", "VH"]
S1_MODE       = "IW"
S1_SCALE      = 10

# ── Landsat 8 / 9 ─────────────────────────────────────────────────────────────
L8_COLLECTION  = "LANDSAT/LC08/C02/T1_L2"
L9_COLLECTION  = "LANDSAT/LC09/C02/T1_L2"
LANDSAT_SCALE  = 30
LANDSAT_CLOUD  = 20

# ── DEM ───────────────────────────────────────────────────────────────────────
DEM_COLLECTION = "USGS/SRTMGL1_003"
DEM_SCALE      = 30

# ── Numerical sampling ────────────────────────────────────────────────────────
CSV_SCALE        = 30
SAMPLES_PER_YEAR = 1750      # per year/season -> ~28,000 total (1750 * 8 * 2)
BATCH_SIZE       = 5000      # max rows per GEE getInfo() call
SAMPLE_SEED      = 42

# ── Image patch export ────────────────────────────────────────────────────────
PATCH_SIZE       = 256
PATCH_SCALE      = 10
PATCHES_PER_YEAR = 350       # patches per year/season -> ~5,600 total (350 * 8 * 2)

# ── Index thresholds ──────────────────────────────────────────────────────────
NDSI_GLACIER_THRESHOLD = 0.40
NDWI_WATER_THRESHOLD   = 0.30
LST_MELT_THRESHOLD_C   = 0.0
