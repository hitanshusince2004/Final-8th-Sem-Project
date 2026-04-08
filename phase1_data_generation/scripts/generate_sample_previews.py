"""
Glacier Melting Detection — Sample Image Preview Generator
===========================================================
Generates realistic-looking SAMPLE PNG/JPG files for all 14 image types
using NumPy + Pillow. No GEE connection needed.

Use this to:
  1. See exactly what each image type looks like BEFORE running GEE
  2. Test downstream ML pipeline with realistic images
  3. Create demo / presentation screenshots

Run:
    python scripts/generate_sample_previews.py

Output:  local_dataset/sample_previews/
    01_RGB.jpg
    02_False_Color.jpg
    03_Grayscale.jpg
    04_NDSI.png
    05_NDWI.png
    06_NDVI.png
    07_Glacier_Mask.png
    08_Multiclass_Mask.png
    09_LST.png
    10_DEM.png
    11_SAR_VV.png
    12_SAR_VH.png
    13_SWIR.png
    14_NIR.png
    00_ALL_TYPES_OVERVIEW.png   ← grid of all 14 images
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE    = 512     # pixels per image
SEED    = 42

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "local_dataset", "sample_previews"
)
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# TERRAIN / GLACIER BASE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def perlin_noise(shape, scale=8, octaves=6, rng=None):
    """Simple multi-octave smooth noise for terrain simulation."""
    if rng is None:
        rng = np.random.default_rng(42)
    H, W   = shape
    result = np.zeros(shape)
    amp    = 1.0
    freq   = 1.0 / scale
    for _ in range(octaves):
        grid_h = max(2, int(H * freq) + 1)
        grid_w = max(2, int(W * freq) + 1)
        grid   = rng.random((grid_h, grid_w)).astype(np.float32)
        from PIL import Image as PIL_Image
        img    = PIL_Image.fromarray((grid * 255).astype(np.uint8))
        img    = img.resize((W, H), PIL_Image.BILINEAR)
        result += np.array(img, dtype=np.float32) / 255.0 * amp
        amp   *= 0.5
        freq  *= 2.0
    return (result - result.min()) / (result.max() - result.min() + 1e-8)


def generate_base_terrain(size=SIZE):
    """Generate elevation map, glacier mask, water mask."""
    elev    = perlin_noise((size, size), scale=size//4, octaves=7, rng=rng)

    # Ridge + valley structure (mountain terrain)
    ridge   = np.abs(perlin_noise((size, size), scale=size//2, octaves=4, rng=rng) - 0.5) * 2
    elev    = elev * 0.6 + ridge * 0.4

    # High-elevation = glacier (top 35% of terrain)
    glacier_thresh = np.percentile(elev, 65)
    glacier = elev > glacier_thresh

    # Add fractal edges to glacier boundary
    noise_edge = perlin_noise((size, size), scale=size//8, octaves=3, rng=rng)
    glacier_mask = (elev + noise_edge * 0.12) > glacier_thresh

    # Low-lying areas with NDWI > 0.3 → water / meltwater lakes
    water_mask = (elev < 0.25) & (noise_edge > 0.55) & (~glacier_mask)

    # Debris-covered ice: transition zone between glacier & rock
    debris_dist = np.abs(elev - glacier_thresh)
    debris_mask = (debris_dist < 0.08) & (~glacier_mask) & (~water_mask)

    return elev, glacier_mask, water_mask, debris_mask


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR MAPPING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_colormap(data, palette_hex, vmin=0.0, vmax=1.0):
    """Map float array [vmin, vmax] to RGB using a hex palette."""
    def hex2rgb(h):
        h = h.lstrip("#")
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

    colors = np.array([hex2rgb(h) for h in palette_hex], dtype=np.float32)  # (N,3)
    n      = len(colors) - 1
    norm   = np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1)
    idx    = norm * n
    lo     = np.floor(idx).astype(int).clip(0, n - 1)
    hi     = (lo + 1).clip(0, n)
    frac   = (idx - lo)[..., np.newaxis]                  # (..., 1)

    rgb = (1 - frac) * colors[lo] + frac * colors[hi]    # (..., 3)
    return rgb.clip(0, 255).astype(np.uint8)


def add_atmosphere_haze(rgb_array, strength=0.08):
    """Subtle bluish haze over high-elevation areas for realism."""
    haze  = np.array([180, 210, 240], dtype=np.float32)
    alpha = np.ones(rgb_array.shape[:2], dtype=np.float32) * strength
    alpha = alpha[..., np.newaxis]
    return (rgb_array * (1 - alpha) + haze * alpha).clip(0,255).astype(np.uint8)


def add_shading(array, elev, strength=0.25):
    """Simple hillshading based on gradient of elevation."""
    gy, gx = np.gradient(elev)
    shade  = 1.0 - np.clip(gx * strength + gy * strength, -0.4, 0.4)
    return (array * shade[..., np.newaxis]).clip(0, 255).astype(np.uint8)


def pil_save(arr, path, fmt="JPEG"):
    img = Image.fromarray(arr)
    if fmt == "JPEG":
        img = img.convert("RGB")
        img.save(path, "JPEG", quality=95, optimize=True)
    else:
        img.save(path, "PNG", optimize=True)
    kb = os.path.getsize(path) // 1024
    print(f"  ✓  {os.path.basename(path):<45}  {kb:5d} KB")


def add_label_bar(img_arr, label, color=(255,255,255), bg=(20,20,18)):
    """Add a small label bar at the bottom of the image."""
    try:
        img  = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        H, W = img_arr.shape[:2]
        draw.rectangle([0, H-28, W, H], fill=bg)
        draw.text((10, H-22), label, fill=color, font=font)
        return np.array(img)
    except Exception:
        return img_arr


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL IMAGE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def gen_rgb(elev, glacier, water, debris):
    """True-colour satellite composite — snow white, ice blue-white, rock brown, water teal."""
    rgb = np.zeros((SIZE, SIZE, 3), dtype=np.float32)

    # Rock / bare ground: brownish-grey
    rock_color  = np.array([110, 90, 72], dtype=np.float32)
    rgb += rock_color

    # Glacier / snow: white with slight blue tint
    snow_color  = np.array([235, 240, 248], dtype=np.float32)
    rgb[glacier] = snow_color

    # Ice with crevasses: slightly blue-grey
    crev = perlin_noise((SIZE,SIZE), scale=30, octaves=3, rng=rng) > 0.65
    ice_color   = np.array([195, 215, 230], dtype=np.float32)
    rgb[glacier & crev] = ice_color

    # Meltwater / proglacial lakes: teal-blue (like the uploaded images!)
    water_color = np.array([60, 160, 170], dtype=np.float32)
    rgb[water]  = water_color

    # Deep meltwater: darker blue
    deep_water  = water & (elev < 0.15)
    rgb[deep_water] = np.array([35, 110, 140], dtype=np.float32)

    # Debris ice: brownish-white mix
    debris_color = np.array([155, 140, 115], dtype=np.float32)
    rgb[debris] = debris_color

    # Shadow in valleys
    gy, gx = np.gradient(elev)
    shadow  = np.clip(1 + gy * 0.5 + gx * 0.3, 0.55, 1.0)[..., np.newaxis]
    rgb     = (rgb * shadow).clip(0, 255).astype(np.uint8)

    # Subtle haze
    rgb = add_atmosphere_haze(rgb, strength=0.06)

    # Light sharpening
    img = Image.fromarray(rgb)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=40, threshold=2))
    return np.array(img)


def gen_false_color(elev, glacier, water, debris):
    """NIR false-colour: ice/snow=white, water=dark blue, rock=brown-red, vegetation=red."""
    rgb = np.zeros((SIZE, SIZE, 3), dtype=np.float32)

    # Rock (medium NIR reflectance): brownish-red
    rgb += np.array([140, 80, 60], dtype=np.float32)

    # Snow/ice: very high NIR → bright white
    rgb[glacier] = np.array([245, 245, 240], dtype=np.float32)

    # Water: absorbs NIR → very dark blue
    rgb[water]   = np.array([20, 40, 80], dtype=np.float32)

    # Debris: lower NIR, brownish
    rgb[debris]  = np.array([160, 100, 70], dtype=np.float32)

    # Shade
    rgb = add_shading(rgb, elev, strength=0.3)
    rgb = add_atmosphere_haze(rgb, strength=0.04)

    img = Image.fromarray(rgb.astype(np.uint8))
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=30))
    return np.array(img)


def gen_grayscale(elev, glacier, water, debris):
    """Panchromatic grayscale."""
    rgb  = gen_rgb(elev, glacier, water, debris)
    gray = (0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def gen_ndsi(elev, glacier, water):
    """NDSI: high (white-blue) = snow/ice, low (dark) = bare land."""
    ndsi_val         = np.full((SIZE, SIZE), -0.1)
    ndsi_val[glacier]= 0.4 + rng.random(glacier.sum()) * 0.55
    ndsi_val[water]  = rng.uniform(-0.3, 0.1, water.sum())
    noise = perlin_noise((SIZE, SIZE), scale=40, octaves=4, rng=rng) * 0.15
    ndsi_val = (ndsi_val + noise).clip(-1, 1)
    pal = ["#001f4d","#003d99","#0066ff","#66b3ff","#cce5ff","#ffffff"]
    rgb = apply_colormap(ndsi_val, pal, vmin=-0.3, vmax=1.0)
    return rgb


def gen_ndwi(elev, glacier, water):
    """NDWI: high (blue) = water, negative (brown) = land."""
    ndwi_val          = np.full((SIZE, SIZE), -0.3)
    ndwi_val[water]   = 0.3 + rng.random(water.sum()) * 0.4
    ndwi_val[glacier] = -0.1 + rng.random(glacier.sum()) * 0.15
    noise = perlin_noise((SIZE, SIZE), scale=50, octaves=3, rng=rng) * 0.12
    ndwi_val = (ndwi_val + noise).clip(-1, 1)
    pal = ["#7f3300","#cc6600","#ffcc00","#66ccff","#0066cc","#003380"]
    return apply_colormap(ndwi_val, pal, vmin=-0.6, vmax=0.7)


def gen_ndvi(elev, glacier, water):
    """NDVI: high (green) = vegetation, negative = snow/rock."""
    ndvi_val          = np.full((SIZE, SIZE), 0.15)
    ndvi_val[glacier] = -0.1 + rng.random(glacier.sum()) * 0.05
    ndvi_val[water]   = -0.2 + rng.random(water.sum()) * 0.1
    # Low-elevation slopes have some vegetation
    veg_zone = (elev < 0.5) & (~glacier) & (~water)
    ndvi_val[veg_zone] = 0.2 + rng.random(veg_zone.sum()) * 0.35
    noise = perlin_noise((SIZE, SIZE), scale=40, octaves=3, rng=rng) * 0.08
    ndvi_val = (ndvi_val + noise).clip(-1, 1)
    pal = ["#7f3300","#cc8800","#ffff00","#66cc00","#009900","#003300"]
    return apply_colormap(ndvi_val, pal, vmin=-0.3, vmax=0.6)


def gen_glacier_mask(glacier):
    """Binary glacier mask: teal = glacier, dark = non-glacier."""
    rgb = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    rgb[:,:] = [44, 44, 42]       # background dark
    rgb[glacier] = [29, 158, 117] # glacier teal (#1D9E75)
    return rgb


def gen_multiclass_mask(glacier, water, debris):
    """4-class segmentation: land=grey, snow/ice=light blue, water=blue, debris=amber."""
    rgb = np.full((SIZE, SIZE, 3), [136, 135, 128], dtype=np.uint8)  # land
    rgb[glacier] = [181, 212, 244]  # snow/ice (#B5D4F4)
    rgb[water]   = [59, 139, 212]   # water    (#3B8BD4)
    rgb[debris]  = [250, 199, 117]  # debris   (#FAC775)
    return rgb


def gen_lst(elev, glacier, water):
    """LST: cold=blue (glacier), warm=red (valleys)."""
    lst = 15 - elev * 40           # high elev = very cold
    lst[glacier] -= 10             # glacier is coldest
    lst[water]   += 3              # water slightly warmer than ice
    noise = perlin_noise((SIZE,SIZE), scale=60, octaves=3, rng=rng) * 5
    lst   = (lst + noise)
    pal   = ["#2c7bb6","#abd9e9","#ffffbf","#fdae61","#d7191c","#7b0000"]
    return apply_colormap(lst, pal, vmin=-30, vmax=20)


def gen_dem(elev):
    """DEM: green lowlands → white high peaks."""
    pal = ["#004d00","#339900","#99cc33","#cccc00","#996600","#d4b483","#ffffff"]
    rgb = apply_colormap(elev, pal, vmin=0, vmax=1)
    # Hillshade
    gy, gx  = np.gradient(elev)
    shade   = np.clip(1 - gx*0.5 - gy*0.3, 0.5, 1.0)[..., np.newaxis]
    return (rgb * shade).clip(0,255).astype(np.uint8)


def gen_sar_vv(elev, glacier, water):
    """SAR VV backscatter: bright = rough surface, dark = smooth ice/water."""
    sar = 0.4 + elev * 0.3
    sar[glacier] *= 0.5            # smooth ice = low backscatter
    sar[water]   *= 0.2            # water = very low (specular reflection)
    noise = perlin_noise((SIZE,SIZE), scale=20, octaves=5, rng=rng) * 0.3
    sar   = (sar + noise).clip(0, 1)
    pal   = ["#000000","#404040","#808080","#c0c0c0","#ffffff"]
    return apply_colormap(sar, pal, vmin=0, vmax=1)


def gen_sar_vh(elev, glacier, water):
    """SAR VH cross-polarisation: highlights volume scatterers."""
    sar = 0.2 + elev * 0.25
    sar[glacier]  *= 0.6
    sar[water]    *= 0.1
    rough  = perlin_noise((SIZE,SIZE), scale=15, octaves=6, rng=rng) * 0.35
    sar    = (sar + rough).clip(0, 1)
    pal    = ["#000000","#303060","#606090","#9090c0","#c0c0ff"]
    return apply_colormap(sar, pal, vmin=0, vmax=1)


def gen_swir(elev, glacier, water):
    """SWIR: ice/snow absorbs SWIR → dark; bare rock bright."""
    swir = 0.3 + elev * 0.2
    swir[glacier] = 0.02 + rng.random(glacier.sum()) * 0.08  # ice absorbs SWIR
    swir[water]   = 0.01 + rng.random(water.sum()) * 0.05
    noise = perlin_noise((SIZE,SIZE), scale=50, octaves=4, rng=rng) * 0.1
    swir  = (swir + noise).clip(0, 1)
    pal   = ["#000033","#003366","#0066cc","#66b3ff","#cce0ff","#ffffff"]
    return apply_colormap(swir, pal, vmin=0, vmax=0.6)


def gen_nir(elev, glacier, water):
    """NIR: ice/snow very bright, water dark, vegetation moderate."""
    nir = 0.25 + elev * 0.15
    nir[glacier]  = 0.55 + rng.random(glacier.sum()) * 0.35
    nir[water]    = 0.01 + rng.random(water.sum()) * 0.05
    noise = perlin_noise((SIZE,SIZE), scale=45, octaves=4, rng=rng) * 0.08
    nir   = (nir + noise).clip(0, 1)
    pal   = ["#000000","#003300","#006600","#33cc33","#99ff99","#ffffff"]
    return apply_colormap(nir, pal, vmin=0, vmax=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW GRID
# ══════════════════════════════════════════════════════════════════════════════

def make_overview_grid(images_dict, out_path):
    """
    Combine all image arrays into one large annotated grid PNG.
    Layout: 4 columns × 4 rows (14 images + 2 placeholders)
    """
    COLS, ROWS = 4, 4
    THUMB      = 300
    LABEL_H    = 32
    PAD        = 4
    BG         = (18, 18, 16)

    W = COLS * (THUMB + PAD) + PAD
    H = ROWS * (THUMB + LABEL_H + PAD) + PAD + 44  # 44 = title bar

    sheet = Image.new("RGB", (W, H), BG)
    draw  = ImageDraw.Draw(sheet)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font_title = font_label = ImageFont.load_default()

    # Title
    draw.text((10, 10), "Glacier Melting Detection — All 14 Image Types", fill=(210,210,205), font=font_title)

    items = list(images_dict.items())

    for i, (name, arr) in enumerate(items):
        col = i % COLS
        row = i // COLS
        x   = PAD + col * (THUMB + PAD)
        y   = 44 + PAD + row * (THUMB + LABEL_H + PAD)

        thumb = Image.fromarray(arr).resize((THUMB, THUMB), Image.LANCZOS)
        sheet.paste(thumb, (x, y))

        # Label bar
        draw.rectangle([x, y+THUMB, x+THUMB, y+THUMB+LABEL_H], fill=(28,28,26))
        draw.text((x+6, y+THUMB+8), name, fill=(200,200,195), font=font_label)

    sheet.save(out_path, "PNG", optimize=True)
    kb = os.path.getsize(out_path) // 1024
    print(f"\n  ✓  {'00_ALL_TYPES_OVERVIEW.png':<45}  {kb:5d} KB  ← Grid of all 14 types")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Glacier Melting Detection — Sample Preview Generator")
    print(f"Output: {OUT_DIR}\n")

    # Generate base terrain
    print("  Generating realistic terrain…")
    elev, glacier, water, debris = generate_base_terrain(SIZE)

    # Generate all image types
    IMAGES = {
        "01_RGB":             (gen_rgb(elev, glacier, water, debris),        "JPEG"),
        "02_False_Color":     (gen_false_color(elev, glacier, water, debris),"JPEG"),
        "03_Grayscale":       (gen_grayscale(elev, glacier, water, debris),  "JPEG"),
        "04_NDSI":            (gen_ndsi(elev, glacier, water),               "PNG"),
        "05_NDWI":            (gen_ndwi(elev, glacier, water),               "PNG"),
        "06_NDVI":            (gen_ndvi(elev, glacier, water),               "PNG"),
        "07_Glacier_Mask":    (gen_glacier_mask(glacier),                    "PNG"),
        "08_Multiclass_Mask": (gen_multiclass_mask(glacier, water, debris),  "PNG"),
        "09_LST":             (gen_lst(elev, glacier, water),                "PNG"),
        "10_DEM":             (gen_dem(elev),                                "PNG"),
        "11_SAR_VV":          (gen_sar_vv(elev, glacier, water),             "PNG"),
        "12_SAR_VH":          (gen_sar_vh(elev, glacier, water),             "PNG"),
        "13_SWIR":            (gen_swir(elev, glacier, water),               "PNG"),
        "14_NIR":             (gen_nir(elev, glacier, water),                "PNG"),
    }

    # Save each image
    print(f"\n  Saving images to {OUT_DIR}/\n")
    ext_map = {"JPEG": ".jpg", "PNG": ".png"}
    saved   = {}

    for name, (arr, fmt) in IMAGES.items():
        labeled = add_label_bar(arr, name, color=(255,255,255), bg=(18,18,16))
        ext     = ext_map[fmt]
        path    = os.path.join(OUT_DIR, name + ext)
        pil_save(labeled, path, fmt=fmt)
        saved[name] = labeled

    # Overview grid
    grid_path = os.path.join(OUT_DIR, "00_ALL_TYPES_OVERVIEW.png")
    make_overview_grid(saved, grid_path)

    print(f"\n{'═'*65}")
    print(f"  DONE — {len(IMAGES)} images saved")
    print(f"  Location: {OUT_DIR}")
    print(f"\n  To use with GEE for real satellite data, run:")
    print(f"  python scripts/export_images_png_jpg.py --year 2022 --season melt")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
