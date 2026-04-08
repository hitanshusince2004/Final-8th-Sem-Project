import os
import glob
import numpy as np
import rasterio
from PIL import Image
from pathlib import Path

def convert_tif_to_png(tif_path, output_dir):
    """Convert a 13-band Sentinel-2 GeoTIFF patch to RGB, Binary Mask, and Multi Mask PNGs."""
    try:
        with rasterio.open(tif_path) as src:
            # Band mapping from generate_image_dataset.py:
            # [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDSI, NDWI, NDVI]
            # Actually, check generate_image_dataset.py: 
            # bands = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","NDSI","NDWI","NDVI"] (13 bands)
            # Labels (masks) are added in create_glacier_mask and create_multiclass_mask.
            # Let's verify the band count.
            
            # RGB: B4(3), B3(2), B2(1)
            r = src.read(3)
            g = src.read(2)
            b = src.read(1)
            
            rgb = np.dstack((r, g, b))
            p2, p98 = np.percentile(rgb, (2, 98))
            rgb = np.clip((rgb - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(output_dir, Path(tif_path).stem + "_rgb.png"))
            
            # Binary Mask: If present (usually after index 12)
            # In our dataset creation, we add glacier_mask and multiclass_mask.
            # If src.count > 13, then band 14 is glacier_mask, band 15 is multiclass_mask.
            if src.count >= 14:
                mask = src.read(14)
                # Scale 0-1 to 0-255 for visibility
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(os.path.join(output_dir, Path(tif_path).stem + "_mask.png"))
            
            if src.count >= 15:
                multi = src.read(15)
                # Classes: 0, 1, 2, 3 -> Map to distinct values for visibility
                multi_img = (multi * 64).astype(np.uint8)
                Image.fromarray(multi_img).save(os.path.join(output_dir, Path(tif_path).stem + "_multimask.png"))
                
            return True
    except Exception as e:
        print(f"Error converting {tif_path}: {e}")
        return False

def main():
    base_dir = Path(__file__).parent.parent / "local_dataset"
    patch_dir = base_dir / "patches"
    output_base = base_dir / "patches_png"
    
    tif_files = glob.glob(str(patch_dir / "**" / "*.tif"), recursive=True)
    print(f"Found {len(tif_files)} patches to convert...")
    
    count = 0
    for tif in tif_files:
        # Maintain directory structure (year/season)
        rel_path = os.path.relpath(os.path.dirname(tif), patch_dir)
        out_dir = output_base / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if convert_tif_to_png(tif, out_dir):
            count += 1
            if count % 50 == 0:
                print(f"Converted {count} patches...")
                
    print(f"Successfully converted {count} patches to PNG.")

if __name__ == "__main__":
    main()
