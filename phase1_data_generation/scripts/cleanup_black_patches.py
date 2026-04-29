import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path

def is_black_image(img_path, threshold=5):
    """
    Check if an image is mostly black/empty.
    threshold: mean pixel value below which it's considered black.
    """
    try:
        with Image.open(img_path) as img:
            # Convert to grayscale to check brightness
            gray = img.convert('L')
            arr = np.array(gray)
            mean_val = np.mean(arr)
            return mean_val < threshold
    except Exception as e:
        print(f"Error checking {img_path}: {e}")
        return False

def cleanup_black_images():
    base_dir = Path("d:/Final Year Project/UsingClaude/GlacierMeltingDetection_COMPLETE/GlacierMeltingDetection/phase1_data_generation/local_dataset")
    png_dir = base_dir / "patches_png"
    tif_dir = base_dir / "patches"
    
    # We only care about the RGB images to decide if a patch is 'empty'
    rgb_files = glob.glob(str(png_dir / "**" / "*_rgb.png"), recursive=True)
    
    print(f"Scanning {len(rgb_files)} RGB patches for empty/black content...")
    
    removed_count = 0
    for rgb_path in rgb_files:
        if is_black_image(rgb_path):
            p = Path(rgb_path)
            # Identify the base filename (e.g., glacier_patch_2018_melt_0001)
            base_name = p.name.replace("_rgb.png", "")
            
            # Identify all related PNGs (mask, multimask, rgb, and the plain one if exists)
            related_pngs = glob.glob(str(p.parent / f"{base_name}*.png"))
            
            # Identify the source TIF
            # rel_path matches year/season
            rel_path = p.relative_to(png_dir).parent
            source_tif = tif_dir / rel_path / f"{base_name}.tif"
            
            # Deleting files
            print(f"Removing black patch: {base_name}")
            for f in related_pngs:
                try:
                    os.remove(f)
                except: pass
            
            if source_tif.exists():
                try:
                    os.remove(source_tif)
                except: pass
            
            removed_count += 1
            
    print(f"\n✅ CLEANUP COMPLETE")
    print(f"Removed {removed_count} empty/black patches from the research dataset.")

if __name__ == "__main__":
    cleanup_black_images()
