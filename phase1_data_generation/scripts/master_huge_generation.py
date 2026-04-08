import subprocess
import time
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow imports from phase1_data_generation
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Also add phase1_data_generation to sys.path for relative imports within it
sys.path.append(str(root_dir / "phase1_data_generation"))

from config.settings import YEARS

def run_huge_generation():
    seasons = ["melt", "winter"]
    
    print("STARTING HUGE DATASET GENERATION (2018-2025)")
    print(f"Targeting years: {YEARS}")
    
    for year in YEARS:
        for season in seasons:
            print(f"\n{'='*60}")
            print(f"  YEAR: {year}   SEASON: {season}")
            print(f"{'='*60}")
            
            # 1. Run Numerical Dataset Generation
            print(f"--- Generating Numerical Samples for {year} {season} ---")
            try:
                subprocess.run([
                    "python", "phase1_data_generation/scripts/generate_numerical_dataset.py",
                    "--year", str(year),
                    "--season", season
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error generating numerical data for {year} {season}: {e}")
            
            # 2. Run Image Patch Generation
            print(f"\n--- Generating Image Patches for {year} {season} ---")
            try:
                # Using --no_full_aoi to save time/space for this huge run, focusing on patches
                subprocess.run([
                    "python", "phase1_data_generation/scripts/generate_image_dataset.py",
                    "--year", str(year),
                    "--season", season,
                    "--patches", "312",
                    "--no_full_aoi"
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error generating image data for {year} {season}: {e}")
            
            # Small delay to prevent GEE rate limiting
            time.sleep(2)

    print("\nHUGE DATASET GENERATION COMPLETE!")
    print("Next steps: Run process_downloaded_data.py and convert_patches_to_png.py")

if __name__ == "__main__":
    run_huge_generation()
