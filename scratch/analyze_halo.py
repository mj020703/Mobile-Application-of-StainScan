import cv2
import numpy as np

def analyze_stain_halo(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Background brightness (top 10% brightest pixels)
    bg_thresh = np.percentile(gray, 90)
    
    # 1. Dark core: pixels significantly darker than background
    dark_core = gray < (bg_thresh - 50)
    
    # 2. Halo / gradient: pixels between dark core and background
    # E.g. between (bg_thresh - 45) and (bg_thresh - 5)
    halo = (gray >= (bg_thresh - 45)) & (gray < (bg_thresh - 8))
    
    # Check hue in halo
    if np.sum(halo) > 20:
        halo_h = hsv[:, :, 0][halo]
        halo_s = hsv[:, :, 1][halo]
        halo_v = hsv[:, :, 2][halo]
        
        # In OpenCV:
        # Yellow/amber hue is roughly 15 - 45
        yellow_halo_pixels = np.sum((halo_h >= 12) & (halo_h <= 45) & (halo_s >= 8))
        yellow_halo_ratio = yellow_halo_pixels / np.sum(halo)
    else:
        yellow_halo_ratio = 0.0
        
    print(f"\n{img_path}:")
    print(f"  Background P90: {bg_thresh:.1f}")
    print(f"  Dark core pixels: {np.sum(dark_core)}")
    print(f"  Halo pixels: {np.sum(halo)}")
    print(f"  Yellow/amber halo ratio: {yellow_halo_ratio*100:.1f}%")

analyze_stain_halo("scratch/user_scan_0.jpg")
analyze_stain_halo("scratch/sample_dataset_oil.jpg")
analyze_stain_halo("scratch/sample_dataset_mud.jpg")
analyze_stain_halo("scratch/mud_0.jpg")
