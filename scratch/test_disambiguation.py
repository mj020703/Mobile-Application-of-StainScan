import cv2
import numpy as np
from PIL import Image

def analyze_stain_characteristics(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Background fabric estimation (excluding dark spots)
    p90 = np.percentile(gray, 90)
    
    # 2. Stain region: where pixels are darker than fabric background
    stain_mask = gray < (p90 - 15)
    
    if np.sum(stain_mask) < 20:
        return {"error": "no stain detected"}
        
    stain_h = hsv[:, :, 0][stain_mask]
    stain_s = hsv[:, :, 1][stain_mask]
    stain_v = hsv[:, :, 2][stain_mask]
    stain_bgr = img[stain_mask]
    
    # Yellow/amber hue range in OpenCV (Hue is 0-179):
    # Pure red: 0-10, Orange: 10-22, Yellow/Gold/Amber: 22-45, Green: 45-85
    # Used cooking oil has distinct yellow/amber hue: 15 to 45 with saturation >= 15
    amber_pixels = (stain_h >= 14) & (stain_h <= 45) & (stain_s >= 12)
    amber_ratio = np.mean(amber_pixels)
    
    # In BGR: Cooking oil has Red > Green > Blue (warm golden/amber)
    r = stain_bgr[:, 2].astype(float)
    g = stain_bgr[:, 1].astype(float)
    b = stain_bgr[:, 0].astype(float)
    
    # Amber/yellow lipid indicator: R significantly greater than B (R - B > 15) and R >= G
    warm_lipid_pixels = (r - b > 15) & (r >= g) & (g > b)
    warm_lipid_ratio = np.mean(warm_lipid_pixels)
    
    # Capillary diffusion: Oil stains have a broad, smooth gradient (low Laplacian inside stain)
    # Mud has abrupt, chalky, sharp crust edges
    
    print(f"File: {img_path}")
    print(f"  Stain pixels: {np.sum(stain_mask)}")
    print(f"  Mean Hue: {np.mean(stain_h):.1f}")
    print(f"  Mean Saturation: {np.mean(stain_s):.1f}")
    print(f"  Amber ratio: {amber_ratio*100:.1f}%")
    print(f"  Warm lipid ratio (R>G>B): {warm_lipid_ratio*100:.1f}%")

print("--- USER SCAN ---")
analyze_stain_characteristics("scratch/user_scan_0.jpg")
print("\n--- DATASET OIL ---")
analyze_stain_characteristics("scratch/sample_dataset_oil.jpg")
print("\n--- DATASET MUD ---")
analyze_stain_characteristics("scratch/sample_dataset_mud.jpg")
print("\n--- SAMPLE MUD ---")
analyze_stain_characteristics("samples/sample_mud.jpg")
print("\n--- SAMPLE GREY MUD ---")
analyze_stain_characteristics("samples/sample_grey_mud.jpg")
