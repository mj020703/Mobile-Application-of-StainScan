import cv2
import numpy as np
from PIL import Image

def analyze_fabric_texture_and_color(path):
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Take fabric pixels (excluding dark stains)
    # Brightest 30% of pixels in central region
    h, w = gray.shape
    crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    crop_rgb = arr[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    
    p90 = np.percentile(crop, 90)
    p70 = np.percentile(crop, 70)
    
    fabric_mask = (crop >= (p90 - 25)) & (crop <= p90)
    if np.sum(fabric_mask) < 100:
        fabric_mask = crop >= p70
        
    fab_pixels = crop_rgb[fabric_mask]
    fab_gray = crop[fabric_mask]
    
    # 1. Color channels in fabric
    r = float(np.mean(fab_pixels[:, 0]))
    g = float(np.mean(fab_pixels[:, 1]))
    b = float(np.mean(fab_pixels[:, 2]))
    
    # Warmth index: R - B (White cotton is warmer / cream: R > B; Grey cotton is neutral/cool: R ~ B)
    warmth = r - b
    
    # 2. Local texture variation of the fabric (Heather grey has high-frequency thread noise: black & white threads)
    # White cotton has uniform smooth threads
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    lap_var = float(np.var(lap[fabric_mask]))
    
    # Gray thread std dev
    gray_std = float(np.std(fab_gray))
    
    print(f"{path}:")
    print(f"  P90: {p90:.1f}, R: {r:.1f}, G: {g:.1f}, B: {b:.1f}, Warmth (R-B): {warmth:.2f}")
    print(f"  Fabric gray std: {gray_std:.2f}, Laplacian var: {lap_var:.2f}")

for p in [
    "scratch/user_scan_0.jpg",
    "samples/sample_ballpen_ink.jpg",
    "samples/sample_cooking_oil.jpg",
    "samples/sample_mud.jpg",
    "samples/sample_grey_ink.jpg",
    "samples/sample_grey_oil.jpg",
    "samples/sample_grey_mud.jpg"
]:
    analyze_fabric_texture_and_color(p)
