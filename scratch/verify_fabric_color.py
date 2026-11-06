import os
import glob
import cv2
import numpy as np
from PIL import Image

def detect_fabric_color_from_image(pil_image: Image.Image) -> str:
    """
    Detects whether the cotton garment is White Cotton or Grey Cotton.
    Isolates unsoiled fabric pixels to prevent dark stains from lowering the fabric brightness,
    and checks texture contrast to separate smooth white weave from heather/melange grey.
    """
    try:
        arr = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
        # Take central 80% to avoid extreme image borders / table surfaces
        h, w = gray.shape
        crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        
        # Upper percentile represents the unsoiled garment fabric
        p90 = float(np.percentile(crop, 90))
        p75 = float(np.percentile(crop, 75))
        
        # Fabric mask: pixels representing the garment background (near p90)
        fab_mask = (crop >= (p90 - 20)) & (crop <= (p90 + 5))
        
        # Laplacian variance of fabric region (heather grey has dark/light mottled fibers)
        lap = cv2.Laplacian(crop, cv2.CV_64F)
        lap_var = float(np.var(lap[fab_mask])) if np.sum(fab_mask) > 50 else float(cv2.Laplacian(crop, cv2.CV_64F).var())
        
        # White cotton:
        # 1. Under smartphone/indoor light: p90 >= 150 with smooth weave (lap_var < 100)
        # 2. Under bright studio light: p90 >= 190
        # Grey cotton:
        # 1. Under normal light: p90 is dark (< 145)
        # 2. Under studio flash: p90 is 155-180, but lap_var is very high (> 110) due to heather yarn
        if p90 >= 190:
            return "White Cotton"
        elif p90 < 145:
            return "Grey Cotton"
        else:
            # Ambiguous range (145 <= p90 < 190)
            if lap_var > 110:
                return "Grey Cotton"
            else:
                return "White Cotton"
    except Exception:
        return "White Cotton"

test_cases = {
    "User Scan 0 (White phone photo)": ("scratch/user_scan_0.jpg", "White Cotton"),
    "User Scan 1 (Grey phone photo)":  ("scratch/user_scan_1.jpg", "Grey Cotton"),
    "Sample Ballpen (White studio)":   ("samples/sample_ballpen_ink.jpg", "White Cotton"),
    "Sample Cooking Oil (White studio)": ("samples/sample_cooking_oil.jpg", "White Cotton"),
    "Sample Mud (White studio)":       ("samples/sample_mud.jpg", "White Cotton"),
    "Sample Grey Ink (Grey studio)":   ("samples/sample_grey_ink.jpg", "Grey Cotton"),
    "Sample Grey Oil (Grey studio)":   ("samples/sample_grey_oil.jpg", "Grey Cotton"),
    "Sample Grey Mud (Grey studio)":   ("samples/sample_grey_mud.jpg", "Grey Cotton"),
}

print(f"{'Image':<35} {'Expected':<15} {'Detected':<15} {'Status'}")
print("-" * 75)
all_pass = True
for name, (path, expected) in test_cases.items():
    im = Image.open(path)
    res = detect_fabric_color_from_image(im)
    match = "[PASS]" if res == expected else "[FAIL]"
    if res != expected: all_pass = False
    print(f"{name:<35} {expected:<15} {res:<15} {match}")

print("-" * 75)
print(f"All Passed: {all_pass}")
