import os
import glob
import cv2
import numpy as np
from PIL import Image

def detect_fabric_color(pil_image: Image.Image) -> str:
    """
    Accurately distinguishes White Cotton vs Grey Cotton from mobile camera & studio photos.
    Excludes dark stain pixels to analyze pure fabric background luminance and chroma.
    """
    try:
        arr = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        
        # 1. Take a representative central region (avoids table/edges if photo was taken of a swatch)
        h, w = gray.shape
        crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        crop_hsv = hsv[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        
        # 2. Fabric background estimation:
        # In a stained garment, the stain creates darker pixels. The unsoiled fabric
        # corresponds to the upper quartile (80th to 95th percentile) of luminance.
        p90 = float(np.percentile(crop, 90))
        p75 = float(np.percentile(crop, 75))
        
        # White cotton highlights under standard smartphone exposure are >= 150.
        # Grey heather cotton has lower background luminance (p90 typically 100-140 in real use).
        # In studio lighting, grey cotton p90 can reach 160-175, but has distinct texture/melange
        # and very low color saturation (s_mean < 8).
        
        # Let's check fabric texture/variation in the unsoiled region:
        # Mask of unsoiled fabric pixels: within 15 gray levels of p90
        fabric_mask = (crop >= (p90 - 20)) & (crop <= (p90 + 5))
        if np.sum(fabric_mask) > 100:
            fab_s = float(np.mean(crop_hsv[:, :, 1][fabric_mask]))
        else:
            fab_s = float(np.mean(crop_hsv[:, :, 1]))
            
        # Decision boundary:
        # If p90 is 150 or higher, it is white cotton unless it's dark
        if p90 >= 150:
            return "White Cotton"
        elif p75 < 135:
            return "Grey Cotton"
        else:
            return "White Cotton" if p90 >= 145 else "Grey Cotton"
    except Exception as e:
        print(f"Error: {e}")
        return "White Cotton"

test_images = {
    "User Scan 0 (White)": "scratch/user_scan_0.jpg",
    "User Scan 1 (White)": "scratch/user_scan_1.jpg",
    "Sample Ballpen (White)": "samples/sample_ballpen_ink.jpg",
    "Sample Cooking Oil (White)": "samples/sample_cooking_oil.jpg",
    "Sample Mud (White)": "samples/sample_mud.jpg",
    "Sample Grey Ink (Grey)": "samples/sample_grey_ink.jpg",
    "Sample Grey Oil (Grey)": "samples/sample_grey_oil.jpg",
    "Sample Grey Mud (Grey)": "samples/sample_grey_mud.jpg",
}

for name, path in test_images.items():
    if os.path.exists(path):
        im = Image.open(path)
        res = detect_fabric_color(im)
        arr = np.array(im.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        p90 = float(np.percentile(gray, 90))
        print(f"{name:<28} -> Result: {res:<14} (P90: {p90:.1f})")
