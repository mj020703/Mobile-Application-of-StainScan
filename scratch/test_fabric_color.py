import cv2
import numpy as np
from PIL import Image

def test_fabric_detection(img_path):
    im = Image.open(img_path).convert("RGB")
    arr = np.array(im)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # 1. Old method:
    w, h = im.size
    crop_box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
    cropped = im.crop(crop_box).convert("L")
    mean_lum = float(np.mean(np.array(cropped)))
    old_res = "Grey Cotton" if mean_lum < 165 else "White Cotton"
    
    # 2. Improved method:
    # Fabric background is the brighter part of the fabric (excluding stains/shadows)
    # 85th to 95th percentile of the fabric luminance
    p85 = float(np.percentile(gray, 85))
    p90 = float(np.percentile(gray, 90))
    # In grey cotton, even the brightest highlights rarely exceed 130-140.
    # In white cotton, the cloth highlights/weave are always > 150.
    
    # Also in CIELAB: L channel
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    l_p90 = float(np.percentile(l_chan, 90))
    
    new_res = "Grey Cotton" if p90 < 140 else "White Cotton"
    
    print(f"File: {img_path}")
    print(f"  Old Mean: {mean_lum:.1f} -> {old_res}")
    print(f"  P85: {p85:.1f}, P90: {p90:.1f}, L_P90: {l_p90:.1f} -> {new_res}")

test_fabric_detection("scratch/user_scan_0.jpg")
test_fabric_detection("scratch/crop_tight.jpg")
