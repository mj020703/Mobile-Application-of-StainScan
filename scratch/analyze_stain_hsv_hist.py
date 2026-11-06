import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from PIL import Image

for name in ["scratch/recent_mobile_scan_2.jpg", "scratch/user_scan_0.jpg"]:
    if not os.path.exists(name):
        continue
    img = Image.open(name).convert("RGB")
    arr = np.array(img)
    h_img, w_img, _ = arr.shape
    crop_rgb = arr[int(h_img * 0.10):int(h_img * 0.90), int(w_img * 0.10):int(w_img * 0.90)]
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = crop_hsv[:, :, 0], crop_hsv[:, :, 1], crop_hsv[:, :, 2]
    
    p90 = float(np.percentile(crop_gray, 90))
    stain_mask = (crop_gray < (p90 - 15)) | ((s_chan > 25) & (crop_gray < (p90 - 5)))
    
    stain_h = h_chan[stain_mask]
    stain_s = s_chan[stain_mask]
    stain_v = v_chan[stain_mask]
    stain_gray = crop_gray[stain_mask]
    
    print(f"\n==========================================")
    print(f"IMAGE: {name}")
    print(f"Total stain pixels: {len(stain_h)}")
    print(f"Hue mean: {np.mean(stain_h):.1f}, median: {np.median(stain_h):.1f}")
    print(f"Sat mean: {np.mean(stain_s):.1f}, median: {np.median(stain_s):.1f}")
    print(f"Val mean: {np.mean(stain_v):.1f}, median: {np.median(stain_v):.1f}")
    print(f"Gray mean: {np.mean(stain_gray):.1f}, min: {np.min(stain_gray):.1f}, p5: {np.percentile(stain_gray, 5):.1f}")
    
    # Check hue ranges:
    # 1. Amber/gold: 14 <= H <= 45
    amber_count = np.sum((stain_h >= 14) & (stain_h <= 45))
    # 2. Broad warm amber: 12 <= H <= 55
    broad_amber = np.sum((stain_h >= 12) & (stain_h <= 55))
    # 3. Blue ink: 100 <= H <= 180 (or 130 <= H <= 170)
    blue_ink = np.sum((stain_h >= 100) & (stain_h <= 180) & (stain_s > 25))
    # 4. Pure dark/black: gray < 45
    dark_count = np.sum(stain_gray < 45)
    
    print(f"Amber (14-45): {amber_count} ({amber_count/len(stain_h):.2%})")
    print(f"Broad warm (12-55): {broad_amber} ({broad_amber/len(stain_h):.2%})")
    print(f"Blue ink (100-180, S>25): {blue_ink} ({blue_ink/len(stain_h):.2%})")
    print(f"Dark pixels (Gray<45): {dark_count} ({dark_count/len(stain_h):.2%})")
