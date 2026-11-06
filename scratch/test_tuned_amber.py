import sys, os
sys.path.insert(0, os.path.abspath("."))
import glob
import cv2
import numpy as np
from PIL import Image

def get_stain_color_ratios(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    h_img, w_img, _ = arr.shape
    crop_rgb = arr[int(h_img * 0.10):int(h_img * 0.90), int(w_img * 0.10):int(w_img * 0.90)]
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = crop_hsv[:, :, 0], crop_hsv[:, :, 1], crop_hsv[:, :, 2]
    
    p90 = float(np.percentile(crop_gray, 90))
    stain_mask = (crop_gray < (p90 - 15)) | ((s_chan > 25) & (crop_gray < (p90 - 5)))
    num_stain = int(np.sum(stain_mask))
    if num_stain < 25:
        stain_mask = (crop_gray < (p90 - 8))
        num_stain = int(np.sum(stain_mask))
    
    min_gray = float(np.min(crop_gray))
    
    # Ink clustering: Hue <= 15 or >= 140, dark V < 65, low sat
    ink_hue = ((h_chan <= 15) | (h_chan >= 140))
    dark_pix = (crop_gray < 55) & (v_chan < 65)
    ink_pix = ink_hue & dark_pix & (s_chan < 110)
    ink_count = int(np.sum(ink_pix & (stain_mask > 0)))
    ink_density = ink_count / max(1, num_stain)
    
    blue_ink = ((h_chan >= 110) & (h_chan <= 180) & (s_chan > 20) & (v_chan < 100))
    blue_ink_ratio = int(np.sum(blue_ink & (stain_mask > 0))) / max(1, num_stain)
    
    # Tuned Amber/Gold: Hue 12-48, Sat >= 6, Val >= 45
    amber_mask = (h_chan >= 12) & (h_chan <= 48) & (s_chan >= 6) & (v_chan >= 45)
    amber_ratio = int(np.sum(amber_mask & (stain_mask > 0))) / max(1, num_stain)
    
    val_drop = p90 - crop_gray[stain_mask > 0] if num_stain > 0 else np.array([0.0])
    mean_drop = float(np.mean(val_drop))
    opaque_ratio = float(np.mean(val_drop > 40))
    
    return {
        "min_gray": min_gray,
        "ink_count": ink_count,
        "ink_density": ink_density,
        "blue_ink_ratio": blue_ink_ratio,
        "amber_ratio": amber_ratio,
        "mean_drop": mean_drop,
        "opaque_ratio": opaque_ratio
    }

print("=== User Scans (Cooking Oil) ===")
for p in ["scratch/user_scan_0.jpg", "scratch/user_scan_1.jpg", "scratch/user_scan_2.jpg", "scratch/recent_mobile_scan_2.jpg"]:
    if os.path.exists(p):
        r = get_stain_color_ratios(Image.open(p))
        print(f"{os.path.basename(p)}: amber={r['amber_ratio']:.2%}, ink_dens={r['ink_density']:.4f}, blue_ink={r['blue_ink_ratio']:.4f}, ink_cnt={r['ink_count']}, min_g={r['min_gray']}")

print("\n=== Real Dataset Ink Scans ===")
for p in glob.glob(r"DataSet/StainScan/dataset/Ballpen Ink_cotton/*.JPG")[:6]:
    r = get_stain_color_ratios(Image.open(p))
    print(f"{os.path.basename(p)}: amber={r['amber_ratio']:.2%}, ink_dens={r['ink_density']:.4f}, blue_ink={r['blue_ink_ratio']:.4f}, ink_cnt={r['ink_count']}, min_g={r['min_gray']}")

print("\n=== Real Dataset Mud Scans ===")
for p in glob.glob(r"DataSet/StainScan/dataset/Mud_cotton/*.JPG")[:6]:
    r = get_stain_color_ratios(Image.open(p))
    print(f"{os.path.basename(p)}: amber={r['amber_ratio']:.2%}, ink_dens={r['ink_density']:.4f}, blue_ink={r['blue_ink_ratio']:.4f}, ink_cnt={r['ink_count']}, opaque={r['opaque_ratio']:.2%}")
