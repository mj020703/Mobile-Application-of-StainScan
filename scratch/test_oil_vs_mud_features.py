import glob
import cv2
import numpy as np
from PIL import Image

def extract_stain_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Estimate background fabric luminance
    p90 = np.percentile(gray, 90)
    p10 = np.percentile(gray, 10)
    
    # Stain mask: pixels darker than fabric
    # (using dynamic threshold based on background)
    stain_thresh = p90 - max(12.0, (p90 - p10) * 0.18)
    stain_mask = (gray < stain_thresh).astype(np.uint8)
    
    # Exclude tiny noise speckles
    num_stain_pixels = int(np.sum(stain_mask))
    if num_stain_pixels < 30:
        return {"num_pixels": 0}
        
    # Find contours / connected components of stain spots
    contours, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze largest contours
    circularities = []
    solidity_list = []
    aspect_ratios = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circ = 4 * np.pi * (area / (perimeter * perimeter))
                circularities.append(circ)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity_list.append(area / hull_area)
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / max(1, h)
            aspect_ratios.append(min(ar, 1.0/ar))
            
    mean_circ = float(np.mean(circularities)) if circularities else 0.0
    mean_solidity = float(np.mean(solidity_list)) if solidity_list else 0.0
    
    # Color features in stain
    stain_h = hsv[:, :, 0][stain_mask > 0]
    stain_s = hsv[:, :, 1][stain_mask > 0]
    stain_v = hsv[:, :, 2][stain_mask > 0]
    stain_bgr = img[stain_mask > 0]
    
    # Amber/yellow lipid pixels (H 14 to 45, S >= 15)
    amber_count = np.sum((stain_h >= 14) & (stain_h <= 45) & (stain_s >= 14))
    amber_ratio = float(amber_count / max(1, num_stain_pixels))
    
    # Internal gradient smoothness (Oil is a liquid with smooth internal wicking gradient; Mud has gritty crust)
    # Laplacian variance INSIDE stain mask
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    stain_lap_var = float(np.var(lap[stain_mask > 0])) if num_stain_pixels > 0 else 0.0
    
    # Halo analysis: annular region surrounding the core stain
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(stain_mask, kernel, iterations=2)
    halo_mask = (dilated > 0) & (stain_mask == 0)
    
    halo_pixels = int(np.sum(halo_mask))
    if halo_pixels > 20:
        halo_h = hsv[:, :, 0][halo_mask]
        halo_s = hsv[:, :, 1][halo_mask]
        # Translucent amber/gold lipid halo
        halo_amber = float(np.mean((halo_h >= 12) & (halo_h <= 48) & (halo_s >= 10)))
    else:
        halo_amber = 0.0
        
    return {
        "num_pixels": num_stain_pixels,
        "num_spots": len(circularities),
        "mean_circ": mean_circ,
        "mean_solidity": mean_solidity,
        "amber_ratio": amber_ratio,
        "stain_lap_var": stain_lap_var,
        "halo_amber": halo_amber
    }

samples = {
    "User Scan 0 (Used Oil droplets on White)": "scratch/user_scan_0.jpg",
    "User Scan 1 (Used Oil on Grey)":           "scratch/user_scan_1.jpg",
    "Sample Cooking Oil (White)":               "samples/sample_cooking_oil.jpg",
    "Sample Grey Oil (Grey)":                   "samples/sample_grey_oil.jpg",
    "Sample Mud (White)":                       "samples/sample_mud.jpg",
    "Sample Grey Mud (Grey)":                   "samples/sample_grey_mud.jpg",
}

print(f"{'Image':<40} {'Spots':<6} {'Circ':<6} {'Solid':<6} {'Amber%':<8} {'HaloAmber%':<12} {'LapVar':<8}")
print("-" * 90)
for name, p in samples.items():
    feat = extract_stain_features(p)
    if feat:
        print(f"{name:<40} {feat['num_spots']:<6} {feat['mean_circ']:<6.2f} {feat['mean_solidity']:<6.2f} {feat['amber_ratio']*100:<8.1f} {feat['halo_amber']*100:<12.1f} {feat['stain_lap_var']:<8.1f}")
