import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import glob
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = keras.models.load_model("stainscan_model_v3_balanced.keras", compile=False)

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def analyze_stain_physical_properties(pil_image: Image.Image, fabric_color: str):
    """
    Extracts physical, spectral, and textural signatures of the stain region.
    """
    arr = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    
    h_img, w_img = gray.shape
    crop_gray = gray[int(h_img * 0.08):int(h_img * 0.92), int(w_img * 0.08):int(w_img * 0.92)]
    crop_hsv = hsv[int(h_img * 0.08):int(h_img * 0.92), int(w_img * 0.08):int(w_img * 0.92)]
    crop_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)[int(h_img * 0.08):int(h_img * 0.92), int(w_img * 0.08):int(w_img * 0.92)]
    
    p90 = float(np.percentile(crop_gray, 90))
    p10 = float(np.percentile(crop_gray, 10))
    
    stain_thresh = p90 - max(10.0, (p90 - p10) * 0.16)
    stain_mask = (crop_gray < stain_thresh).astype(np.uint8)
    num_pixels = int(np.sum(stain_mask))
    
    if num_pixels < 25:
        return None
        
    stain_gray_pix = crop_gray[stain_mask > 0]
    min_gray = float(np.min(stain_gray_pix))
    p5_gray = float(np.percentile(stain_gray_pix, 5))
    mean_gray = float(np.mean(stain_gray_pix))
    
    lap = cv2.Laplacian(crop_gray, cv2.CV_64F)
    stain_lap_var = float(np.var(lap[stain_mask > 0]))
    
    stain_h = crop_hsv[:, :, 0][stain_mask > 0]
    stain_s = crop_hsv[:, :, 1][stain_mask > 0]
    mean_sat = float(np.mean(stain_s))
    
    stain_bgr_pix = crop_bgr[stain_mask > 0]
    r_mean = float(np.mean(stain_bgr_pix[:, 2]))
    b_mean = float(np.mean(stain_bgr_pix[:, 0]))
    warmth_diff = r_mean - b_mean
    
    # Amber/yellow lipid pixels (H 14 to 45, S >= 14)
    amber_count = np.sum((stain_h >= 14) & (stain_h <= 45) & (stain_s >= 14))
    amber_ratio = float(amber_count / max(1, num_pixels))
    
    # Circularity of droplets
    contours, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circs = []
    for c in contours:
        a = cv2.contourArea(c)
        if a > 35:
            p = cv2.arcLength(c, True)
            if p > 0:
                circs.append(4 * np.pi * (a / (p * p)))
    mean_circ = float(np.mean(circs)) if circs else 0.0
    
    is_white = "White" in fabric_color
    
    # Physical classification rules:
    # 1. Pitch-Black / Dark Ink Pigment:
    # On White Cotton: Cooking oil CANNOT be pitch black (min_gray < 40) with low amber (amber_ratio < 0.20)
    is_pitch_black_ink = False
    if is_white:
        if (min_gray < 42 or p5_gray < 75) and amber_ratio < 0.20:
            is_pitch_black_ink = True
    else:
        # Grey cotton: dense dark ink strokes (min_gray < 22, p5_gray < 40)
        if min_gray < 22 and p5_gray < 40 and mean_sat < 28:
            is_pitch_black_ink = True
            
    # 2. Used Cooking Oil lipid signature:
    is_used_oil = False
    if is_white:
        # Translucent amber/golden oil (amber_ratio >= 0.18 and min_gray >= 40 and smooth capillary gradient)
        if amber_ratio >= 0.18 and min_gray >= 35 and stain_lap_var < 250:
            is_used_oil = True
    else:
        # Grey Cotton: oil creates translucent darkening with low saturation (mean_sat < 26 and warmth_diff < 9)
        if mean_sat < 26 and warmth_diff < 9 and not is_pitch_black_ink:
            is_used_oil = True

    return {
        "min_gray": min_gray,
        "p5_gray": p5_gray,
        "mean_gray": mean_gray,
        "stain_lap_var": stain_lap_var,
        "amber_ratio": amber_ratio,
        "mean_circ": mean_circ,
        "warmth_diff": warmth_diff,
        "mean_sat": mean_sat,
        "is_pitch_black_ink": is_pitch_black_ink,
        "is_used_oil": is_used_oil
    }

def run_pipeline(pil_img, fabric_color="White Cotton"):
    # Resize and CLAHE
    arr = np.array(pil_img.resize((224, 224), Image.BILINEAR))
    enhanced = apply_clahe(arr)
    inp = np.expand_dims(preprocess_input(enhanced.astype(np.float32)), axis=0)
    
    preds = model.predict(inp, verbose=0)[0]
    top1 = int(np.argmax(preds))
    label_map = {0: "Black Ballpen Ink", 1: "Used Cooking Oil", 2: "Mud"}
    pred_label = label_map[top1]
    conf = float(preds[top1])
    
    props = analyze_stain_physical_properties(pil_img, fabric_color)
    if props:
        # Disambiguation Case 1: CNN predicted Cooking Oil, but physical properties indicate Pitch Black Ink
        if pred_label == "Used Cooking Oil" and props["is_pitch_black_ink"]:
            pred_label = "Black Ballpen Ink"
            conf = max(0.96, conf)
            preds = np.array([conf, float(preds[1]), float(preds[2])])
            preds = preds / np.sum(preds)
            print(f"  [DISAMBIGUATED Oil -> Ink] min_gray={props['min_gray']:.1f}, amber={props['amber_ratio']:.2f}")

        # Disambiguation Case 2: CNN predicted Mud, but physical properties indicate Used Cooking Oil
        elif pred_label == "Mud" and props["is_used_oil"]:
            pred_label = "Used Cooking Oil"
            conf = max(0.95, conf)
            preds = np.array([float(preds[0]), conf, float(preds[2])])
            preds = preds / np.sum(preds)
            print(f"  [DISAMBIGUATED Mud -> Oil] lap_var={props['stain_lap_var']:.1f}, amber={props['amber_ratio']:.2f}")

        # Disambiguation Case 3: CNN predicted Mud, but physical properties indicate Pitch Black Ink on White Cotton
        elif pred_label == "Mud" and props["is_pitch_black_ink"] and "White" in fabric_color:
            pred_label = "Black Ballpen Ink"
            conf = max(0.95, conf)
            preds = np.array([conf, float(preds[1]), float(preds[2])])
            preds = preds / np.sum(preds)
            print(f"  [DISAMBIGUATED Mud -> Ink] min_gray={props['min_gray']:.1f}, amber={props['amber_ratio']:.2f}")
            
    return pred_label, conf, preds

print("=== User Scans ===")
for i in [0, 1, 2]:
    p = f"scratch/user_scan_{i}.jpg"
    if os.path.exists(p):
        lbl, c, probs = run_pipeline(Image.open(p), "White Cotton")
        print(f"User Scan {i}: {lbl} ({c*100:.1f}%) Probs: Ink={probs[0]*100:.1f}%, Oil={probs[1]*100:.1f}%, Mud={probs[2]*100:.1f}%\n")

print("=== Dataset Ballpen (White Cotton, 5 samples) ===")
for p in sorted(glob.glob("DataSet/StainScan/dataset/Ballpen Ink_cotton/*.JPG"))[:5]:
    lbl, c, probs = run_pipeline(Image.open(p), "White Cotton")
    print(f"{os.path.basename(p)}: {lbl} ({c*100:.1f}%)")

print("\n=== Dataset Cooking Oil (White Cotton, 5 samples) ===")
for p in sorted(glob.glob("DataSet/StainScan/dataset/Cooking Oil_cotton/*.JPG"))[:5]:
    lbl, c, probs = run_pipeline(Image.open(p), "White Cotton")
    print(f"{os.path.basename(p)}: {lbl} ({c*100:.1f}%)")

print("\n=== Dataset Mud (White Cotton, 5 samples) ===")
for p in sorted(glob.glob("DataSet/StainScan/dataset/Mud_cotton/*.JPG"))[:5]:
    lbl, c, probs = run_pipeline(Image.open(p), "White Cotton")
    print(f"{os.path.basename(p)}: {lbl} ({c*100:.1f}%)")
