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

def verify_and_disambiguate_stain(pil_image: Image.Image, cnn_label: str, cnn_probs: np.ndarray, fabric_color: str):
    """
    Physically validates CNN prediction against optical, textural, and chrominance ground truth.
    Prevents CNN from misidentifying dark ink as translucent cooking oil or vice versa.
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
    
    # Segment dark stain pixels
    stain_thresh = p90 - max(10.0, (p90 - p10) * 0.16)
    stain_mask = (crop_gray < stain_thresh).astype(np.uint8)
    num_stain_pixels = int(np.sum(stain_mask))
    total_crop_pixels = crop_gray.size
    
    if num_stain_pixels < 25:
        return cnn_label, cnn_probs
        
    stain_gray = crop_gray[stain_mask > 0]
    min_gray = float(np.min(stain_gray))
    p5_gray = float(np.percentile(stain_gray, 5))
    
    # Count dense dark pixels (< 65)
    dark_pixels = np.sum(crop_gray < 65)
    dark_fraction = float(dark_pixels / total_crop_pixels)
    
    # Internal texture variance
    lap = cv2.Laplacian(crop_gray, cv2.CV_64F)
    stain_lap_var = float(np.var(lap[stain_mask > 0]))
    
    # Color characteristics of the stain
    stain_bgr_pix = crop_bgr[stain_mask > 0]
    r_mean = float(np.mean(stain_bgr_pix[:, 2]))
    b_mean = float(np.mean(stain_bgr_pix[:, 0]))
    warmth_diff = r_mean - b_mean
    
    stain_h = crop_hsv[:, :, 0][stain_mask > 0]
    stain_s = crop_hsv[:, :, 1][stain_mask > 0]
    mean_sat = float(np.mean(stain_s))
    
    # Amber lipid hue (H 14 to 45, S >= 14)
    amber_pixels = np.sum((stain_h >= 14) & (stain_h <= 45) & (stain_s >= 14))
    amber_ratio = float(amber_pixels / max(1, num_stain_pixels))
    
    is_white = "White" in fabric_color
    
    # ── RULE 1: BLACK BALLPEN INK vs USED COOKING OIL ───────────────────
    # Used Cooking Oil on White Cotton is a translucent liquid (min_gray >= 50, amber_ratio >= 0.15).
    # It NEVER possesses dense pitch-black ink pigments (min_gray < 40 or dark_pixels > 1% of frame).
    # If the CNN predicted Used Cooking Oil on an image that has dark ink pigment, correct it!
    if cnn_label == "Used Cooking Oil":
        if is_white:
            if (min_gray < 40 or p5_gray < 70 or dark_fraction > 0.015) and warmth_diff < 15:
                print(f"[DISAMBIGUATION] Overriding 'Used Cooking Oil' -> 'Black Ballpen Ink' (min_gray={min_gray:.1f}, dark_frac={dark_fraction:.3f}, warmth={warmth_diff:.1f})")
                p_ink = max(0.96, float(np.max(cnn_probs)))
                p_oil = float(cnn_probs[1]) * 0.05
                p_mud = float(cnn_probs[2]) * 0.05
                new_probs = np.array([p_ink, p_oil, p_mud])
                return "Black Ballpen Ink", new_probs / np.sum(new_probs)
        else:
            # Grey Cotton: oil has mean_sat < 26 and warmth_diff < 9 with no deep black strokes.
            # Ink has sharp dark strokes (min_gray < 20, p5_gray < 40).
            if min_gray < 20 and p5_gray < 40:
                print(f"[DISAMBIGUATION] Overriding 'Used Cooking Oil' -> 'Black Ballpen Ink' on Grey Cotton (min_gray={min_gray:.1f})")
                p_ink = max(0.96, float(np.max(cnn_probs)))
                p_oil = float(cnn_probs[1]) * 0.05
                p_mud = float(cnn_probs[2]) * 0.05
                new_probs = np.array([p_ink, p_oil, p_mud])
                return "Black Ballpen Ink", new_probs / np.sum(new_probs)

    # ── RULE 2: MUD vs USED COOKING OIL (on Grey Cotton) ────────────────
    # Mud has high saturation (mean_sat > 35) and warmth_diff > 10.
    # Cooking oil creates translucent darkening with low saturation (mean_sat < 26, warmth_diff < 9).
    if cnn_label == "Mud":
        if not is_white:
            if mean_sat < 26 and warmth_diff < 9 and min_gray >= 20:
                print(f"[DISAMBIGUATION] Overriding 'Mud' -> 'Used Cooking Oil' on Grey Cotton (mean_sat={mean_sat:.1f}, warmth={warmth_diff:.1f})")
                p_oil = max(0.95, float(np.max(cnn_probs)))
                p_mud = float(cnn_probs[2]) * 0.05
                p_ink = float(cnn_probs[0]) * 0.05
                new_probs = np.array([p_ink, p_oil, p_mud])
                return "Used Cooking Oil", new_probs / np.sum(new_probs)
        else:
            # On White Cotton: if CNN predicted Mud, but it's pitch black ink with no clay warmth:
            if (min_gray < 35 or p5_gray < 65) and warmth_diff < 12 and amber_ratio < 0.20:
                print(f"[DISAMBIGUATION] Overriding 'Mud' -> 'Black Ballpen Ink' on White Cotton (min_gray={min_gray:.1f}, warmth={warmth_diff:.1f})")
                p_ink = max(0.96, float(np.max(cnn_probs)))
                p_oil = float(cnn_probs[1]) * 0.05
                p_mud = float(cnn_probs[2]) * 0.05
                new_probs = np.array([p_ink, p_oil, p_mud])
                return "Black Ballpen Ink", new_probs / np.sum(new_probs)

    return cnn_label, cnn_probs

def test_image(img_path, fabric="White Cotton"):
    im = Image.open(img_path)
    arr = np.array(im.convert("RGB").resize((224, 224), Image.BILINEAR))
    enhanced = apply_clahe(arr)
    inp = np.expand_dims(preprocess_input(enhanced.astype(np.float32)), axis=0)
    
    preds = model.predict(inp, verbose=0)[0]
    top1 = int(np.argmax(preds))
    label_map = {0: "Black Ballpen Ink", 1: "Used Cooking Oil", 2: "Mud"}
    cnn_label = label_map[top1]
    
    final_label, final_probs = verify_and_disambiguate_stain(im, cnn_label, preds, fabric)
    return final_label, float(np.max(final_probs)), final_probs

print("=== User Scans ===")
for i in [0, 1, 2]:
    p = f"scratch/user_scan_{i}.jpg"
    if os.path.exists(p):
        lbl, conf, probs = test_image(p, "White Cotton")
        print(f"User Scan {i} -> {lbl} ({conf*100:.1f}%) | Ink={probs[0]*100:.1f}%, Oil={probs[1]*100:.1f}%, Mud={probs[2]*100:.1f}%\n")

print("=== Dataset 10 Ballpen (White Cotton) ===")
for p in sorted(glob.glob("DataSet/StainScan/dataset/Ballpen Ink_cotton/*.JPG"))[:10]:
    lbl, conf, _ = test_image(p, "White Cotton")
    print(f"  {os.path.basename(p)} -> {lbl} ({conf*100:.1f}%)")

print("\n=== Dataset 10 Cooking Oil (White Cotton) ===")
for p in sorted(glob.glob("DataSet/StainScan/dataset/Cooking Oil_cotton/*.JPG"))[:10]:
    lbl, conf, _ = test_image(p, "White Cotton")
    print(f"  {os.path.basename(p)} -> {lbl} ({conf*100:.1f}%)")

print("\n=== Dataset 10 Mud (White Cotton) ===")
for p in sorted(glob.glob("DataSet/StainScan/dataset/Mud_cotton/*.JPG"))[:10]:
    lbl, conf, _ = test_image(p, "White Cotton")
    print(f"  {os.path.basename(p)} -> {lbl} ({conf*100:.1f}%)")
