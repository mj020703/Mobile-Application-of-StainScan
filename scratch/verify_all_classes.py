import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import glob, io, cv2
import numpy as np
from PIL import Image
import rawpy
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
    try:
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
        num_stain_pixels = int(np.sum(stain_mask))
        total_crop_pixels = crop_gray.size
        
        if num_stain_pixels < 25:
            return cnn_label, cnn_probs
            
        stain_gray = crop_gray[stain_mask > 0]
        min_gray = float(np.min(stain_gray))
        p5_gray = float(np.percentile(stain_gray, 5))
        
        dark_pixels = np.sum(crop_gray < 65)
        dark_fraction = float(dark_pixels / total_crop_pixels)
        
        lap = cv2.Laplacian(crop_gray, cv2.CV_64F)
        stain_lap_var = float(np.var(lap[stain_mask > 0]))
        
        stain_bgr_pix = crop_bgr[stain_mask > 0]
        r_mean = float(np.mean(stain_bgr_pix[:, 2]))
        b_mean = float(np.mean(stain_bgr_pix[:, 0]))
        warmth_diff = r_mean - b_mean
        
        stain_h = crop_hsv[:, :, 0][stain_mask > 0]
        stain_s = crop_hsv[:, :, 1][stain_mask > 0]
        mean_sat = float(np.mean(stain_s))
        
        amber_pixels = np.sum((stain_h >= 14) & (stain_h <= 45) & (stain_s >= 14))
        amber_ratio = float(amber_pixels / max(1, num_stain_pixels))
        
        is_white = "White" in fabric_color
        
        # 1. Used Cooking Oil -> Black Ballpen Ink override
        if cnn_label == "Used Cooking Oil":
            if is_white:
                # On White Cotton: cooking oil is translucent amber (min_gray >= 50, amber_ratio >= 0.15).
                # Pitch black ink has min_gray < 40 or p5_gray < 70, with non-amber warmth.
                if (min_gray < 40 or p5_gray < 70 or dark_fraction > 0.015) and warmth_diff < 15:
                    p_ink = max(0.96, float(np.max(cnn_probs)))
                    p_oil = float(cnn_probs[1]) * 0.04
                    p_mud = float(cnn_probs[2]) * 0.04
                    new_probs = np.array([p_ink, p_oil, p_mud])
                    return "Black Ballpen Ink", new_probs / np.sum(new_probs)
            else:
                if min_gray < 20 and p5_gray < 40:
                    p_ink = max(0.96, float(np.max(cnn_probs)))
                    p_oil = float(cnn_probs[1]) * 0.04
                    p_mud = float(cnn_probs[2]) * 0.04
                    new_probs = np.array([p_ink, p_oil, p_mud])
                    return "Black Ballpen Ink", new_probs / np.sum(new_probs)

        # 2. Mud -> Used Cooking Oil override (on Grey Cotton)
        if cnn_label == "Mud":
            if not is_white:
                if mean_sat < 26 and warmth_diff < 9 and min_gray >= 20:
                    p_oil = max(0.95, float(np.max(cnn_probs)))
                    p_mud = float(cnn_probs[2]) * 0.04
                    p_ink = float(cnn_probs[0]) * 0.04
                    new_probs = np.array([p_ink, p_oil, p_mud])
                    return "Used Cooking Oil", new_probs / np.sum(new_probs)
            else:
                if (min_gray < 35 or p5_gray < 65) and warmth_diff < 12 and amber_ratio < 0.20:
                    p_ink = max(0.96, float(np.max(cnn_probs)))
                    p_oil = float(cnn_probs[1]) * 0.04
                    p_mud = float(cnn_probs[2]) * 0.04
                    new_probs = np.array([p_ink, p_oil, p_mud])
                    return "Black Ballpen Ink", new_probs / np.sum(new_probs)
                elif stain_lap_var < 180 and amber_ratio >= 0.40 and min_gray >= 45:
                    p_oil = max(0.95, float(np.max(cnn_probs)))
                    p_mud = float(cnn_probs[2]) * 0.04
                    p_ink = float(cnn_probs[0]) * 0.04
                    new_probs = np.array([p_ink, p_oil, p_mud])
                    return "Used Cooking Oil", new_probs / np.sum(new_probs)

        return cnn_label, cnn_probs
    except Exception as e:
        print(f"Error in disambiguation: {e}")
        return cnn_label, cnn_probs

def predict_single(pil_im, fabric="White Cotton"):
    arr = np.array(pil_im.convert("RGB").resize((224, 224), Image.BILINEAR))
    enhanced = apply_clahe(arr)
    inp = np.expand_dims(preprocess_input(enhanced.astype(np.float32)), axis=0)
    preds = model.predict(inp, verbose=0)[0]
    top1 = int(np.argmax(preds))
    label_map = {0: "Black Ballpen Ink", 1: "Used Cooking Oil", 2: "Mud"}
    cnn_label = label_map[top1]
    final_lbl, final_probs = verify_and_disambiguate_stain(pil_im, cnn_label, preds, fabric)
    return final_lbl, float(np.max(final_probs))

print("=== User Scans ===")
for i in [0, 1, 2]:
    p = f"scratch/user_scan_{i}.jpg"
    if os.path.exists(p):
        l, c = predict_single(Image.open(p), "White Cotton")
        print(f"User Scan {i}: {l} ({c*100:.1f}%)")

print("\n=== White Cotton Tests (5 each) ===")
for cname, expected in [("Ballpen Ink_cotton", "Black Ballpen Ink"), ("Cooking Oil_cotton", "Used Cooking Oil"), ("Mud_cotton", "Mud")]:
    correct = 0
    for p in sorted(glob.glob(f"DataSet/StainScan/dataset/{cname}/*.JPG"))[:5]:
        l, _ = predict_single(Image.open(p), "White Cotton")
        if l == expected: correct += 1
    print(f"  {cname} (JPG): {correct}/5 correct")

print("\n=== Grey Cotton Tests (3 each) ===")
for cname, expected in [("Ballpen Ink_cotton", "Black Ballpen Ink"), ("Cooking Oil_cotton", "Used Cooking Oil"), ("Mud_cotton", "Mud")]:
    correct = 0
    for p in sorted(glob.glob(f"DataSet/StainScan/dataset/{cname}/*.CR2"))[:3]:
        with rawpy.imread(p) as raw:
            thumb = raw.extract_thumb()
            im = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        l, _ = predict_single(im, "Grey Cotton")
        if l == expected: correct += 1
    print(f"  {cname} (CR2): {correct}/3 correct")
