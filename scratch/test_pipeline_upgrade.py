import os
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

def normalize_fabric_illumination(img_rgb: np.ndarray) -> np.ndarray:
    try:
        r_p92 = float(np.percentile(img_rgb[:, :, 0], 92))
        g_p92 = float(np.percentile(img_rgb[:, :, 1], 92))
        b_p92 = float(np.percentile(img_rgb[:, :, 2], 92))
        max_chan = max(r_p92, g_p92, b_p92)
        if max_chan < 40.0:
            return img_rgb
        if r_p92 > (b_p92 + 15.0) or g_p92 > (b_p92 + 12.0):
            target_val = min(245.0, max_chan)
            scale_r = np.clip(target_val / max(1.0, r_p92), 0.70, 1.40)
            scale_g = np.clip(target_val / max(1.0, g_p92), 0.70, 1.40)
            scale_b = np.clip(target_val / max(1.0, b_p92), 0.70, 1.40)
            norm = np.zeros_like(img_rgb, dtype=np.float32)
            norm[:, :, 0] = np.clip(img_rgb[:, :, 0].astype(np.float32) * scale_r, 0, 255)
            norm[:, :, 1] = np.clip(img_rgb[:, :, 1].astype(np.float32) * scale_g, 0, 255)
            norm[:, :, 2] = np.clip(img_rgb[:, :, 2].astype(np.float32) * scale_b, 0, 255)
            return norm.astype(np.uint8)
        return img_rgb
    except Exception:
        return img_rgb

def preprocess_image_upgraded(pil_img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    w, h = pil_img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    square_crop = pil_img.crop((left, top, left + crop_size, top + crop_size))
    img_resized = square_crop.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img_resized.convert("RGB"), dtype=np.uint8)
    arr = normalize_fabric_illumination(arr)
    arr_clahe = apply_clahe(arr)
    arr_norm = preprocess_input(arr_clahe.astype(np.float32))
    return np.expand_dims(arr_norm, axis=0)

print("Testing Upgraded Preprocessing on User Scans:")
label_map = {0: "Black Ballpen Ink", 1: "Used Cooking Oil", 2: "Mud"}
for p in ["scratch/user_scan_0.jpg", "scratch/user_scan_1.jpg", "scratch/user_scan_2.jpg", "scratch/recent_mobile_scan_2.jpg"]:
    if os.path.exists(p):
        img = Image.open(p)
        t_arr = preprocess_image_upgraded(img)
        preds = model.predict(t_arr, verbose=0)[0]
        top = int(np.argmax(preds))
        print(f"  {os.path.basename(p)}: {label_map[top]} ({preds[top]*100:.1f}%) | Ink={preds[0]*100:.1f}%, Oil={preds[1]*100:.1f}%, Mud={preds[2]*100:.1f}%")

print("\nTesting on Dataset Samples (White Cotton):")
for stain, folder in [("Ink", "DataSet/StainScan/dataset/Ballpen Ink_cotton"), ("Oil", "DataSet/StainScan/dataset/Cooking Oil_cotton"), ("Mud", "DataSet/StainScan/dataset/Mud_cotton")]:
    files = glob.glob(os.path.join(folder, "*.JPG"))[:3]
    for f in files:
        img = Image.open(f)
        t_arr = preprocess_image_upgraded(img)
        preds = model.predict(t_arr, verbose=0)[0]
        top = int(np.argmax(preds))
        print(f"  {stain} -> {os.path.basename(f)}: {label_map[top]} ({preds[top]*100:.1f}%)")
