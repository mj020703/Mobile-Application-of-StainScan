import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

img_path = "scratch/user_scan_0.jpg"
pil_image = Image.open(img_path).convert("RGB")

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

# Test 128x128
arr128 = np.array(pil_image.resize((128, 128), Image.BILINEAR), dtype=np.uint8)
enh128 = apply_clahe(arr128)
in128_clahe = np.expand_dims(preprocess_input(enh128.astype(np.float32)), axis=0)
in128_raw = np.expand_dims(preprocess_input(arr128.astype(np.float32)), axis=0)

mp = "stainscan_model_v3_high_confidence.keras"
m = keras.models.load_model(mp, compile=False)
print(f"Model: {mp} (Input: {m.input_shape})")

p_clahe = m.predict(in128_clahe, verbose=0)[0]
print("With CLAHE (128x128):")
print(f"  Ballpen: {p_clahe[0]*100:.2f}%")
print(f"  Cooking Oil: {p_clahe[1]*100:.2f}%")
print(f"  Mud: {p_clahe[2]*100:.2f}%")

p_raw = m.predict(in128_raw, verbose=0)[0]
print("Without CLAHE (128x128):")
print(f"  Ballpen: {p_raw[0]*100:.2f}%")
print(f"  Cooking Oil: {p_raw[1]*100:.2f}%")
print(f"  Mud: {p_raw[2]*100:.2f}%")
