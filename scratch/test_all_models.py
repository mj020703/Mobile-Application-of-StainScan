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
arr_resized = np.array(pil_image.resize((224, 224), Image.BILINEAR), dtype=np.uint8)

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

enhanced = apply_clahe(arr_resized)
input_clahe = np.expand_dims(preprocess_input(enhanced.astype(np.float32)), axis=0)

models_to_test = [
    "stainscan_model_v3_balanced.keras",
    "stainscan_model_v3_high_confidence.keras"
]

for mp in models_to_test:
    if os.path.exists(mp):
        try:
            m = keras.models.load_model(mp, compile=False)
            preds = m.predict(input_clahe, verbose=0)[0]
            print(f"\nModel: {mp} (Input: {m.input_shape})")
            print(f"  Ballpen: {preds[0]*100:.2f}%")
            print(f"  Cooking Oil: {preds[1]*100:.2f}%")
            print(f"  Mud: {preds[2]*100:.2f}%")
        except Exception as e:
            print(f"\nModel: {mp} FAILED: {e}")
