import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from PIL import Image
import cv2
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

def eval_pil(im):
    arr = np.array(im.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    # CLAHE
    c_arr = apply_clahe(arr)
    inp = np.expand_dims(preprocess_input(c_arr.astype(np.float32)), axis=0)
    preds = model.predict(inp, verbose=0)[0]
    return preds

for name in ["scratch/user_scan_0.jpg", "scratch/crop_center.jpg", "scratch/crop_tight.jpg"]:
    im = Image.open(name)
    p = eval_pil(im)
    print(f"{name}:")
    print(f"  Ballpen: {p[0]*100:.2f}%")
    print(f"  Cooking Oil: {p[1]*100:.2f}%")
    print(f"  Mud: {p[2]*100:.2f}%")
