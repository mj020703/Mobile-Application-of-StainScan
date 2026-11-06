import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import glob
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

def eval_img(p):
    im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = apply_clahe(np.array(im, dtype=np.uint8))
    inp = np.expand_dims(preprocess_input(arr.astype(np.float32)), axis=0)
    preds = model.predict(inp, verbose=0)[0]
    return preds

print("--- Testing Cooking Oil images from Dataset ---")
for p in glob.glob("DataSet/StainScan/dataset/Cooking Oil_cotton/*.JPG")[:5]:
    preds = eval_img(p)
    print(f"{os.path.basename(p)} -> Ink: {preds[0]*100:.1f}%, Oil: {preds[1]*100:.1f}%, Mud: {preds[2]*100:.1f}%")

print("\n--- Testing Mud images from Dataset ---")
for p in glob.glob("DataSet/StainScan/dataset/Mud_cotton/*.JPG")[:5]:
    preds = eval_img(p)
    print(f"{os.path.basename(p)} -> Ink: {preds[0]*100:.1f}%, Oil: {preds[1]*100:.1f}%, Mud: {preds[2]*100:.1f}%")
