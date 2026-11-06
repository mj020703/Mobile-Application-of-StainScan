import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load the model
model_path = "stainscan_model_v3_balanced.keras"
model = keras.models.load_model(model_path, compile=False)
print("Model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Classes
CLASS_MAPPING_RAW = {
    0: "ballpen ink_cotton",
    1: "Cooking Oil_cotton",
    2: "Mud_cotton"
}
CLASS_MAPPING_DISPLAY = {
    0: "Black Ballpen Ink",
    1: "Used Cooking Oil",
    2: "Mud"
}

# 2. Load the user image
img_path = "scratch/user_scan_0.jpg"
pil_image = Image.open(img_path).convert("RGB")
print(f"Image size: {pil_image.size}")

# 3. Test fabric color detection
w, h = pil_image.size
crop_box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
cropped = pil_image.crop(crop_box).convert("L")
mean_lum = float(np.mean(np.array(cropped)))
print(f"Fabric Mean Luminance: {mean_lum:.2f} -> {'Grey Cotton' if mean_lum < 165 else 'White Cotton'}")

# 4. Check Preprocessing: CLAHE vs Raw
def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

arr_resized = np.array(pil_image.resize((224, 224), Image.BILINEAR), dtype=np.uint8)

# With CLAHE (as in server.py):
enhanced = apply_clahe(arr_resized)
input_clahe = np.expand_dims(preprocess_input(enhanced.astype(np.float32)), axis=0)
pred_clahe = model.predict(input_clahe, verbose=0)[0]
print("\n--- Predictions WITH CLAHE (current server.py) ---")
for idx, p in enumerate(pred_clahe):
    print(f"  [{idx}] {CLASS_MAPPING_DISPLAY[idx]}: {p*100:.2f}%")

# WITHOUT CLAHE:
input_raw = np.expand_dims(preprocess_input(arr_resized.astype(np.float32)), axis=0)
pred_raw = model.predict(input_raw, verbose=0)[0]
print("\n--- Predictions WITHOUT CLAHE ---")
for idx, p in enumerate(pred_raw):
    print(f"  [{idx}] {CLASS_MAPPING_DISPLAY[idx]}: {p*100:.2f}%")

# Save the CLAHE enhanced image to see what CLAHE did to it!
cv2.imwrite("scratch/clahe_debug.jpg", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
cv2.imwrite("scratch/raw_debug.jpg", cv2.cvtColor(arr_resized, cv2.COLOR_RGB2BGR))
print("\nSaved clahe_debug.jpg and raw_debug.jpg to scratch/")
