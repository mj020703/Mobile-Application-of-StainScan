import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import cv2
from PIL import Image
from server import model, analyze_stain_colorimetry, verify_and_disambiguate_stain_physics, CLASS_MAPPING_DISPLAY

img = Image.open("scratch/recent_mobile_scan_2.jpg").convert("RGB")
print(f"Loaded scratch/recent_mobile_scan_2.jpg, size={img.size}")

img_224 = img.resize((224, 224), Image.Resampling.BILINEAR)
arr = np.array(img_224, dtype=np.float32) / 255.0
arr = np.expand_dims(arr, axis=0)
raw_preds = model.predict(arr, verbose=0)[0]
top_idx = int(np.argmax(raw_preds))
cnn_label = CLASS_MAPPING_DISPLAY.get(top_idx, "Unknown")
probs_dict = {CLASS_MAPPING_DISPLAY.get(i, f"Class {i}"): round(float(raw_preds[i])*100, 2) for i in range(len(raw_preds))}
print(f"Raw CNN Pred: {cnn_label} ({round(float(raw_preds[top_idx])*100, 2)}%) | Probs: {probs_dict}")

metrics = analyze_stain_colorimetry(img, "White Cotton")
print("Colorimetry metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

lbl, probs, top1, changed, reason, _ = verify_and_disambiguate_stain_physics(img, cnn_label, raw_preds, "White Cotton")
print(f"\nGuardrail Result: {lbl} (changed={changed})")
print(f"Reason: {reason}")
