import os, sys
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
import numpy as np
import cv2
from server import (
    model,
    preprocess_image,
    detect_fabric_color_from_image,
    analyze_stain_colorimetry,
    verify_and_disambiguate_stain_physics,
    CLASS_MAPPING_DISPLAY
)

img_path = "scratch/latest_mobile_scan.jpg"
img = Image.open(img_path)
print(f"Loaded {img_path}: size={img.size}, mode={img.mode}")

# 1. Test Fabric Detection
fabric_detected = detect_fabric_color_from_image(img)
print(f"\n[FABRIC DETECTION] Result: {fabric_detected}")

# Let's see the luminance values in the image
arr = np.array(img.convert("RGB"))
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
h, w = gray.shape
crop = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
p90 = float(np.percentile(crop, 90))
mean_val = float(np.mean(crop))
p50 = float(np.percentile(crop, 50))
print(f"Luminance: P90={p90:.1f}, P50={p50:.1f}, Mean={mean_val:.1f}")

# 2. Test Model Prediction
input_arr = preprocess_image(img)
preds = model.predict(input_arr, verbose=0)[0]
top_idx = int(np.argmax(preds))
cnn_label = CLASS_MAPPING_DISPLAY.get(top_idx, "Unknown")
probs_dict = {CLASS_MAPPING_DISPLAY.get(k): round(float(preds[k])*100, 2) for k in range(len(preds))}
print(f"\n[RAW CNN INFERENCE] Pred: {cnn_label} ({preds[top_idx]*100:.1f}%) | Probs: {probs_dict}")

# 3. Test Colorimetric Analysis
metrics = analyze_stain_colorimetry(img, fabric_detected)
print(f"\n[COLORIMETRIC METRICS]:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# 4. Test Disambiguation / Guardrail
final_label, probs, top1, changed, reason, _ = verify_and_disambiguate_stain_physics(
    img, cnn_label, preds, fabric_detected
)
print(f"\n[GUARDRAIL DECISION]:")
print(f"  Final Label: {final_label} (changed={changed})")
print(f"  Reason: {reason}")
