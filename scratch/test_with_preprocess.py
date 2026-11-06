import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image
from server import model, preprocess_image, analyze_stain_colorimetry, verify_and_disambiguate_stain_physics, CLASS_MAPPING_DISPLAY

for name in ["scratch/recent_mobile_scan_2.jpg", "scratch/recent_mobile_scan_4.jpg", "scratch/user_scan_1.jpg", "scratch/user_scan_2.jpg"]:
    if not os.path.exists(name):
        continue
    img = Image.open(name).convert("RGB")
    print(f"\n==========================================")
    print(f"Testing: {name}")
    input_arr = preprocess_image(img)
    preds = model.predict(input_arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    cnn_label = CLASS_MAPPING_DISPLAY.get(top_idx, "Unknown")
    probs_dict = {CLASS_MAPPING_DISPLAY.get(k, f"Class {k}"): round(float(preds[k])*100, 2) for k in range(len(preds))}
    print(f"WITH PREPROCESS_IMAGE (CLAHE):")
    print(f"  Pred: {cnn_label} ({round(float(preds[top_idx])*100, 2)}%) | Probs: {probs_dict}")
    
    lbl, probs, top1, changed, reason, _ = verify_and_disambiguate_stain_physics(img, cnn_label, preds, "White Cotton")
    print(f"  Guardrail Decision: {lbl} (changed={changed}) | Reason: {reason}")
