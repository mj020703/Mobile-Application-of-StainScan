import sys, os
sys.path.insert(0, os.path.abspath("."))
from server import db, init_mongodb, analyze_stain_colorimetry, verify_and_disambiguate_stain_physics, model, CLASS_MAPPING_DISPLAY, CLASS_MAPPING_RAW
CLASS_MAPPING = CLASS_MAPPING_DISPLAY
import numpy as np
from PIL import Image
import base64
from io import BytesIO

print("--- Checking MongoDB for latest scans ---")
try:
    if db is None:
        init_mongodb()
    scans = list(db["scans"].find().sort("timestamp", -1).limit(5))
    print(f"Found {len(scans)} recent scans:")
    for i, s in enumerate(scans):
        print(f"[{i}] Time: {s.get('timestamp')} | Stain: {s.get('stain')} | Fabric: {s.get('fabric')} | Email: {s.get('email')} | ID: {s.get('id')}")
        img_b64 = s.get("image", "")
        if img_b64:
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(img_b64)
            save_name = f"scratch/recent_mobile_scan_{i}.jpg"
            with open(save_name, "wb") as f:
                f.write(img_bytes)
            print(f"     Saved image: {save_name} ({len(img_bytes)} bytes)")
except Exception as e:
    print(f"Mongo error: {e}")

print("\n--- Inspecting local user scan images ---")
for p in ["scratch/user_scan_0.jpg", "scratch/user_scan_1.jpg", "scratch/user_scan_2.jpg", "scratch/recent_mobile_scan_0.jpg"]:
    if not os.path.exists(p):
        continue
    img = Image.open(p).convert("RGB")
    print(f"\nAnalyzing: {p} (size={img.size})")
    
    # Run CNN inference
    img_224 = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img_224, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    raw_preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(raw_preds))
    cnn_label = CLASS_MAPPING.get(top_idx, "Unknown")
    probs_dict = {CLASS_MAPPING.get(i, f"Class {i}"): round(float(raw_preds[i])*100, 2) for i in range(len(raw_preds))}
    print(f"  Raw CNN Pred: {cnn_label} ({round(float(raw_preds[top_idx])*100, 2)}%) | Probs: {probs_dict}")
    
    # Run colorimetry
    metrics = analyze_stain_colorimetry(img, "White Cotton")
    print("  Colorimetry metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
        
    # Run guardrail
    lbl, probs, top1, changed, reason, _ = verify_and_disambiguate_stain_physics(img, cnn_label, raw_preds, "White Cotton")
    print(f"  Guardrail Result: {lbl} (changed={changed}) | Reason: {reason}")
