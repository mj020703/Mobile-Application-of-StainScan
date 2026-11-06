import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image
from server import model, analyze_stain_colorimetry, verify_and_disambiguate_stain_physics, CLASS_MAPPING_DISPLAY

for i in range(5):
    p = f"scratch/recent_mobile_scan_{i}.jpg"
    if not os.path.exists(p):
        continue
    img = Image.open(p).convert("RGB")
    print(f"\n==========================================")
    print(f"SCAN {i}: {p} (size={img.size})")
    
    img_224 = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img_224, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    raw_preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(raw_preds))
    cnn_label = CLASS_MAPPING_DISPLAY.get(top_idx, "Unknown")
    probs_dict = {CLASS_MAPPING_DISPLAY.get(k, f"Class {k}"): round(float(raw_preds[k])*100, 2) for k in range(len(raw_preds))}
    print(f"Raw CNN: {cnn_label} ({round(float(raw_preds[top_idx])*100, 2)}%) | Probs: {probs_dict}")
    
    metrics = analyze_stain_colorimetry(img, "White Cotton")
    print(f"Metrics: min_gray={metrics['min_gray']}, p5={metrics['p5_gray']}, ink_count={metrics['ink_pixel_count']}, dark_count={metrics['dark_pixel_count']}, ink_dens={metrics['ink_density']}, amber_ratio={metrics['amber_ratio']}, opaque_ratio={metrics['opaque_ratio']}, lipid_halo={metrics['has_translucent_lipid_halo']}, mean_val_drop={metrics['mean_val_drop']}")
    
    lbl, probs, top1, changed, reason, _ = verify_and_disambiguate_stain_physics(img, cnn_label, raw_preds, "White Cotton")
    print(f"Final Decision: {lbl} (changed={changed})")
    print(f"Reason: {reason}")
