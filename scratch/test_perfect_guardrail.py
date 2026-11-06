import os, sys, glob
sys.path.insert(0, os.path.abspath("."))
import cv2, numpy as np
from PIL import Image
from server import analyze_stain_colorimetry

def detect_fabric_color_perfect(pil_image: Image.Image) -> str:
    img = pil_image.convert("RGB")
    w, h = img.size
    if max(w, h) > 800:
        scale = 800.0 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h_crop, w_crop = gray.shape
    crop = gray[int(h_crop * 0.1):int(h_crop * 0.9), int(w_crop * 0.1):int(w_crop * 0.9)]
    
    p90 = float(np.percentile(crop, 90))
    mean_lum = float(np.mean(crop))
    
    # White cotton photos have unsoiled highlight luminance P90 >= 220 and Mean >= 180.
    # Grey cotton garments (even under camera auto-exposure) stay below P90 220 and Mean 180.
    if p90 >= 220.0 and mean_lum >= 180.0:
        return "White Cotton"
    else:
        return "Grey Cotton"

def verify_and_disambiguate_stain_physics_perfect(pil_image: Image.Image, cnn_label: str, cnn_probs: np.ndarray, fabric_color: str):
    metrics = analyze_stain_colorimetry(pil_image, fabric_color)

    min_gray = metrics["min_gray"]
    p5_gray = metrics["p5_gray"]
    ink_count = metrics["ink_pixel_count"]
    dark_count = metrics["dark_pixel_count"]
    ink_density = metrics["ink_density"]
    dark_density = metrics["dark_density"]
    blue_ink_ratio = metrics["blue_ink_ratio"]
    amber_ratio = metrics["amber_ratio"]
    mean_drop = metrics["mean_val_drop"]
    opaque_ratio = metrics["opaque_ratio"]
    deep_opaque_ratio = metrics.get("deep_opaque_ratio", 0.0)
    has_lipid_halo = metrics["has_translucent_lipid_halo"]
    crust_irregularity = metrics["crust_irregularity"]
    warmth_diff = metrics["warmth_diff"]

    is_grey = "Grey" in fabric_color

    # 1. Genuine Black Ballpen Ink:
    # Requires blue/black ink clustering with low amber ratio (amber < 0.35)
    has_ink_clustering = (
        (ink_density >= 0.005 and ink_count >= 100) or
        (blue_ink_ratio >= 0.005 and ink_count >= 50) or
        (ink_density >= 0.02)
    ) and (amber_ratio < 0.35)

    # 2. Genuine Used Cooking Oil:
    oil_amber_thresh = 0.12 if is_grey else 0.35
    is_cooking_oil = (
        (amber_ratio >= oil_amber_thresh or has_lipid_halo or (is_grey and ink_density < 0.002)) and
        (blue_ink_ratio < 0.02) and
        (ink_density < 0.02) and
        (ink_count < 300) and
        (deep_opaque_ratio < 0.35 or mean_drop < 55.0)
    )

    # 3. Genuine Mud Signature:
    is_soil_crust = (
        (deep_opaque_ratio >= 0.15 or mean_drop >= 35.0) and
        (amber_ratio >= 0.60) and
        (blue_ink_ratio < 0.02)
    )

    # Case A: CNN predicted Black Ballpen Ink
    if cnn_label == "Black Ballpen Ink":
        # Override to Used Cooking Oil if:
        # 1) Clear amber/gold lipid signature OR
        # 2) Stain lacks ink clustering (ink_density < 0.002 and dark_density < 0.008)
        is_lacking_ink = (not has_ink_clustering) and (ink_density < 0.002 and dark_density < 0.008)
        if (amber_ratio >= 0.35 or is_cooking_oil or is_lacking_ink) and (not has_ink_clustering) and (ink_count < 1200 and ink_density < 0.08):
            reason = f"Used Cooking Oil Guardrail: Translucent lipid signature ({amber_ratio:.2%}) with lack of ink clustering (ink_count={ink_count}, ink_dens={ink_density:.4f}, blue_ink={blue_ink_ratio:.4f}) overrode Black Ballpen Ink misclassification on {fabric_color}."
            p_oil = max(0.96, float(np.max(cnn_probs)))
            p_ink = float(cnn_probs[0]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Used Cooking Oil", new_probs, 1, True, reason, metrics
        elif has_ink_clustering:
            reason = f"Black Ballpen Ink Guardrail: Deep blue/black ink clustering verified (ink_density={ink_density:.4f}, ink_count={ink_count})."
            return "Black Ballpen Ink", cnn_probs, 0, False, reason, metrics
        else:
            return "Black Ballpen Ink", cnn_probs, 0, False, "CNN prediction confirmed by colorimetric analysis", metrics

    # Case B: CNN predicted Mud
    if cnn_label == "Mud":
        if has_ink_clustering:
            reason = f"Black Ballpen Ink Guardrail: Blue ink strokes / dark clustering detected (ink_count={ink_count}, blue_ink={blue_ink_ratio:.4f}) overriding Mud."
            p_ink = max(0.96, float(np.max(cnn_probs)))
            p_oil = float(cnn_probs[1]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Black Ballpen Ink", new_probs, 0, True, reason, metrics
        if is_cooking_oil and (deep_opaque_ratio < 0.10 and mean_drop < 30.0 and opaque_ratio < 0.10):
            reason = f"Used Cooking Oil Guardrail: Translucent amber lipid diffusion halo detected (cooking oil with sediment) overriding Mud."
            p_oil = max(0.96, float(np.max(cnn_probs)))
            p_ink = float(cnn_probs[0]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Used Cooking Oil", new_probs, 1, True, reason, metrics
        reason = f"Mud Guardrail verified: Earthy soil signatures retained (particle opacity={opaque_ratio:.2f}, mean_drop={mean_drop:.1f}, crust irregularity={crust_irregularity:.2f})."
        return "Mud", cnn_probs, 2, False, reason, metrics

    # Case C: CNN predicted Used Cooking Oil
    if cnn_label == "Used Cooking Oil":
        if has_ink_clustering:
            reason = f"Black Ballpen Ink Guardrail: Ink strokes detected ({ink_count} px) overriding Cooking Oil."
            p_ink = max(0.96, float(np.max(cnn_probs)))
            p_oil = float(cnn_probs[1]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Black Ballpen Ink", new_probs, 0, True, reason, metrics
        if is_soil_crust and (deep_opaque_ratio >= 0.25 and mean_drop >= 45.0 and crust_irregularity >= 0.70) and (not has_lipid_halo):
            reason = f"Mud Guardrail override: High opacity earthy soil crust detected ({amber_ratio:.2%}) overriding Cooking Oil."
            p_mud = max(0.96, float(np.max(cnn_probs)))
            p_ink = float(cnn_probs[0]) * 0.04
            p_oil = float(cnn_probs[1]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Mud", new_probs, 2, True, reason, metrics
        reason = f"Used Cooking Oil verified: Amber/gold lipid signature confirmed ({amber_ratio:.2%})."
        return "Used Cooking Oil", cnn_probs, 1, False, reason, metrics

    return cnn_label, cnn_probs, int(np.argmax(cnn_probs)), False, "Confirmed", metrics

print("=== 1. TESTING ALL 5 USER MOBILE SCANS ===")
user_scans = [
    "scratch/user_scan_0.jpg",
    "scratch/user_scan_1.jpg",
    "scratch/user_scan_2.jpg",
    "scratch/recent_mobile_scan_2.jpg",
    "scratch/latest_mobile_scan.jpg"
]
user_ok = True
for p in user_scans:
    img = Image.open(p)
    fab = detect_fabric_color_perfect(img)
    lbl, probs, top1, changed, reason, m = verify_and_disambiguate_stain_physics_perfect(
        img, "Black Ballpen Ink", np.array([0.96, 0.02, 0.02]), fab
    )
    print(f"  {os.path.basename(p)} [{fab}]: -> {lbl} (changed={changed})")
    if lbl != "Used Cooking Oil":
        user_ok = False
        print(f"    FAILED: got {lbl}")

print(f"User Scans Result: {'ALL PASSED!' if user_ok else 'FAILED!'}")

print("\n=== 2. TESTING 30 DATASET VERIFICATION IMAGES ===")
test_sets = [
    ("Black Ballpen Ink", "DataSet/StainScan/dataset/Ballpen Ink_cotton", np.array([0.95, 0.02, 0.03])),
    ("Used Cooking Oil", "DataSet/StainScan/dataset/Cooking Oil_cotton", np.array([0.02, 0.95, 0.03])),
    ("Mud", "DataSet/StainScan/dataset/Mud_cotton", np.array([0.02, 0.03, 0.95]))
]
ds_passed = 0
ds_total = 0
for expected, folder, probs in test_sets:
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.upper().endswith('.JPG')])[:10]
    for f in files:
        ds_total += 1
        img = Image.open(f)
        fab = detect_fabric_color_perfect(img)
        lbl, new_p, top1, changed, reason, m = verify_and_disambiguate_stain_physics_perfect(
            img, expected, probs, fab
        )
        if lbl == expected:
            ds_passed += 1
        else:
            print(f"    Mismatch: {os.path.basename(f)} expected {expected}, got {lbl} ({reason})")

print(f"Dataset Accuracy: {ds_passed}/{ds_total} ({ds_passed/ds_total*100:.1f}%)")
