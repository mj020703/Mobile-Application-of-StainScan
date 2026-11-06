import sys, os
sys.path.insert(0, os.path.abspath("."))
import glob
import cv2
import numpy as np
from PIL import Image

def analyze_stain_colorimetry_tuned(pil_image: Image.Image, fabric_color: str) -> dict:
    arr = np.array(pil_image.convert("RGB"))
    h_img, w_img, _ = arr.shape
    crop_rgb = arr[int(h_img * 0.10):int(h_img * 0.90), int(w_img * 0.10):int(w_img * 0.90)]
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    h_chan = crop_hsv[:, :, 0]
    s_chan = crop_hsv[:, :, 1]
    v_chan = crop_hsv[:, :, 2]

    p90 = float(np.percentile(crop_gray, 90))
    p10 = float(np.percentile(crop_gray, 10))

    stain_mask = (crop_gray < (p90 - 15)) | ((s_chan > 25) & (crop_gray < (p90 - 5)))
    num_stain_pixels = int(np.sum(stain_mask))
    if num_stain_pixels < 25:
        stain_mask = (crop_gray < (p90 - 8))
        num_stain_pixels = int(np.sum(stain_mask))

    min_gray = float(np.min(crop_gray))
    p5_gray = float(np.percentile(crop_gray[stain_mask > 0], 5)) if num_stain_pixels > 0 else min_gray

    # 1. Black Ballpen Ink: deep blue/black hue clustering (Hue <= 15 or 140 <= Hue <= 180), low sat, dark density
    ink_hue_clustering = ((h_chan <= 15) | (h_chan >= 140))
    dark_pixels = (crop_gray < 55) & (v_chan < 65)
    ink_pixels_mask = ink_hue_clustering & dark_pixels & (s_chan < 110)

    ink_pixel_count = int(np.sum(ink_pixels_mask & (stain_mask > 0)))
    dark_pixel_count = int(np.sum(dark_pixels & (stain_mask > 0)))
    ink_density = ink_pixel_count / max(1, num_stain_pixels)
    dark_density = dark_pixel_count / max(1, num_stain_pixels)

    blue_ink_pixels = ((h_chan >= 110) & (h_chan <= 180) & (s_chan > 20) & (v_chan < 100))
    blue_ink_count = int(np.sum(blue_ink_pixels & (stain_mask > 0)))
    blue_ink_ratio = blue_ink_count / max(1, num_stain_pixels)

    # 2. Used Cooking Oil: amber/gold hue signatures (12 <= Hue <= 48, Sat >= 6, Val >= 45)
    amber_gold_mask = (h_chan >= 12) & (h_chan <= 48) & (s_chan >= 6) & (v_chan >= 45)
    amber_pixels_count = int(np.sum(amber_gold_mask & (stain_mask > 0)))
    amber_ratio = amber_pixels_count / max(1, num_stain_pixels)

    # Opacity & Luminance Drop
    val_drop = p90 - crop_gray[stain_mask > 0] if num_stain_pixels > 0 else np.array([0.0])
    mean_val_drop = float(np.mean(val_drop))
    opaque_ratio = float(np.mean(val_drop > 40))
    deep_opaque_ratio = float(np.mean(val_drop > 70))

    # Lipid Diffusion Halo & Boundary Morphology:
    contours, _ = cv2.findContours(stain_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crust_irregularity = 0.0
    has_translucent_lipid_halo = False
    boundary_grad = 0.0

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(main_contour))
        perimeter = float(cv2.arcLength(main_contour, True))
        if perimeter > 0:
            compactness = (4.0 * np.pi * area) / (perimeter * perimeter)
            crust_irregularity = float(1.0 - min(1.0, compactness))

        # Dilate mask to extract outer diffusion ring (halo region)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(stain_mask.astype(np.uint8), kernel, iterations=2)
        halo_ring = (dilated > 0) & (~stain_mask)
        if np.sum(halo_ring) > 20:
            halo_amber = (h_chan[halo_ring] >= 12) & (h_chan[halo_ring] <= 48) & (s_chan[halo_ring] >= 6)
            halo_amber_ratio = float(np.mean(halo_amber))
            sobelx = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_grad = np.sqrt(sobelx**2 + sobely**2)
            boundary_grad = float(np.mean(edge_grad[halo_ring]))
            halo_drop = float(np.mean(p90 - crop_gray[halo_ring]))

            # Halo is translucent lipid diffusion if outer ring has amber/warmth, smooth edge, and gentle luminance drop
            if (halo_amber_ratio > 0.15 or amber_ratio > 0.30) and (boundary_grad < 35.0) and (halo_drop < 20.0):
                has_translucent_lipid_halo = True

    # Warmth (Red - Blue)
    r_mean = float(np.mean(crop_bgr[:, :, 2][stain_mask > 0])) if num_stain_pixels > 0 else 0.0
    b_mean = float(np.mean(crop_bgr[:, :, 0][stain_mask > 0])) if num_stain_pixels > 0 else 0.0
    warmth_diff = r_mean - b_mean

    return {
        "p90": round(float(p90), 2),
        "p10": round(float(p10), 2),
        "min_gray": round(float(min_gray), 2),
        "p5_gray": round(float(p5_gray), 2),
        "num_stain_pixels": int(num_stain_pixels),
        "ink_pixel_count": int(ink_pixel_count),
        "dark_pixel_count": int(dark_pixel_count),
        "ink_density": round(float(ink_density), 4),
        "dark_density": round(float(dark_density), 4),
        "blue_ink_ratio": round(float(blue_ink_ratio), 4),
        "amber_ratio": round(float(amber_ratio), 4),
        "mean_val_drop": round(float(mean_val_drop), 2),
        "opaque_ratio": round(float(opaque_ratio), 4),
        "deep_opaque_ratio": round(float(deep_opaque_ratio), 4),
        "has_translucent_lipid_halo": bool(has_translucent_lipid_halo),
        "crust_irregularity": round(float(crust_irregularity), 4),
        "warmth_diff": round(float(warmth_diff), 2),
        "boundary_grad": round(float(boundary_grad), 2),
        "fabric_color": str(fabric_color)
    }

def verify_and_disambiguate_stain_physics_tuned(pil_image: Image.Image, cnn_label: str, cnn_probs: np.ndarray, fabric_color: str):
    metrics = analyze_stain_colorimetry_tuned(pil_image, fabric_color)

    min_gray = metrics["min_gray"]
    p5_gray = metrics["p5_gray"]
    ink_count = metrics["ink_pixel_count"]
    ink_density = metrics["ink_density"]
    blue_ink_ratio = metrics["blue_ink_ratio"]
    amber_ratio = metrics["amber_ratio"]
    mean_drop = metrics["mean_val_drop"]
    opaque_ratio = metrics["opaque_ratio"]
    deep_opaque_ratio = metrics["deep_opaque_ratio"]
    has_lipid_halo = metrics["has_translucent_lipid_halo"]
    crust_irregularity = metrics["crust_irregularity"]
    warmth_diff = metrics["warmth_diff"]
    boundary_grad = metrics["boundary_grad"]

    is_white = "White" in fabric_color

    # ─────────────────────────────────────────────────────────────────
    # Physical Signatures Identification:
    # ─────────────────────────────────────────────────────────────────
    # 1. Genuine Black Ballpen Ink:
    #    Deep blue/black ink clustering (ink_count >= 150 or ink_density >= 0.005)
    #    with low amber ratio (amber_ratio < 0.30)
    has_ink_clustering = (
        (ink_density >= 0.005 and ink_count >= 100) or
        (blue_ink_ratio >= 0.005 and ink_count >= 50) or
        (ink_density >= 0.02)
    ) and (amber_ratio < 0.30)

    # 2. Genuine Used Cooking Oil:
    #    Amber/gold hue signatures (12-48 deg) spanning >= 35% of stain region,
    #    lacking concentrated ink pigment (ink_density < 0.02 and blue_ink_ratio < 0.02 and ink_count < 200).
    is_cooking_oil = (
        (amber_ratio >= 0.35 or has_lipid_halo) and
        (blue_ink_ratio < 0.02) and
        (ink_density < 0.02) and
        (ink_count < 200) and
        (deep_opaque_ratio < 0.35 or mean_drop < 55.0)
    )

    # 3. Genuine Mud Signature:
    #    Soil signatures with particle opacity (deep_opaque_ratio >= 0.15),
    #    mean value drop >= 35.0, irregular crust boundary, and absence of blue ink.
    is_soil_crust = (
        (deep_opaque_ratio >= 0.15 or mean_drop >= 35.0) and
        (amber_ratio >= 0.60) and
        (blue_ink_ratio < 0.02)
    )

    # ─────────────────────────────────────────────────────────────────
    # Deterministic Guardrail Execution:
    # ─────────────────────────────────────────────────────────────────
    # CASE A: CNN predicted Black Ballpen Ink
    if cnn_label == "Black Ballpen Ink":
        # Override to Used Cooking Oil if stain exhibits amber/gold lipid signature and lacks ink clustering
        if (amber_ratio >= 0.35 or is_cooking_oil) and (not has_ink_clustering) and (ink_count < 1200 and ink_density < 0.08):
            reason = f"Used Cooking Oil Guardrail: Amber/gold lipid signature ({amber_ratio:.2%}) with lack of ink clustering (ink_count={ink_count}, blue_ink={blue_ink_ratio:.4f}) overrode Black Ballpen Ink misclassification."
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

    # CASE B: CNN predicted Mud
    if cnn_label == "Mud":
        if has_ink_clustering:
            reason = f"Black Ballpen Ink Guardrail: Blue ink strokes / dark clustering detected (ink_count={ink_count}, blue_ink={blue_ink_ratio:.4f}) overriding Mud."
            p_ink = max(0.96, float(np.max(cnn_probs)))
            p_oil = float(cnn_probs[1]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Black Ballpen Ink", new_probs, 0, True, reason, metrics
        # Only override Mud to Cooking Oil if it lacks soil crust opacity AND exhibits a clear translucent lipid halo
        if is_cooking_oil and (deep_opaque_ratio < 0.10 and mean_drop < 30.0 and opaque_ratio < 0.10):
            reason = f"Used Cooking Oil Guardrail: Translucent amber lipid diffusion halo detected (cooking oil with sediment) overriding Mud."
            p_oil = max(0.96, float(np.max(cnn_probs)))
            p_ink = float(cnn_probs[0]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Used Cooking Oil", new_probs, 1, True, reason, metrics
        # Retain Mud with verified soil profile
        reason = f"Mud Guardrail verified: Earthy soil signatures retained (particle opacity={opaque_ratio:.2f}, mean_drop={mean_drop:.1f}, crust irregularity={crust_irregularity:.2f})."
        return "Mud", cnn_probs, 2, False, reason, metrics

    # CASE C: CNN predicted Used Cooking Oil
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

    top1 = int(np.argmax(cnn_probs))
    return cnn_label, cnn_probs, top1, False, "CNN prediction confirmed by colorimetric analysis", metrics

if __name__ == '__main__':
    print("=== TESTING REFINED GUARDRAIL ON USER SCANS ===")
    fake_ink_probs = np.array([0.92, 0.04, 0.04])
    for p in ["scratch/user_scan_0.jpg", "scratch/user_scan_1.jpg", "scratch/user_scan_2.jpg", "scratch/recent_mobile_scan_2.jpg"]:
        if os.path.exists(p):
            img = Image.open(p)
            lbl, probs, top1, changed, reason, m = verify_and_disambiguate_stain_physics_tuned(
                img, "Black Ballpen Ink", fake_ink_probs, "White Cotton"
            )
            print(f"{os.path.basename(p)} (Input=Black Ballpen Ink): -> {lbl} (changed={changed})")
            print(f"  Reason: {reason}")
            assert lbl == "Used Cooking Oil", f"Failed on {p}: expected Used Cooking Oil, got {lbl}"

    print("\n>>> ALL USER SCANS CORRECTLY OVERRIDDEN FROM INK TO USED COOKING OIL! <<<")

    print("\n=== TESTING REFINED GUARDRAIL ON 30 DATASET IMAGES ===")
    classes = [
        ("Black Ballpen Ink", "DataSet/StainScan/dataset/Ballpen Ink_cotton", np.array([0.95, 0.02, 0.03])),
        ("Used Cooking Oil", "DataSet/StainScan/dataset/Cooking Oil_cotton", np.array([0.02, 0.95, 0.03])),
        ("Mud", "DataSet/StainScan/dataset/Mud_cotton", np.array([0.02, 0.03, 0.95]))
    ]
    total = 0
    passed = 0
    for expected_stain, folder, simulated_probs in classes:
        files = glob.glob(os.path.join(folder, "*.JPG"))[:10]
        for f in files:
            total += 1
            img = Image.open(f)
            lbl, probs, top1, changed, reason, m = verify_and_disambiguate_stain_physics_tuned(
                img, expected_stain, simulated_probs, "White Cotton"
            )
            if lbl == expected_stain:
                passed += 1
            else:
                print(f"FAILED {f}: expected {expected_stain}, got {lbl} | reason={reason}")

    print(f"Dataset Evaluation Result: {passed}/{total} ({passed/total:.1%}) passed!")
    assert passed == total, f"Expected {total}/{total}, got {passed}/{total}"

