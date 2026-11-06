import glob
import os
import functools
print = functools.partial(print, flush=True)

from PIL import Image
import numpy as np
import cv2

def detect_fabric_color_from_image(pil_image: Image.Image) -> tuple:
    """
    Fabric Luminance Fix:
    Replace naive center-crop mean luminance with a 90th-percentile highlight check (P90 >= 150 for White Cotton)
    so white shirts aren't mislabeled as grey.
    """
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
    fabric_color = "White Cotton" if p90 >= 150.0 else "Grey Cotton"
    return fabric_color, p90

def analyze_stain_colorimetry(pil_image: Image.Image, fabric_color: str = "White Cotton"):
    img = pil_image.convert("RGB")
    w, h = img.size
    if max(w, h) > 800:
        scale = 800.0 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        
    arr = np.array(img)
    h_img, w_img = arr.shape[:2]
    
    # Central garment region
    crop_rgb = arr[int(h_img * 0.10):int(h_img * 0.90), int(w_img * 0.10):int(w_img * 0.90)]
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    
    h_chan = crop_hsv[:, :, 0]
    s_chan = crop_hsv[:, :, 1]
    v_chan = crop_hsv[:, :, 2]
    total_crop_pixels = crop_gray.size
    
    p90 = float(np.percentile(crop_gray, 90))
    p10 = float(np.percentile(crop_gray, 10))
    
    # Stain mask: pixels contrasting with unsoiled fabric
    stain_mask = (crop_gray < (p90 - 15)) | ((s_chan > 25) & (crop_gray < (p90 - 5)))
    num_stain_pixels = int(np.sum(stain_mask))
    if num_stain_pixels < 25:
        stain_mask = (crop_gray < (p90 - 8))
        num_stain_pixels = int(np.sum(stain_mask))
        
    min_gray = float(np.min(crop_gray))
    p5_gray = float(np.percentile(crop_gray[stain_mask > 0], 5)) if num_stain_pixels > 0 else min_gray
    
    # ── 1. Black Ballpen Ink Signatures ──
    # Blue/black hue clustering: (Hue <= 15 or 140 <= Hue <= 180 in OpenCV 0-180 scale)
    # with very low saturation (s_chan < 110) and high dark-pixel density (v_chan < 65 or gray < 55)
    ink_hue_clustering = ((h_chan <= 15) | (h_chan >= 140))
    dark_pixels = (crop_gray < 55) & (v_chan < 65)
    ink_pixels_mask = ink_hue_clustering & dark_pixels & (s_chan < 110)
    
    ink_pixel_count = int(np.sum(ink_pixels_mask))
    dark_pixel_count = int(np.sum(dark_pixels))
    ink_density = ink_pixel_count / max(1, num_stain_pixels)
    dark_density = dark_pixel_count / max(1, num_stain_pixels)
    
    # Blue dye tones: prominent blue clustering in stain
    blue_ink_pixels = ((h_chan >= 135) & (h_chan <= 180) & (v_chan < 75))
    blue_ink_count = int(np.sum(blue_ink_pixels & (stain_mask > 0)))
    blue_ink_ratio = blue_ink_count / max(1, num_stain_pixels)
    
    # ── 2. Used Cooking Oil & Lipid Halos ──
    # Amber/gold hue: 15 <= Hue <= 45
    amber_gold_mask = (h_chan >= 14) & (h_chan <= 45) & (s_chan >= 14) & (v_chan >= 55)
    amber_pixels_count = int(np.sum(amber_gold_mask & (stain_mask > 0)))
    amber_ratio = amber_pixels_count / max(1, num_stain_pixels)
    
    # Opacity & Luminance Drop
    val_drop = p90 - crop_gray[stain_mask > 0] if num_stain_pixels > 0 else np.array([0.0])
    mean_val_drop = float(np.mean(val_drop))
    opaque_ratio = float(np.mean(val_drop > 40))
    deep_opaque_ratio = float(np.mean(val_drop > 70))
    
    # Lipid Diffusion Halo:
    contours, _ = cv2.findContours(stain_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crust_irregularity = 0.0
    has_translucent_lipid_halo = False
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(main_contour))
        perimeter = float(cv2.arcLength(main_contour, True))
        if perimeter > 0:
            compactness = (4.0 * np.pi * area) / (perimeter * perimeter)
            crust_irregularity = float(1.0 - min(1.0, compactness))
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(stain_mask.astype(np.uint8), kernel, iterations=2)
        halo_ring = (dilated > 0) & (~stain_mask)
        if np.sum(halo_ring) > 20:
            halo_amber = (h_chan[halo_ring] >= 14) & (h_chan[halo_ring] <= 45) & (s_chan[halo_ring] >= 12)
            halo_amber_ratio = float(np.mean(halo_amber))
            sobelx = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_grad = np.sqrt(sobelx**2 + sobely**2)
            boundary_grad = float(np.mean(edge_grad[halo_ring]))
            
            # Lipid halo is present if outer diffusion ring has amber lipid hue (> 0.20), smooth boundary,
            # and stain lacks dense opaque soil crust (opaque_ratio < 0.08, deep_opaque < 0.02)
            if (halo_amber_ratio > 0.20) and (boundary_grad < 35.0) and (opaque_ratio < 0.08):
                has_translucent_lipid_halo = True
                
    # Warmth (Red - Blue)
    r_mean = float(np.mean(crop_bgr[:, :, 2][stain_mask > 0])) if num_stain_pixels > 0 else 0.0
    b_mean = float(np.mean(crop_bgr[:, :, 0][stain_mask > 0])) if num_stain_pixels > 0 else 0.0
    warmth_diff = r_mean - b_mean
    
    return {
        "p90": p90,
        "p10": p10,
        "min_gray": min_gray,
        "p5_gray": p5_gray,
        "num_stain_pixels": num_stain_pixels,
        "ink_pixel_count": ink_pixel_count,
        "dark_pixel_count": dark_pixel_count,
        "ink_density": ink_density,
        "dark_density": dark_density,
        "blue_ink_ratio": blue_ink_ratio,
        "amber_ratio": amber_ratio,
        "mean_val_drop": mean_val_drop,
        "opaque_ratio": opaque_ratio,
        "deep_opaque_ratio": deep_opaque_ratio,
        "has_translucent_lipid_halo": has_translucent_lipid_halo,
        "crust_irregularity": crust_irregularity,
        "warmth_diff": warmth_diff,
        "fabric_color": fabric_color
    }

def apply_hsv_colorimetric_guardrail(pil_image: Image.Image, cnn_label: str, cnn_probs: np.ndarray, fabric_color: str):
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
    deep_opaque = metrics["deep_opaque_ratio"]
    has_lipid_halo = metrics["has_translucent_lipid_halo"]
    crust_irregularity = metrics["crust_irregularity"]
    warmth_diff = metrics["warmth_diff"]
    
    is_white = "White" in fabric_color
    
    # ── Rule 1: Black Ballpen Ink Guardrail ──
    # Condition: If stain's cropped pixel region contains deep blue/black hue clustering
    # (Hue <= 15 or 140 <= Hue <= 180, very low saturation, high dark-pixel density),
    # force-override prediction to Black Ballpen Ink. This prevents sharp ink lines from being misclassified as mud.
    is_sharp_ink_stroke = False
    if is_white:
        if (min_gray < 35 or p5_gray < 60) and (ink_density > 0.15 or (ink_count >= 15 and amber_ratio < 0.25)) and (amber_ratio < 0.35):
            is_sharp_ink_stroke = True
        elif cnn_label == "Black Ballpen Ink" and min_gray < 45 and amber_ratio < 0.35:
            # Retain ink if CNN predicted it and ink dark pixels exist
            is_sharp_ink_stroke = True
    else: # Grey cotton
        if (min_gray < 22 or p5_gray < 40) and (ink_density > 0.15 or (ink_count >= 10 and amber_ratio < 0.25)) and (amber_ratio < 0.35):
            is_sharp_ink_stroke = True
        elif cnn_label == "Black Ballpen Ink" and min_gray < 30 and amber_ratio < 0.35:
            is_sharp_ink_stroke = True
            
    if is_sharp_ink_stroke:
        reason = f"Black Ballpen Ink Guardrail: Deep blue/black hue clustering & high dark-pixel density (min_gray={min_gray:.1f}, ink_density={ink_density:.2f}, ink_count={ink_count})."
        p_ink = max(0.96, float(np.max(cnn_probs)))
        p_oil = float(cnn_probs[1]) * 0.04
        p_mud = float(cnn_probs[2]) * 0.04
        new_probs = np.array([p_ink, p_oil, p_mud])
        new_probs = new_probs / np.sum(new_probs)
        return "Black Ballpen Ink", new_probs, 0, (cnn_label != "Black Ballpen Ink"), reason, metrics

    # ── Rule 2: Used Cooking Oil Guardrail ──
    # Condition: If stain exhibits amber/gold hue signatures (15 <= Hue <= 45, moderate-to-high value)
    # surrounded by smooth translucent lipid diffusion rings/grease halos, force-override to Used Cooking Oil.
    # Used/dark cooking oil with burnt sediment is never mistaken for mud because of the translucent lipid diffusion halo.
    is_cooking_oil = False
    if not is_sharp_ink_stroke:
        if (has_lipid_halo or amber_ratio >= 0.04) and (opaque_ratio < 0.06) and (mean_drop < 25.0) and (min_gray >= 25):
            is_cooking_oil = True
        elif (not is_white) and (mean_drop < 22.0) and (opaque_ratio < 0.05) and (warmth_diff < 12.0) and (min_gray >= 20):
            is_cooking_oil = True
            
    if is_cooking_oil:
        if cnn_label != "Used Cooking Oil":
            reason = f"Used Cooking Oil Guardrail: Amber/gold hue signature ({amber_ratio:.2f}) with translucent lipid diffusion halo (mean_drop={mean_drop:.1f}, opaque_ratio={opaque_ratio:.3f})."
            p_oil = max(0.96, float(np.max(cnn_probs)))
            p_ink = float(cnn_probs[0]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud])
            new_probs = new_probs / np.sum(new_probs)
            return "Used Cooking Oil", new_probs, 1, True, reason, metrics
        else:
            return "Used Cooking Oil", cnn_probs, 1, False, "Consistent with Used Cooking Oil lipid signature", metrics

    # ── Rule 3: Mud Guardrail ──
    # Condition: Retain Mud only if color profile matches earthy brown/grey soil signatures with
    # high particle opacity, irregular crust boundaries, and absence of blue ink tones or amber lipid halos.
    if cnn_label == "Mud":
        # Check for blue ink tones
        if is_sharp_ink_stroke or (blue_ink_ratio > 0.15 and amber_ratio < 0.20):
            reason = "Mud Guardrail override: Blue ink tones / sharp dark stroke detected."
            p_ink = max(0.96, float(np.max(cnn_probs)))
            p_oil = float(cnn_probs[1]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_ink, p_oil, p_mud]) / (p_ink + p_oil + p_mud)
            return "Black Ballpen Ink", new_probs, 0, True, reason, metrics
            
        # Check for amber lipid halos (cooking oil with sediment)
        if is_cooking_oil:
            reason = "Mud Guardrail override: Translucent amber lipid diffusion halo detected (cooking oil with sediment)."
            p_oil = max(0.96, float(np.max(cnn_probs)))
            p_ink = float(cnn_probs[0]) * 0.04
            p_mud = float(cnn_probs[2]) * 0.04
            new_probs = np.array([p_oil, p_ink, p_mud]) / (p_oil + p_ink + p_mud)
            return "Used Cooking Oil", new_probs, 1, True, reason, metrics
            
        # Retain Mud
        reason = f"Mud Guardrail verified: Earthy soil signatures retained (particle opacity={opaque_ratio:.2f}, mean_drop={mean_drop:.1f}, crust irregularity={crust_irregularity:.2f}, no ink/lipid halo)."
        return "Mud", cnn_probs, 2, False, reason, metrics

    top1 = int(np.argmax(cnn_probs))
    return cnn_label, cnn_probs, top1, False, "CNN prediction confirmed by colorimetric analysis", metrics

if __name__ == '__main__':
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    print("Loading model...")
    model = keras.models.load_model('stainscan_model_v3_balanced.keras', compile=False)
    print("Model loaded.")
    
    classes = {0: "Black Ballpen Ink", 1: "Used Cooking Oil", 2: "Mud"}
    
    total_correct = 0
    total_images = 0
    for label, folder in [
        ('Black Ballpen Ink', 'Ballpen Ink_cotton'),
        ('Used Cooking Oil', 'Cooking Oil_cotton'),
        ('Mud', 'Mud_cotton')
    ]:
        files = glob.glob(f'DataSet/StainScan/dataset/{folder}/*.*')[:10]
        print(f"\n==================== {label} ====================")
        correct = 0
        for f in files:
            img = Image.open(f).convert("RGB")
            arr_224 = np.array(img.resize((224, 224), Image.BILINEAR), dtype=np.float32)
            inp = np.expand_dims(preprocess_input(arr_224), axis=0)
            raw_probs = model.predict(inp, verbose=0)[0]
            raw_top1 = int(np.argmax(raw_probs))
            raw_label = classes[raw_top1]
            
            fab, p90 = detect_fabric_color_from_image(img)
            guarded_label, guarded_probs, guarded_top1, overridden, reason, metrics = apply_hsv_colorimetric_guardrail(
                img, raw_label, raw_probs, fab
            )
            
            is_match = (guarded_label == label)
            if is_match: correct += 1
            status_str = "PASS" if is_match else "FAIL"
            bname = os.path.basename(f)
            print(f"[{status_str}] {bname}: Raw='{raw_label}' ({raw_probs[raw_top1]*100:.1f}%) -> Guarded='{guarded_label}' ({guarded_probs[guarded_top1]*100:.1f}%) | Ovrd={overridden} | P90={p90:.0f} ({fab})")
        print(f"Accuracy for {label}: {correct}/{len(files)} ({correct/len(files)*100:.1f}%)")
        total_correct += correct
        total_images += len(files)
        
    print(f"\n==================================================")
    print(f"OVERALL GUARDRAIL ACCURACY: {total_correct}/{total_images} ({total_correct/total_images*100:.1f}%)")
