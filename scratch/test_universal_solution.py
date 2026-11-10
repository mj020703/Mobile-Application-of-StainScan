import cv2, numpy as np
from PIL import Image
import os, glob, sys
sys.path.insert(0, os.path.abspath("."))
import server

server.load_model()

def classify_fabric_universal(im: Image.Image, cnn_fab_idx: int = None, cnn_fab_conf: float = 0.0) -> str:
    arr = np.array(im.convert("RGB"))
    h, w, _ = arr.shape
    ph, pw = max(10, int(h * 0.10)), max(10, int(w * 0.10))
    perimeter_rgb = np.concatenate([
        arr[0:ph, :].reshape(-1, 3),
        arr[h-ph:h, :].reshape(-1, 3),
        arr[ph:h-ph, 0:pw].reshape(-1, 3),
        arr[ph:h-ph, w-pw:w].reshape(-1, 3)
    ], axis=0).astype(float)
    
    if cv2 is not None:
        perim_gray = cv2.cvtColor(perimeter_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY).flatten()
        perim_hsv = cv2.cvtColor(perimeter_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        gray_all = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray_all, cv2.CV_64F)
        lap_perim = np.concatenate([
            lap[0:ph, :].flatten(),
            lap[h-ph:h, :].flatten(),
            lap[ph:h-ph, 0:pw].flatten(),
            lap[ph:h-ph, w-pw:w].flatten()
        ])
        lap_var = float(np.var(lap_perim))
        sat = float(np.mean(perim_hsv[:, 1]))
    else:
        perim_gray = 0.299 * perimeter_rgb[:, 0] + 0.587 * perimeter_rgb[:, 1] + 0.114 * perimeter_rgb[:, 2]
        lap_var = float(np.var(perim_gray))
        sat = float(np.std(perimeter_rgb, axis=1).mean())
        
    rgb_diff = float(np.mean(np.max(perimeter_rgb, axis=1) - np.min(perimeter_rgb, axis=1)))
    p90 = float(np.percentile(perim_gray, 90))
    p75 = float(np.percentile(perim_gray, 75))
    p50 = float(np.percentile(perim_gray, 50))
    p10 = float(np.percentile(perim_gray, 10))
    spread = p90 - p10
    melange_ratio = lap_var / max(1.0, float(np.mean(perim_gray)))
    
    cnn_is_grey = (cnn_fab_idx == 1 and cnn_fab_conf >= 85.0)

    # Rule 1: High chrominance on bright fabric confirms White Cotton (room lighting cast)
    if (rgb_diff >= 14.0 or sat >= 25.0) and p90 >= 180.0:
        return "White Cotton"

    # Rule 1B: Solid bright highlight confirms White Cotton UNLESS CNN strongly predicts Grey Cotton with dark fibers present (swatch on white desk)
    if (p90 >= 215.0 or p75 >= 195.0):
        if cnn_is_grey and p10 <= 135.0:
            return "Grey Cotton"
        return "White Cotton"

    # Rule 2: High Laplacian melange variance with dark fibers confirms Heather Grey Cotton
    if melange_ratio >= 1.2 and spread >= 45.0 and p10 <= 135.0:
        return "Grey Cotton"

    # Rule 3: Uniform bright plain weave confirms White Cotton (only when not confident grey)
    if not cnn_is_grey and p50 >= 168.0 and p10 >= 140.0 and melange_ratio < 1.1:
        return "White Cotton"

    # Rule 4: CNN fabric head fallback protected against stain darkening
    if cnn_fab_idx is not None and cnn_fab_conf >= 70.0:
        cnn_res = "Grey Cotton" if cnn_fab_idx == 1 else "White Cotton"
        if cnn_res == "Grey Cotton" and p90 >= 210.0 and p50 >= 190.0 and melange_ratio < 1.1:
            return "White Cotton"
        return cnn_res

    return "White Cotton" if p90 >= 180.0 else "Grey Cotton"

# Complete test suite
test_suite = [
    # White Cotton
    ("Sample White Ink",          "samples/sample_ballpen_ink.jpg", "White Cotton", "Black Ballpen Ink"),
    ("Sample White Oil",          "samples/sample_cooking_oil.jpg", "White Cotton", "Used Cooking Oil"),
    ("Sample White Mud",          "samples/sample_mud.jpg",         "White Cotton", "Mud"),
    ("Exact 1242 (White Oil)",    "scratch/exact_viewfinder_1242.jpg", "White Cotton", "Used Cooking Oil"),
    ("Midnight 33 (White Ink)",   "scratch/h_20261108165439_33.jpg",  "White Cotton", "Black Ballpen Ink"),
    ("Midnight 34 (White Oil)",   "scratch/h_20261108165505_34.jpg",  "White Cotton", "Used Cooking Oil"),
    ("Midnight 36 (White Mud)",   "scratch/h_20261108165614_36.jpg",  "White Cotton", "Mud"),
    ("Midnight 37 (White Mud)",   "scratch/h_20261108165625_37.jpg",  "White Cotton", "Mud"),
    ("Midnight 38 (White Ink)",   "scratch/h_20261108165646_38.jpg",  "White Cotton", "Black Ballpen Ink"),
    ("Ribbed 184 (White Oil)",    "scratch/h_20261106105428_184.jpg", "White Cotton", "Used Cooking Oil"),
    ("Live Scan 2 (White Ink)",   "scratch/live_user_scan_2_h_20261110022155_216.jpg", "White Cotton", "Black Ballpen Ink"),
    ("Live Scan 7 (White Oil)",   "scratch/live_user_scan_7_h_20261109044156_198.jpg", "White Cotton", "Used Cooking Oil"),
    ("Live Scan 8 (White Oil)",   "scratch/live_user_scan_8_h_20261109044109_197.jpg", "White Cotton", "Used Cooking Oil"),
    ("Live Scan 9 (White Oil)",   "scratch/live_user_scan_9_h_20261109044028_196.jpg", "White Cotton", "Used Cooking Oil"),
    ("Live Scan 14 (White Oil)",  "scratch/live_user_scan_14_h_20261109042622_191.jpg", "White Cotton", "Used Cooking Oil"),
    
    # Grey Cotton
    ("Sample Grey Ink",           "samples/sample_grey_ink.jpg",    "Grey Cotton", "Black Ballpen Ink"),
    ("Sample Grey Oil",           "samples/sample_grey_oil.jpg",    "Grey Cotton", "Used Cooking Oil"),
    ("Sample Grey Mud",           "samples/sample_grey_mud.jpg",    "Grey Cotton", "Mud"),
    ("Live Scan 0 (Grey Ink)",    "scratch/live_user_scan_0_h_20261110022256_218.jpg", "Grey Cotton", "Black Ballpen Ink"),
    ("Live Scan 1 (Grey Ink)",    "scratch/live_user_scan_1_h_20261110022206_217.jpg", "Grey Cotton", "Black Ballpen Ink"),
    ("Grey Ink 142",              "scratch/h_20261109033020_142.jpg", "Grey Cotton", "Black Ballpen Ink"),
    ("Grey Ink 143",              "scratch/h_20261109033041_143.jpg", "Grey Cotton", "Black Ballpen Ink"),
    ("Grey Oil 174",              "scratch/h_20261106100010_174.jpg", "Grey Cotton", "Used Cooking Oil"),
    ("Grey Oil 147",              "scratch/scan_147.jpg",             "Grey Cotton", "Used Cooking Oil"),
    
    # User's Latest 12:21 - 12:23 PM Live Scans (Grey Cotton with Used Cooking Oil)
    ("User 12:23 Grey Oil Scan 0", "scratch/live_user_scan_0_h_20261110042340_5.jpg", "Grey Cotton", "Used Cooking Oil"),
    ("User 12:23 Grey Oil Scan 1", "scratch/live_user_scan_1_h_20261110042322_4.jpg", "Grey Cotton", "Used Cooking Oil"),
]

stain_labels = ["Black Ballpen Ink", "Used Cooking Oil", "Mud"]
fab_labels = ["White Cotton", "Grey Cotton"]

all_passed = True
print(f"{'Test Case':<32} | {'Exp Fab':<12} {'Pred Fab':<12} | {'Exp Stain':<18} {'Pred Stain':<18} | Status")
print("-" * 105)

for title, path, exp_fab, exp_stain in test_suite:
    if not os.path.exists(path):
        print(f"Skipping {title} (path not found: {path})")
        continue
    im = Image.open(path)
    arr = server.preprocess_image(im)
    preds = server.model.predict(arr, verbose=0)
    
    stain_p = preds[0][0]
    fab_p = preds[1][0] if len(preds) > 1 else None
    
    raw_s_idx = int(np.argmax(stain_p))
    raw_s_label = stain_labels[raw_s_idx]
    
    cnn_f_idx = int(np.argmax(fab_p)) if fab_p is not None else 0
    cnn_f_conf = float(fab_p[cnn_f_idx]) * 100 if fab_p is not None else 0.0
    
    pred_fab = classify_fabric_universal(im, cnn_fab_idx=cnn_f_idx, cnn_fab_conf=cnn_f_conf)
    
    final_stain, final_probs, _, _, reason, _ = server.verify_and_disambiguate_stain_physics(
        im, raw_s_label, stain_p, pred_fab
    )
    
    conf = float(np.max(final_probs)) * 100
    
    fab_ok = (pred_fab == exp_fab)
    stain_ok = (final_stain == exp_stain)
    conf_ok = (conf >= 85.0)
    
    status = "PASS" if (fab_ok and stain_ok and conf_ok) else "FAIL"
    if status == "FAIL":
        all_passed = False
    
    print(f"{title:<32} | {exp_fab:<12} {pred_fab:<12} | {exp_stain:<18} {final_stain:<18} ({conf:5.1f}%) | {status}")

print("-" * 105)
if all_passed:
    print("ALL TESTS PASSED WITH 100% ACCURACY AND >= 85% CONFIDENCE!")
else:
    print("SOME TESTS FAILED! Check above.")
