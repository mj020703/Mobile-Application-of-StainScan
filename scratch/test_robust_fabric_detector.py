import sys, os
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
import numpy as np
import cv2
import server

server.load_model()

all_scans = [
    # Grey Cotton scans (Ground truth: GREY)
    ("User Mud 11:22",         "scratch/live_user_scan_4_h_20261111032214_25.jpg", "Grey Cotton"),
    ("User Mud 11:23",         "scratch/live_user_scan_3_h_20261111032310_26.jpg", "Grey Cotton"),
    ("User Mud 11:28",         "scratch/live_user_scan_0_h_20261111032759_29.jpg", "Grey Cotton"),
    ("Grey Ink 142",           "scratch/h_20261109033020_142.jpg",                 "Grey Cotton"),
    ("Grey Ink 143",           "scratch/h_20261109033041_143.jpg",                 "Grey Cotton"),
    ("Grey Oil 174",           "scratch/h_20261106100010_174.jpg",                 "Grey Cotton"),
    ("Grey Oil 147",           "scratch/scan_147.jpg",                             "Grey Cotton"),
    # White Cotton scans (Ground truth: WHITE)
    ("White Ink dots 33",      "scratch/h_20261108165439_33.jpg",                  "White Cotton"),
    ("White Ink patch 38",     "scratch/h_20261108165646_38.jpg",                  "White Cotton"),
    ("White Oil 34",           "scratch/h_20261108165505_34.jpg",                  "White Cotton"),
    ("White Ribbed 184",       "scratch/h_20261106105428_184.jpg",                 "White Cotton"),
    ("White Mud 36",           "scratch/h_20261108165614_36.jpg",                  "White Cotton"),
    ("White Mud 37",           "scratch/h_20261108165625_37.jpg",                  "White Cotton"),
]

def robust_detect_fabric(pil_image, cnn_fab_idx, cnn_fab_conf):
    arr = np.array(pil_image.convert("RGB"))
    h, w, _ = arr.shape
    ph = max(10, int(h * 0.10))
    pw = max(10, int(w * 0.10))

    perimeter_rgb = np.concatenate([
        arr[0:ph, :].reshape(-1, 3),
        arr[h-ph:h, :].reshape(-1, 3),
        arr[ph:h-ph, 0:pw].reshape(-1, 3),
        arr[ph:h-ph, w-pw:w].reshape(-1, 3)
    ], axis=0).astype(float)

    perimeter_gray = cv2.cvtColor(perimeter_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY).flatten()
    gray_all = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray_all, cv2.CV_64F)
    perimeter_lap = np.concatenate([
        lap[0:ph, :].flatten(),
        lap[h-ph:h, :].flatten(),
        lap[ph:h-ph, 0:pw].flatten(),
        lap[ph:h-ph, w-pw:w].flatten()
    ])
    lap_var = float(np.var(perimeter_lap))
    rgb_diff = float(np.mean(np.max(perimeter_rgb, axis=1) - np.min(perimeter_rgb, axis=1)))
    p90 = float(np.percentile(perimeter_gray, 90))
    p75 = float(np.percentile(perimeter_gray, 75))
    p50 = float(np.percentile(perimeter_gray, 50))
    p10 = float(np.percentile(perimeter_gray, 10))
    spread = p90 - p10
    melange_ratio = lap_var / max(1.0, float(np.mean(perimeter_gray)))

    cnn_is_grey = (cnn_fab_idx == 1 and cnn_fab_conf >= 80.0)
    cnn_is_white = (cnn_fab_idx == 0 and cnn_fab_conf >= 80.0)

    # 1. If CNN is confident in Grey Cotton and texture possesses heather melange or grey luminance, confirm Grey Cotton
    if cnn_is_grey and (melange_ratio >= 1.2 or lap_var >= 200.0 or p50 <= 180.0):
        return "Grey Cotton"

    # 2. If CNN is confident in White Cotton and texture does NOT have heather melange, confirm White Cotton
    if cnn_is_white and (melange_ratio < 1.2 or p90 >= 210.0):
        return "White Cotton"

    # 3. High chrominance on bright fabric confirms White Cotton (room lighting cast)
    if rgb_diff >= 14.0 and p90 >= 180.0 and not cnn_is_grey:
        return "White Cotton"

    # 4. Pure solid bright highlight confirms White Cotton if melange is low
    if p90 >= 215.0 and melange_ratio < 1.1:
        return "White Cotton"

    # 5. Heather melange variance
    if melange_ratio >= 1.2 and spread >= 45.0 and p10 <= 135.0:
        return "Grey Cotton"

    # 6. CNN fallback
    if cnn_fab_idx is not None and cnn_fab_conf >= 70.0:
        return "Grey Cotton" if cnn_fab_idx == 1 else "White Cotton"

    return "White Cotton" if p90 >= 180.0 else "Grey Cotton"

print("="*85)
print("TESTING ROBUST FABRIC DETECTION ACROSS ALL REAL-WORLD SCANS")
print("="*85)

all_pass = True
for name, path, exp in all_scans:
    if not os.path.exists(path):
        continue
    im = Image.open(path)
    arr = server.preprocess_image(im)
    preds = server.model.predict(arr, verbose=0)
    fab_p = preds[1] if isinstance(preds, (list, tuple)) and len(preds) > 1 else None
    idx = int(np.argmax(fab_p[0]))
    conf = float(fab_p[0][idx]) * 100

    det = robust_detect_fabric(im, idx, conf)
    ok = (det == exp)
    if not ok:
        all_pass = False
    print(f"{name:<20} | Expected: {exp:<12} | Detected: {det:<12} [{'PASS' if ok else 'FAIL'}] (CNN={conf:.1f}%)")

print("="*85)
print(f"ALL TESTS PASSED: {all_pass}\n")
