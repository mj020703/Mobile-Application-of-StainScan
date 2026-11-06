import sys, os
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
import numpy as np

from server import verify_and_disambiguate_stain_physics, analyze_stain_colorimetry

print("=== 1. TESTING USER MOBILE SCANS WITH server.py ===")
fake_ink_probs = np.array([0.96, 0.02, 0.02])

user_scans = [
    "scratch/user_scan_0.jpg",
    "scratch/user_scan_1.jpg",
    "scratch/user_scan_2.jpg",
    "scratch/recent_mobile_scan_2.jpg",
    "scratch/latest_mobile_scan.jpg",
    "scratch/scan_147.jpg"
]

all_user_passed = True
for scan_path in user_scans:
    if os.path.exists(scan_path):
        img = Image.open(scan_path)
        from server import detect_fabric_color_from_image
        detected_fab = detect_fabric_color_from_image(img)
        lbl, probs, top1, changed, reason, m = verify_and_disambiguate_stain_physics(
            img, "Black Ballpen Ink", fake_ink_probs, detected_fab
        )
        print(f"\n[SCAN] {os.path.basename(scan_path)} (Fabric: {detected_fab}):")
        print(f"  Input CNN: Black Ballpen Ink -> Guardrail Output: {lbl} (changed={changed})")
        print(f"  Reason: {reason}")
        print(f"  Metrics: P90={m['p90']}, amber_ratio={m['amber_ratio']}, ink_density={m['ink_density']}, opaque_ratio={m['opaque_ratio']}, ink_count={m['ink_pixel_count']}")
        if lbl != "Used Cooking Oil":
            all_user_passed = False
            print(f"  FAILED: expected Used Cooking Oil, got {lbl}")

if all_user_passed:
    print("\n>>> ALL USER MOBILE SCANS CORRECTLY PREDICTED AS USED COOKING OIL! <<<")
else:
    print("\n>>> SOME USER SCANS FAILED! <<<")

print("\n=== 2. TESTING REAL DATASET IMAGES (10 per class) ===")
test_sets = [
    ("Black Ballpen Ink", "DataSet/StainScan/dataset/Ballpen Ink_cotton", np.array([0.95, 0.02, 0.03])),
    ("Used Cooking Oil", "DataSet/StainScan/dataset/Cooking Oil_cotton", np.array([0.02, 0.95, 0.03])),
    ("Mud", "DataSet/StainScan/dataset/Mud_cotton", np.array([0.02, 0.03, 0.95]))
]

dataset_passed = 0
dataset_total = 0
for expected_class, folder, default_probs in test_sets:
    if not os.path.exists(folder):
        continue
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
    for f in files:
        dataset_total += 1
        img = Image.open(f)
        lbl, probs, top1, changed, reason, m = verify_and_disambiguate_stain_physics(
            img, expected_class, default_probs, "White Cotton"
        )
        if lbl == expected_class:
            dataset_passed += 1
        else:
            print(f"  Dataset Mismatch: {f}: expected {expected_class}, got {lbl} ({reason})")

print(f"\nDataset Accuracy: {dataset_passed}/{dataset_total} ({dataset_passed/dataset_total*100:.1f}%)")
