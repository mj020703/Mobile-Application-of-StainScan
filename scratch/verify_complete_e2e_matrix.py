import cv2
import numpy as np
from PIL import Image
import os, glob
import sys
sys.path.insert(0, os.path.abspath("."))
import server

server.load_model()

# Ground truth dataset with both Stain and Fabric
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
    
    # User's Latest 12:23 PM Live Scans (Grey Cotton with Used Cooking Oil)
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
    
    pred_fab = server.detect_fabric_color_from_image(im, cnn_fab_idx=cnn_f_idx, cnn_fab_conf=cnn_f_conf)
    
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
