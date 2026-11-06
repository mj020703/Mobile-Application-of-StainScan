import sys, os
sys.path.insert(0, os.path.abspath("."))
from server import db, init_mongodb
import json

if db is None:
    init_mongodb()

for scan_id in ["h_20261030134818_128", "h_20261030134747_126", "h_20261030134839_130"]:
    doc = db["scans"].find_one({"id": scan_id})
    if doc:
        # omit image for display
        doc_copy = {k: v for k, v in doc.items() if k != "image"}
        print(f"\n--- Scan {scan_id} ---")
        for k, v in doc_copy.items():
            print(f"  {k}: {v}")
    else:
        print(f"Scan {scan_id} not found in MongoDB")
