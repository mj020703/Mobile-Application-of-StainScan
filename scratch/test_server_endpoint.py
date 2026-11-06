import base64
import json
from io import BytesIO
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from server import app

def img_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

client = app.test_client()

def test_scan(img_path, label):
    im = Image.open(img_path)
    b64 = img_to_b64(im)
    res = client.post("/predict", json={
        "image": f"data:image/jpeg;base64,{b64}",
        "email": "test@stainscan.com"
    })
    data = res.get_json()
    stain = data.get("stain_classification") or data.get("stain") or data.get("stain_type")
    conf = data.get("confidence")
    probs = data.get("probabilities", {})
    fabric = data.get("fabric") or data.get("fabric_color")
    caution = data.get("caution", "")
    print(f"[{label}]")
    print(f"  Stain: {stain}")
    print(f"  Confidence: {conf}%")
    print(f"  Fabric: {fabric}")
    print(f"  Probabilities: {probs}")
    print(f"  Caution: {caution[:60]}...")
    print()

print("=" * 60)
print("TESTING SERVER /predict ENDPOINT WITH DISAMBIGUATION")
print("=" * 60)

test_scan("scratch/user_scan_0.jpg", "User Scan 0 (Black Ballpen on White Cotton)")
test_scan("scratch/user_scan_1.jpg", "User Scan 1 (Black Ballpen on White Cotton)")
test_scan("DataSet/StainScan/dataset/Ballpen Ink_cotton/IMG_2181.JPG", "Dataset Ballpen Ink (White Cotton)")
test_scan("DataSet/StainScan/dataset/Cooking Oil_cotton/IMG_1833.JPG", "Dataset Cooking Oil (White Cotton)")
test_scan("DataSet/StainScan/dataset/Mud_cotton/IMG_2492.JPG", "Dataset Mud (White Cotton)")
