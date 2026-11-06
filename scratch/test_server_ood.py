import base64
import json
import sys
import os
from io import BytesIO
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.abspath("."))
from server import app

client = app.test_client()

def img_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def check_ood(img, name, expected_rejected=True):
    b64 = img_to_b64(img)
    res = client.post("/predict", json={
        "image": f"data:image/jpeg;base64,{b64}",
        "email": "test@stainscan.com"
    })
    data = res.get_json()
    is_valid = data.get("is_valid_fabric")
    is_classified = data.get("is_classified")
    stain = data.get("stain")
    
    if expected_rejected:
        passed = (is_classified is False) and (stain == "Unidentified Object" or is_valid is False)
    else:
        passed = (is_classified is True) and (is_valid is True)
        
    print(f"{name:<30} -> Result: stain='{stain}', is_classified={is_classified}, is_valid={is_valid} | {'PASS' if passed else 'FAIL'}")
    return passed

print("=" * 70)
print("RUNNING OOD & GUARDRAIL TESTS")
print("=" * 70)

# 1. Clean Blank White Garment
blank_white = Image.new("RGB", (300, 300), (245, 245, 245))
check_ood(blank_white, "Clean White Garment", expected_rejected=True)

# 2. Blue Wall
blue_wall = Image.new("RGB", (300, 300), (30, 80, 220))
check_ood(blue_wall, "Blue Non-Fabric Wall", expected_rejected=True)

# 3. Rug
rug = Image.new("RGB", (300, 300), (180, 50, 50))
d_rug = ImageDraw.Draw(rug)
for x in range(0, 300, 20):
    for y in range(0, 300, 20):
        c = (230, 200, 50) if (x+y)%40 == 0 else (40, 70, 150)
        d_rug.rectangle([x, y, x+15, y+15], fill=c)
check_ood(rug, "Vivid Carpet / Rug", expected_rejected=True)

# 4. User Scan 0 (Must be classified as positive)
user_scan_0 = Image.open("scratch/user_scan_0.jpg")
check_ood(user_scan_0, "User Scan 0 (Ink)", expected_rejected=False)
