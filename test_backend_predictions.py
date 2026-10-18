import base64
from io import BytesIO
import json
import urllib.request
from PIL import Image, ImageDraw

def img_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def send_prediction(pil_img, desc=""):
    b64 = img_to_b64(pil_img)
    payload = json.dumps({
        "image": f"data:image/jpeg;base64,{b64}",
        "email": "test@stainscan.com"
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "http://127.0.0.1:5000/predict",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        return {"error": e.code, "msg": e.read().decode()}

# Test Samples
# 1. Face Simulation
face = Image.new("RGB", (300, 300), (220, 180, 140))
d = ImageDraw.Draw(face)
d.ellipse([80, 90, 110, 120], fill=(50, 40, 30))
d.ellipse([190, 90, 220, 120], fill=(50, 40, 30))
d.polygon([(150, 130), (140, 170), (160, 170)], fill=(180, 130, 100))
d.rectangle([110, 200, 190, 220], fill=(160, 60, 60))

# 2. Rug / Pattern
rug = Image.new("RGB", (300, 300), (180, 50, 50))
d_rug = ImageDraw.Draw(rug)
for x in range(0, 300, 20):
    for y in range(0, 300, 20):
        c = (230, 200, 50) if (x+y)%40 == 0 else (40, 70, 150)
        d_rug.rectangle([x, y, x+15, y+15], fill=c)

# 3. Clean Blank White Garment
blank_white = Image.new("RGB", (300, 300), (245, 245, 245))

# 4. Clean Blank Grey Garment
blank_grey = Image.new("RGB", (300, 300), (128, 128, 128))

# 5. Blue Non-Fabric
blue_wall = Image.new("RGB", (300, 300), (30, 80, 220))

# 6. Real Ink Stain
real_ink = Image.open(r"DataSet/StainScan/dataset/Ballpen Ink_cotton/IMG_2181.JPG")

# 7. Real Oil Stain
real_oil = Image.open(r"DataSet/StainScan/dataset/Cooking Oil_cotton/IMG_1833.JPG")

# 8. Real Mud Stain
real_mud = Image.open(r"DataSet/StainScan/dataset/Mud_cotton/IMG_2492.JPG")

samples = {
    "Non-Stain: Face": face,
    "Non-Stain: Rug / Carpet": rug,
    "Non-Stain: Clean White Shirt": blank_white,
    "Non-Stain: Clean Grey Shirt": blank_grey,
    "Non-Stain: Blue Wall/Surface": blue_wall,
    "Real Stain: Ballpen Ink": real_ink,
    "Real Stain: Cooking Oil": real_oil,
    "Real Stain: Mud": real_mud
}

print("=" * 90)
print(f"{'Input Type':<30} {'Status':<14} {'Stain Output':<22} {'Confidence':<12} {'Caution/Notes'}")
print("-" * 90)

all_passed = True
for name, img in samples.items():
    res = send_prediction(img, name)
    status = res.get("status", "error")
    stain = res.get("stain", "N/A")
    conf = f"{res.get('confidence', 0)}%"
    margin = res.get('margin', 'N/A')
    caution = res.get("caution", "")
    
    is_non_stain = name.startswith("Non-Stain")
    if is_non_stain:
        if status != "Unidentified" or stain != "Unidentified Stain":
            all_passed = False
            flag = "[FAIL - Not Rejected]"
        else:
            flag = "[PASS - Rejected]"
    else:
        if status != "success" or res.get("is_classified") is not True:
            all_passed = False
            flag = "[FAIL - Stained Image Rejected]"
        else:
            flag = "[PASS - Classified]"
            
    print(f"{name:<30} {status:<14} {stain:<22} {conf:<12} {flag} {caution[:35]}")

print("=" * 90)
print(f"Overall Test Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
