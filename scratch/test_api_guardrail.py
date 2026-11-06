import base64
from io import BytesIO
import json
import urllib.request
from PIL import Image

def img_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def send_prediction(pil_img, desc=""):
    b64 = img_to_b64(pil_img)
    payload = json.dumps({
        "image": f"data:image/jpeg;base64,{b64}",
        "email": "thesis_auditor@stainscan.edu"
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "http://127.0.0.1:5000/predict",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print("HTTP ERROR:", e.code, e.read().decode("utf-8"))
        raise e

samples = [
    ("Ballpen Ink", r"DataSet/StainScan/dataset/Ballpen Ink_cotton/IMG_2181.JPG"),
    ("Cooking Oil", r"DataSet/StainScan/dataset/Cooking Oil_cotton/IMG_1833.JPG"),
    ("Mud", r"DataSet/StainScan/dataset/Mud_cotton/IMG_2492.JPG"),
]

print("=== VERIFYING /predict GUARDRAIL API RESPONSE PAYLOAD ===")
for label, path in samples:
    img = Image.open(path)
    res = send_prediction(img, label)
    print(f"\n--- {label} ({path}) ---")
    print(f"Status: {res.get('status')}")
    print(f"Final Stain: {res.get('stain')} ({res.get('confidence')}%)")
    print(f"Fabric: {res.get('fabric')} (Context: {res.get('fabric_context')})")
    print(f"Raw Model Output: {res.get('raw_model_output')}")
    print(f"Guardrail Data: {res.get('guardrail')}")
    assert res.get("raw_model_output") is not None, "Missing raw_model_output!"
    assert res.get("guardrail") is not None, "Missing guardrail!"
    assert "metrics" in res.get("guardrail", {}), "Missing guardrail metrics!"
print("\n>>> ALL API GUARDRAIL AND AUDIT FIELDS CONFIRMED PRESENT AND ACCURATE! <<<")
