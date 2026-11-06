import os, sys, json, base64
sys.path.insert(0, os.path.abspath("."))
from server import app

client = app.test_client()

with open("scratch/latest_mobile_scan.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image": f"data:image/jpeg;base64,{img_b64}",
    "fabric_type": "",
    "fabric_color": "",
    "email": "test@stainscan.com"
}

resp = client.post("/predict", json=payload)
data = resp.get_json()

print(f"Status Code: {resp.status_code}")
print(f"Stain Type:  {data.get('stain')}")
print(f"Fabric:      {data.get('fabric')}")
print(f"Confidence:  {data.get('confidence')}%")
print(f"Guardrail:   {data.get('guardrail')}")
print(f"Rec Title:   {data.get('recommendation', {}).get('stain_type')} on {data.get('recommendation', {}).get('fabric_color')}")
