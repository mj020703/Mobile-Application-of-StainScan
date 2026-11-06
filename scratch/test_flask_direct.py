import sys, os
sys.path.insert(0, os.path.abspath("."))
import base64
from io import BytesIO
import json
from PIL import Image
from server import app

def img_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

client = app.test_client()
img = Image.open(r"DataSet/StainScan/dataset/Ballpen Ink_cotton/IMG_2181.JPG")
b64 = img_to_b64(img)

resp = client.post("/predict", json={
    "image": f"data:image/jpeg;base64,{b64}",
    "email": "thesis_auditor@stainscan.edu"
})

print("STATUS CODE:", resp.status_code)
print("DATA:", resp.get_json() or resp.data.decode("utf-8")[:500])
