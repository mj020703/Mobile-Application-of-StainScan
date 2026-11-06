import urllib.request
import json
import base64
import os
from PIL import Image
from io import BytesIO

url = "https://stainscan-backend-gmbw.onrender.com/api/scans?email=johnmichaeldohinog6%40gmail.com&limit=3&include_images=true"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"Fetched {len(data)} scans.")
        for idx, s in enumerate(data):
            print(f"Scan {idx}: id={s.get('id')} timestamp={s.get('timestamp')} stain={s.get('stain')} fabric={s.get('fabric')} conf={s.get('confidence')}")
            img_data = s.get("image", "")
            if img_data:
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                raw_bytes = base64.b64decode(img_data)
                img_path = f"scratch/user_scan_{idx}.jpg"
                os.makedirs("scratch", exist_ok=True)
                with open(img_path, "wb") as f:
                    f.write(raw_bytes)
                print(f"  Saved image to {img_path} ({len(raw_bytes)} bytes)")
except Exception as e:
    print(f"Error: {e}")
