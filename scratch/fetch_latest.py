import urllib.request, json, base64

url = 'https://stainscan-backend-gmbw.onrender.com/api/scans?limit=5&include_images=true'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f'Fetched {len(data)} scans.')
        for i, s in enumerate(data):
            print(f"Scan {i}: id={s.get('id')} timestamp={s.get('timestamp')} stain={s.get('stain')} fabric={s.get('fabric')} conf={s.get('confidence')}")
            img_b64 = s.get('image', '')
            if img_b64 and i == 0:
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                with open('scratch/latest_mobile_scan.jpg', 'wb') as f:
                    f.write(base64.b64decode(img_b64))
                print('Saved scratch/latest_mobile_scan.jpg')
except Exception as e:
    print('Error:', e)
