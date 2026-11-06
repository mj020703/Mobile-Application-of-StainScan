import glob
import cv2
import numpy as np
import rawpy
from PIL import Image

grey_files = glob.glob("DataSet/StainScan/dataset/*/*.CR2")[:5]
for p in grey_files:
    try:
        with rawpy.imread(p) as raw:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                arr = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            p90 = np.percentile(gray, 90)
            print(f"Grey CR2: {p} -> P90: {p90:.1f}")
    except Exception as e:
        print(f"Error {p}: {e}")
