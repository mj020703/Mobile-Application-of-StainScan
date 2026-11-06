import glob
import os
from PIL import Image
import numpy as np
import cv2

for label, folder in [
    ('Black Ballpen Ink', 'Ballpen Ink_cotton'),
    ('Used Cooking Oil', 'Cooking Oil_cotton'),
    ('Mud', 'Mud_cotton')
]:
    files = glob.glob(f'DataSet/StainScan/dataset/{folder}/*.*')[:10]
    print(f'================ {label} ================')
    for f in files:
        img = Image.open(f).convert('RGB')
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h_chan, s_chan, v_chan = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Central crop
        h, w = gray.shape
        cg = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        ch = h_chan[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        cs = s_chan[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        cv = v_chan[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        
        p90 = np.percentile(cg, 90)
        p10 = np.percentile(cg, 10)
        dark_ratio = np.mean(cg < 70)
        very_dark = np.mean(cg < 45)
        
        # Blue/dark ink condition: (Hue <= 15 or 140 <= Hue <= 180) with low saturation and high dark density
        # In OpenCV Hue is 0-180.
        ink_pixels = ((ch <= 15) | (ch >= 140)) & (cv < 75) & (cs < 110)
        ink_pixel_ratio = np.mean(ink_pixels)
        
        # Amber/gold condition: 15 <= Hue <= 45, moderate-to-high V (cv >= 80)
        amber_pixels = (ch >= 15) & (ch <= 45) & (cs >= 15) & (cv >= 80)
        amber_ratio = np.mean(amber_pixels)
        
        # Edge / gradient
        lap = cv2.Laplacian(cg, cv2.CV_64F)
        edge_max = float(np.percentile(np.abs(lap), 95))
        
        bname = os.path.basename(f)
        print(f'{bname}: min_g={np.min(cg)}, p10={p10:.0f}, p90={p90:.0f}, dark<45={very_dark*100:.2f}%, ink_px={ink_pixel_ratio*100:.2f}%, amber_px={amber_ratio*100:.2f}%, edge_p95={edge_max:.1f}')
