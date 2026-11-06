import cv2
import numpy as np
from PIL import Image

im = Image.open("scratch/user_scan_1.jpg").convert("RGB")
arr = np.array(im)
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
h, w = gray.shape
crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
p90 = float(np.percentile(crop, 90))
p75 = float(np.percentile(crop, 75))
p50 = float(np.percentile(crop, 50))
fab_mask = (crop >= (p90 - 20)) & (crop <= (p90 + 5))
lap = cv2.Laplacian(crop, cv2.CV_64F)
lap_var = float(np.var(lap[fab_mask])) if np.sum(fab_mask) > 50 else float(cv2.Laplacian(crop, cv2.CV_64F).var())

print(f"User Scan 1: size={im.size}")
print(f"  P90: {p90}, P75: {p75}, P50: {p50}")
print(f"  lap_var: {lap_var}")
