import cv2
import numpy as np
from PIL import Image

def analyze_grey(p):
    im = cv2.imread(p)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    p90 = np.percentile(gray, 90)
    stain_mask = gray < (p90 - 20)
    stain_pixels = im[stain_mask]
    stain_hsv = hsv[stain_mask]
    
    # Red channel vs Blue channel
    r = np.mean(stain_pixels[:, 2])
    g = np.mean(stain_pixels[:, 1])
    b = np.mean(stain_pixels[:, 0])
    
    # Saturation of stain
    sat = np.mean(stain_hsv[:, 1])
    
    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = np.var(lap[stain_mask])
    
    print(f"{p}:")
    print(f"  R: {r:.1f}, G: {g:.1f}, B: {b:.1f}, R-B: {r-b:.1f}, Sat: {sat:.1f}, LapVar: {lap_var:.1f}")

analyze_grey("scratch/user_scan_1.jpg")
analyze_grey("samples/sample_grey_oil.jpg")
analyze_grey("samples/sample_grey_mud.jpg")
