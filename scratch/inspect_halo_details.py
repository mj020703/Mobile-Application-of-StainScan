import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from PIL import Image

for name in ["scratch/user_scan_0.jpg", "scratch/user_scan_1.jpg", "scratch/user_scan_2.jpg", "scratch/recent_mobile_scan_2.jpg"]:
    if not os.path.exists(name):
        continue
    img = Image.open(name).convert("RGB")
    arr = np.array(img)
    h_img, w_img, _ = arr.shape
    crop_rgb = arr[int(h_img * 0.10):int(h_img * 0.90), int(w_img * 0.10):int(w_img * 0.90)]
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = crop_hsv[:, :, 0], crop_hsv[:, :, 1], crop_hsv[:, :, 2]
    
    p90 = float(np.percentile(crop_gray, 90))
    stain_mask = (crop_gray < (p90 - 15)) | ((s_chan > 25) & (crop_gray < (p90 - 5)))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(stain_mask.astype(np.uint8), kernel, iterations=2)
    halo_ring = (dilated > 0) & (~stain_mask)
    
    halo_amber = (h_chan[halo_ring] >= 14) & (h_chan[halo_ring] <= 45) & (s_chan[halo_ring] >= 10)
    halo_amber_ratio = float(np.mean(halo_amber)) if np.sum(halo_ring) > 0 else 0.0
    
    sobelx = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_grad = np.sqrt(sobelx**2 + sobely**2)
    boundary_grad = float(np.mean(edge_grad[halo_ring])) if np.sum(halo_ring) > 0 else 0.0
    halo_val_drop = float(np.mean(p90 - crop_gray[halo_ring])) if np.sum(halo_ring) > 0 else 0.0
    
    print(f"\nImage: {name}")
    print(f"  halo_ring pixels: {np.sum(halo_ring)}")
    print(f"  halo_amber_ratio: {halo_amber_ratio:.4f}")
    print(f"  boundary_grad:    {boundary_grad:.2f}")
    print(f"  halo_val_drop:    {halo_val_drop:.2f}")
