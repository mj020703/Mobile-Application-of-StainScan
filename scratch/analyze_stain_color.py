from PIL import Image
import cv2
import numpy as np

img = cv2.imread("scratch/user_scan_0.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# In user_scan_0.jpg:
# Image size is (300, 169) - height 300, width 169
print("Image shape (H, W, C):", img.shape)

# Let's sample:
# 1. Background cloth (e.g. at y=70, x=40)
bg_bgr = img[70, 40]
bg_hsv = hsv[70, 40]
print(f"Background cloth: BGR={bg_bgr}, HSV={bg_hsv}, Gray={int(np.mean(bg_bgr))}")

# 2. Outer halo (e.g. around y=110, x=80)
# Let's find min/max values inside the stain region
stain_roi = img[70:180, 40:130]
stain_hsv = hsv[70:180, 40:130]

print(f"Stain region min BGR: {np.min(stain_roi, axis=(0,1))}")
print(f"Stain region mean BGR: {np.mean(stain_roi, axis=(0,1))}")
print(f"Stain region max BGR: {np.max(stain_roi, axis=(0,1))}")

# Calculate Hue and Saturation of the stain spots
# Stains are where brightness < 180
mask_stain = cv2.cvtColor(stain_roi, cv2.COLOR_BGR2GRAY) < 180
if np.sum(mask_stain) > 0:
    stain_pixels_bgr = stain_roi[mask_stain]
    stain_pixels_hsv = stain_hsv[mask_stain]
    print(f"Stain pixels count: {len(stain_pixels_bgr)}")
    print(f"Mean Hue (0-179): {np.mean(stain_pixels_hsv[:, 0]):.1f}")
    print(f"Mean Saturation (0-255): {np.mean(stain_pixels_hsv[:, 1]):.1f}")
    print(f"Mean Value (0-255): {np.mean(stain_pixels_hsv[:, 2]):.1f}")
