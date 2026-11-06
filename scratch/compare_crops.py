from PIL import Image
import os
import cv2
import numpy as np

img_user = Image.open("scratch/user_scan_0.jpg")
print(f"User image size: {img_user.size}")

img_oil = Image.open("DataSet/StainScan/dataset/Cooking Oil_cotton/IMG_1833.JPG")
print(f"Dataset oil size: {img_oil.size}")

img_mud = Image.open("DataSet/StainScan/dataset/Mud_cotton/IMG_2492.JPG")
print(f"Dataset mud size: {img_mud.size}")

# Let's crop just the white fabric with the stain from user_scan_0.jpg!
# In user_scan_0.jpg, the white cloth is from roughly (3, 7) to (155, 230)
# Let's test various crops of user_scan_0.jpg!
w, h = img_user.size
# Crop 1: centered crop of the stain
crop_center = img_user.crop((int(w*0.1), int(h*0.15), int(w*0.9), int(h*0.75)))
crop_center.save("scratch/crop_center.jpg")

# Crop 2: tight crop on the stain spots
crop_tight = img_user.crop((int(w*0.2), int(h*0.25), int(w*0.8), int(h*0.65)))
crop_tight.save("scratch/crop_tight.jpg")

print("Saved crops to scratch/")
