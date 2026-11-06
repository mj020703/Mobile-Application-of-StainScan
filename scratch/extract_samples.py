import os
from PIL import Image
import cv2

# Save a small preview of dataset oil and dataset mud
im_oil = Image.open("DataSet/StainScan/dataset/Cooking Oil_cotton/IMG_1833.JPG")
im_oil.resize((300, 300)).save("scratch/sample_dataset_oil.jpg")

im_mud = Image.open("DataSet/StainScan/dataset/Mud_cotton/IMG_2492.JPG")
im_mud.resize((300, 300)).save("scratch/sample_dataset_mud.jpg")

# Also check a few more images in Cooking Oil
for f in ["IMG_1835.JPG", "IMG_1840.JPG", "IMG_1850.JPG"]:
    p = os.path.join("DataSet/StainScan/dataset/Cooking Oil_cotton", f)
    if os.path.exists(p):
        Image.open(p).resize((300, 300)).save(f"scratch/sample_{f}")

print("Saved samples to scratch/")
