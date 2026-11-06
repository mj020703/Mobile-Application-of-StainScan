import glob
from PIL import Image
import numpy as np

# Check how stains look in Cooking Oil_cotton
oil_jpgs = glob.glob("DataSet/StainScan/dataset/Cooking Oil_cotton/*.JPG")
mud_jpgs = glob.glob("DataSet/StainScan/dataset/Mud_cotton/*.JPG")

print(f"Cooking oil images: {len(oil_jpgs)}")
print(f"Mud images: {len(mud_jpgs)}")

# Let's inspect 10 random Cooking Oil images
import random
random.seed(42)
sample_oil = random.sample(oil_jpgs, 5)
for p in sample_oil:
    im = Image.open(p)
    print(f"Oil {p}: size={im.size}")

sample_mud = random.sample(mud_jpgs, 5)
for p in sample_mud:
    im = Image.open(p)
    print(f"Mud {p}: size={im.size}")
