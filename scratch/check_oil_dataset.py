import glob
from PIL import Image
import numpy as np

jpgs = glob.glob("DataSet/StainScan/dataset/Cooking Oil_cotton/*.JPG")
print(f"Total Cooking Oil JPGs: {len(jpgs)}")

dark_stains = 0
light_stains = 0

for p in jpgs[:50]:
    im = Image.open(p).convert("L")
    arr = np.array(im)
    min_val = np.min(arr)
    # If minimum brightness in image is dark (< 100)
    if min_val < 100:
        dark_stains += 1
    else:
        light_stains += 1

print(f"Sample of 50: {dark_stains} have dark pixels (<100), {light_stains} are completely light (>100)")
