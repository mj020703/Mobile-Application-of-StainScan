import os, sys, glob
sys.path.insert(0, os.path.abspath("."))
import cv2, numpy as np
from PIL import Image
from server import analyze_stain_colorimetry

print("=== REAL INK SAMPLES ===")
for f in sorted(glob.glob('DataSet/StainScan/dataset/Ballpen Ink_cotton/*.JPG'))[:5]:
    img = Image.open(f)
    m = analyze_stain_colorimetry(img, 'White Cotton')
    print(f"{f.split('/')[-1]}: ink_count={m['ink_pixel_count']}, ink_density={m['ink_density']:.4f}, dark_density={m['dark_density']:.4f}, min_gray={m['min_gray']}, p5_gray={m['p5_gray']}")

if glob.glob('samples/sample_grey_ink.jpg'):
    img = Image.open('samples/sample_grey_ink.jpg')
    m = analyze_stain_colorimetry(img, 'Grey Cotton')
    print(f"sample_grey_ink.jpg: ink_count={m['ink_pixel_count']}, ink_density={m['ink_density']:.4f}, dark_density={m['dark_density']:.4f}, min_gray={m['min_gray']}, p5_gray={m['p5_gray']}")

print("\n=== USER SCAN (OIL ON GREY) ===")
img = Image.open('scratch/latest_mobile_scan.jpg')
m = analyze_stain_colorimetry(img, 'Grey Cotton')
print(f"latest_mobile_scan.jpg: ink_count={m['ink_pixel_count']}, ink_density={m['ink_density']:.4f}, dark_density={m['dark_density']:.4f}, min_gray={m['min_gray']}, p5_gray={m['p5_gray']}")

print("\n=== SAMPLE GREY OIL ===")
if glob.glob('samples/sample_grey_oil.jpg'):
    img = Image.open('samples/sample_grey_oil.jpg')
    m = analyze_stain_colorimetry(img, 'Grey Cotton')
    print(f"sample_grey_oil.jpg: ink_count={m['ink_pixel_count']}, ink_density={m['ink_density']:.4f}, dark_density={m['dark_density']:.4f}, min_gray={m['min_gray']}, p5_gray={m['p5_gray']}")
