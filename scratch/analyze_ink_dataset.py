import sys, os
sys.path.insert(0, os.path.abspath("."))
import glob
from PIL import Image
from server import analyze_stain_colorimetry

ink_files = glob.glob(r"DataSet/StainScan/dataset/Ballpen Ink_cotton/*.JPG")[:10]
print(f"Analyzing {len(ink_files)} real Ink dataset images:")
for f in ink_files:
    img = Image.open(f)
    m = analyze_stain_colorimetry(img, "White Cotton")
    print(f"{os.path.basename(f)}: ink_count={m['ink_pixel_count']}, ink_dens={m['ink_density']}, blue_ink_ratio={m['blue_ink_ratio']}, amber_ratio={m['amber_ratio']}, min_gray={m['min_gray']}, p5={m['p5_gray']}, opaque_ratio={m['opaque_ratio']}")
