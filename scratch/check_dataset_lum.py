import glob
import numpy as np
from PIL import Image

# Check white cotton JPGs
white_lums = []
for p in glob.glob("DataSet/StainScan/dataset/*/*.JPG")[:30]:
    im = Image.open(p).convert("L")
    white_lums.append(float(np.mean(np.array(im))))

print(f"White cotton dataset images mean luminance: min={min(white_lums):.1f}, mean={np.mean(white_lums):.1f}, max={max(white_lums):.1f}")

# Check grey cotton CR2s
import rawpy
grey_lums = []
for p in glob.glob("DataSet/StainScan/dataset/*/*.CR2")[:30]:
    try:
        with rawpy.imread(p) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            im = Image.fromarray(rgb).convert("L")
            grey_lums.append(float(np.mean(np.array(im))))
    except Exception:
        pass

if grey_lums:
    print(f"Grey cotton dataset images mean luminance: min={min(grey_lums):.1f}, mean={np.mean(grey_lums):.1f}, max={max(grey_lums):.1f}")
