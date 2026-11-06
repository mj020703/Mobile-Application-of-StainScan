import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image
from server import verify_and_disambiguate_stain_physics

fake_cnn_probs = np.array([0.90, 0.05, 0.05])  # CNN wrongly predicted Black Ballpen Ink!

for name in ["scratch/user_scan_0.jpg", "scratch/user_scan_1.jpg", "scratch/user_scan_2.jpg", "scratch/recent_mobile_scan_2.jpg"]:
    if not os.path.exists(name):
        continue
    img = Image.open(name).convert("RGB")
    lbl, probs, top1, changed, reason, metrics = verify_and_disambiguate_stain_physics(
        img, "Black Ballpen Ink", fake_cnn_probs, "White Cotton"
    )
    print(f"\nImage: {name}")
    print(f"  Result: {lbl} (changed={changed})")
    print(f"  Reason: {reason}")
    print(f"  Metrics: ink_count={metrics['ink_pixel_count']}, ink_dens={metrics['ink_density']}, blue_ink_ratio={metrics['blue_ink_ratio']}, amber_ratio={metrics['amber_ratio']}, min_gray={metrics['min_gray']}, opaque_ratio={metrics['opaque_ratio']}, lipid_halo={metrics['has_translucent_lipid_halo']}")
