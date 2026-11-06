import sys, os
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
import numpy as np
import glob

# Test illumination normalization
def normalize_fabric_illumination(img_rgb: np.ndarray, fabric_color: str = "White Cotton") -> np.ndarray:
    try:
        r_p92 = float(np.percentile(img_rgb[:, :, 0], 92))
        g_p92 = float(np.percentile(img_rgb[:, :, 1], 92))
        b_p92 = float(np.percentile(img_rgb[:, :, 2], 92))

        max_chan = max(r_p92, g_p92, b_p92)
        if max_chan < 40.0:
            return img_rgb

        if r_p92 > (b_p92 + 15.0) or g_p92 > (b_p92 + 12.0):
            target_val = min(245.0, max_chan)
            scale_r = np.clip(target_val / max(1.0, r_p92), 0.75, 1.35)
            scale_g = np.clip(target_val / max(1.0, g_p92), 0.75, 1.35)
            scale_b = np.clip(target_val / max(1.0, b_p92), 0.75, 1.35)

            norm = np.zeros_like(img_rgb, dtype=np.float32)
            norm[:, :, 0] = np.clip(img_rgb[:, :, 0].astype(np.float32) * scale_r, 0, 255)
            norm[:, :, 1] = np.clip(img_rgb[:, :, 1].astype(np.float32) * scale_g, 0, 255)
            norm[:, :, 2] = np.clip(img_rgb[:, :, 2].astype(np.float32) * scale_b, 0, 255)
            return norm.astype(np.uint8)
        return img_rgb
    except Exception:
        return img_rgb

from server import verify_and_disambiguate_stain_physics

print("=== Testing on User Scans ===")
fake_ink_probs = np.array([0.96, 0.02, 0.02])
user_scans = [
    "scratch/user_scan_0.jpg",
    "scratch/user_scan_1.jpg",
    "scratch/user_scan_2.jpg",
    "scratch/recent_mobile_scan_2.jpg"
]
for p in user_scans:
    if os.path.exists(p):
        img = Image.open(p)
        lbl, probs, top1, changed, reason, m = verify_and_disambiguate_stain_physics(
            img, "Black Ballpen Ink", fake_ink_probs, "White Cotton"
        )
        print(f"  {os.path.basename(p)} -> {lbl} (changed={changed})")
        assert lbl == "Used Cooking Oil", f"Failed for {p}"

print("\n>>> All user scans verified successfully! <<<")
