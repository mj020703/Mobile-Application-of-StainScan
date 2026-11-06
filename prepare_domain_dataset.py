"""
StainScan — Domain Adapted Dataset Builder
==========================================
Loads the 1,800 physical lab images (White Cotton JPG + Grey Cotton CR2)
and augments them with real-world smartphone scans.
Saves to 'dataset_domain_adapted_224.npz'.
"""

import os
import io
import time
import glob
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import rawpy
from PIL import Image

DATASET_DIR = "DataSet/StainScan/dataset"
CACHE_FILE = "dataset_domain_adapted_224.npz"
IMG_SIZE = (224, 224)

CLASSES = [
    "Ballpen Ink_cotton",
    "Cooking Oil_cotton",
    "Mud_cotton"
]

def load_single_image_raw(filepath: str) -> np.ndarray:
    """Loads a JPG or CR2 image, center-crops to square, and resizes to IMG_SIZE (224x224). Returns uint8 RGB."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".cr2":
        with rawpy.imread(filepath) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as img:
                        img = img.convert("RGB")
                        w, h = img.size
                        sz = min(w, h)
                        l = (w - sz) // 2
                        t = (h - sz) // 2
                        img = img.crop((l, t, l + sz, t + sz))
                        img = img.resize(IMG_SIZE, Image.BILINEAR)
                        return np.array(img, dtype=np.uint8)
            except Exception:
                pass
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            img = Image.fromarray(rgb)
            w, h = img.size
            sz = min(w, h)
            l = (w - sz) // 2
            t = (h - sz) // 2
            img = img.crop((l, t, l + sz, t + sz))
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            return np.array(img, dtype=np.uint8)
    else:
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            w, h = img.size
            sz = min(w, h)
            l = (w - sz) // 2
            t = (h - sz) // 2
            img = img.crop((l, t, l + sz, t + sz))
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            return np.array(img, dtype=np.uint8)

def extract_crops_from_mobile(filepath: str, num_crops: int = 8) -> list:
    """Extracts multiple overlapping crops of the stain region from a mobile scan."""
    if not os.path.exists(filepath):
        return []
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        w, h = img.size
        crops = []
        # Center square
        sz = min(w, h)
        l = (w - sz) // 2
        t = (h - sz) // 2
        crops.append(np.array(img.crop((l, t, l + sz, t + sz)).resize(IMG_SIZE, Image.BILINEAR), dtype=np.uint8))
        
        # Tighter center crops (focused on stain core)
        for scale in [0.75, 0.65, 0.55]:
            sub_sz = int(sz * scale)
            sl = (w - sub_sz) // 2
            st = (h - sub_sz) // 2
            crops.append(np.array(img.crop((sl, st, sl + sub_sz, st + sub_sz)).resize(IMG_SIZE, Image.BILINEAR), dtype=np.uint8))
            
            # Slight offsets
            crops.append(np.array(img.crop((sl - 20, st, sl + sub_sz - 20, st + sub_sz)).resize(IMG_SIZE, Image.BILINEAR), dtype=np.uint8))
            crops.append(np.array(img.crop((sl + 20, st, sl + sub_sz + 20, st + sub_sz)).resize(IMG_SIZE, Image.BILINEAR), dtype=np.uint8))
            crops.append(np.array(img.crop((sl, st - 20, sl + sub_sz, st + sub_sz - 20)).resize(IMG_SIZE, Image.BILINEAR), dtype=np.uint8))
            crops.append(np.array(img.crop((sl, st + 20, sl + sub_sz, st + sub_sz + 20)).resize(IMG_SIZE, Image.BILINEAR), dtype=np.uint8))
            
        return crops[:num_crops]

def build_dataset():
    print("=" * 70)
    print("  BUILDING DOMAIN-ADAPTED DATASET FOR STAINSCAN RETRAINING")
    print("=" * 70)
    t0 = time.time()
    
    X, y, substrates = [], [], []  # substrate: 0 = White, 1 = Grey
    stats = {}
    
    for idx, cname in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, cname)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")
            
        all_files = os.listdir(folder)
        white_files = sorted([os.path.join(folder, f) for f in all_files if f.upper().endswith(".JPG") and not f.startswith("aug_")])
        grey_files  = sorted([os.path.join(folder, f) for f in all_files if f.upper().endswith(".CR2")])
        
        print(f"\n[CLASS {idx}] {cname}:")
        print(f"  - White Cotton (.JPG) : {len(white_files)}")
        print(f"  - Grey Cotton (.CR2)  : {len(grey_files)}")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            w_imgs = list(executor.map(load_single_image_raw, white_files))
            g_imgs = list(executor.map(load_single_image_raw, grey_files))
            
        X.extend(w_imgs)
        y.extend([idx] * len(w_imgs))
        substrates.extend([0] * len(w_imgs))
        
        X.extend(g_imgs)
        y.extend([idx] * len(g_imgs))
        substrates.extend([1] * len(g_imgs))
        
        stats[cname] = {
            "white": len(white_files),
            "grey": len(grey_files),
            "total_lab": len(white_files) + len(grey_files)
        }
        
    print(f"\n[LAB IMAGES LOADED] {len(X)} images in {time.time() - t0:.1f}s")
    
    # Inject Real Mobile Captures
    print("\n[MOBILE DOMAIN INJECTION] Adding real smartphone scans...")
    mobile_samples = {
        # Class 1: Cooking Oil
        1: [
            ("scratch/scan_147.jpg", 1, 15),                  # Real smartphone oil on grey cotton
            ("scratch/latest_mobile_scan.jpg", 1, 15),        # Real smartphone oil on grey cotton
            ("scratch/h_20261106075838_162.jpg", 1, 15),     # Real smartphone oil on grey cotton
            ("scratch/h_20261106075932_165.jpg", 0, 15),     # Real smartphone oil on white cotton
            ("scratch/cr2_sample_oil.jpg", 1, 10),            # Additional oil sample
            ("scratch/sample_dataset_oil.jpg", 0, 10),        # Additional oil sample
        ],
        # Class 2: Mud
        2: [
            ("scratch/sample_dataset_mud.jpg", 0, 10),        # Additional mud sample
        ]
    }
    
    added_mobile = 0
    for c_idx, sample_list in mobile_samples.items():
        for fpath, sub_type, n_crops in sample_list:
            if os.path.exists(fpath):
                crops = extract_crops_from_mobile(fpath, n_crops)
                X.extend(crops)
                y.extend([c_idx] * len(crops))
                substrates.extend([sub_type] * len(crops))
                added_mobile += len(crops)
                print(f"  + Added {len(crops)} crops from real scan: {fpath} (Class {c_idx})")
                
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)
    substrates = np.array(substrates, dtype=np.int32)
    
    print(f"\n[DATASET READY] Total Images: {len(X)} (Lab: 1800, Real Mobile: {added_mobile})")
    print(f"  - Shape: {X.shape}, dtype: {X.dtype}")
    print(f"  - Class distribution: {np.bincount(y)}")
    
    print(f"\n[CACHE] Saving compressed dataset to '{CACHE_FILE}'...")
    np.savez_compressed(CACHE_FILE, X=X, y=y, substrates=substrates, classes=CLASSES, stats=stats)
    print(f"[CACHE] Saved successfully in {time.time() - t0:.1f}s. File size: {os.path.getsize(CACHE_FILE) / 1024 / 1024:.2f} MB")
    
if __name__ == "__main__":
    build_dataset()
