import numpy as np
from PIL import Image, ImageDraw
import cv2
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = keras.models.load_model('stainscan_model_v3_balanced.keras', compile=False)

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    try:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:
        return img_rgb

CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD     = 0.25

CLASS_NAMES = ["Black Ballpen Ink", "Used Cooking Oil", "Mud"]

def evaluate_stain_image(pil_image: Image.Image):
    # 1. Image checks
    img_rgb = pil_image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img_rgb, dtype=np.uint8)
    
    # Check 1: Contrast / Variation check (Detects completely clean/blank surface or flat background)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    contrast_std = float(np.std(gray))
    if contrast_std < 8.0:
        return {
            "status": "Unidentified",
            "is_classified": False,
            "stain": "Unidentified Stain",
            "reason": f"No visible stain pattern detected (contrast variance {contrast_std:.1f} < 8.0). Garment appears clean or flat.",
            "confidence": int(round(contrast_std * 2)),
            "probabilities": {"Black Ballpen Ink": 0.0, "Used Cooking Oil": 0.0, "Mud": 0.0}
        }
        
    # Check 2: Color distribution / non-fabric saturation check (e.g. vivid colorful rugs, carpets)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    if mean_sat > 90.0:
        return {
            "status": "Unidentified",
            "is_classified": False,
            "stain": "Unidentified Stain",
            "reason": f"Non-fabric color distribution detected (mean saturation {mean_sat:.1f} > 90.0). Not recognized as white or grey cotton.",
            "confidence": 40,
            "probabilities": {"Black Ballpen Ink": 0.0, "Used Cooking Oil": 0.0, "Mud": 0.0}
        }

    # 2. Run CNN inference with CLAHE
    enhanced = apply_clahe(arr)
    norm = preprocess_input(enhanced.astype(np.float32))
    inp = np.expand_dims(norm, axis=0)
    probs = model.predict(inp, verbose=0)[0]
    
    order = np.argsort(probs)[::-1]
    top1_idx, top2_idx = int(order[0]), int(order[1])
    top1_conf = float(probs[top1_idx])
    top2_conf = float(probs[top2_idx])
    margin = top1_conf - top2_conf
    conf_percent = int(round(top1_conf * 100))
    
    prob_dict = {
        "Black Ballpen Ink": round(float(probs[0]) * 100, 2),
        "Used Cooking Oil":  round(float(probs[1]) * 100, 2),
        "Mud":               round(float(probs[2]) * 100, 2)
    }

    # Check 3: Confidence threshold check (e.g. 75%)
    if top1_conf < CONFIDENCE_THRESHOLD:
        return {
            "status": "Unidentified",
            "is_classified": False,
            "stain": "Unidentified Stain",
            "reason": f"Confidence below safety threshold ({conf_percent}% < 75%). Ambiguous scan.",
            "confidence": conf_percent,
            "margin": round(margin * 100, 2),
            "probabilities": prob_dict
        }

    # Check 4: Margin check (top-1 vs top-2 narrow margin indicating uncertainty)
    if margin < MARGIN_THRESHOLD:
        return {
            "status": "Unidentified",
            "is_classified": False,
            "stain": "Unidentified Stain",
            "reason": f"Narrow margin between top classes ({round(margin*100, 1)}% < 25%). Model is uncertain between {CLASS_NAMES[top1_idx]} and {CLASS_NAMES[top2_idx]}.",
            "confidence": conf_percent,
            "margin": round(margin * 100, 2),
            "probabilities": prob_dict
        }

    # Valid positive stain
    return {
        "status": "success",
        "is_classified": True,
        "stain": CLASS_NAMES[top1_idx],
        "confidence": conf_percent,
        "margin": round(margin * 100, 2),
        "probabilities": prob_dict
    }

# Build test fixtures
# 1. Face simulation
face = Image.new("RGB", (300, 300), (220, 180, 140))
d = ImageDraw.Draw(face)
d.ellipse([80, 90, 110, 120], fill=(50, 40, 30))
d.ellipse([190, 90, 220, 120], fill=(50, 40, 30))
d.polygon([(150, 130), (140, 170), (160, 170)], fill=(180, 130, 100))
d.rectangle([110, 200, 190, 220], fill=(160, 60, 60))

# 2. Rug / carpet simulation (intricate non-fabric pattern)
rug = Image.new("RGB", (300, 300), (180, 50, 50))
d_rug = ImageDraw.Draw(rug)
for x in range(0, 300, 20):
    for y in range(0, 300, 20):
        c = (230, 200, 50) if (x+y)%40 == 0 else (40, 70, 150)
        d_rug.rectangle([x, y, x+15, y+15], fill=c)

# 3. Blank / Clean fabrics and solid backgrounds
blank_white = Image.new("RGB", (300, 300), (245, 245, 245))
blank_grey  = Image.new("RGB", (300, 300), (128, 128, 128))
solid_blue  = Image.new("RGB", (300, 300), (30, 80, 220))

# 4. Real Stains
real_ink = Image.open(r"DataSet/StainScan/dataset/Ballpen Ink_cotton/IMG_2181.JPG")
real_oil = Image.open(r"DataSet/StainScan/dataset/Cooking Oil_cotton/IMG_1833.JPG")
real_mud = Image.open(r"DataSet/StainScan/dataset/Mud_cotton/IMG_2492.JPG")

test_suite = {
    "Face Simulation": face,
    "Rug / Pattern": rug,
    "Blank White Garment": blank_white,
    "Blank Grey Garment": blank_grey,
    "Solid Blue Wall/Floor": solid_blue,
    "Real Ink Stain": real_ink,
    "Real Oil Stain": real_oil,
    "Real Mud Stain": real_mud
}

print("=" * 80)
print(f"{'Input Sample':<25} {'Result Status':<15} {'Stain Output':<22} {'Conf':<6} {'Notes'}")
print("-" * 80)
for name, img in test_suite.items():
    res = evaluate_stain_image(img)
    notes = res.get("reason", f"Margin: {res.get('margin', 'N/A')}%")
    print(f"{name:<25} {res['status']:<15} {res['stain']:<22} {res['confidence']}%   {notes[:40]}")
