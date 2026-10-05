"""
StainScan v2 — Optimized Retraining Pipeline
==============================================
Improvements over v1 (augment_and_retrain.py):
  1. Enhanced grey-fabric augmentation:
        - Random Brightness, Contrast, Saturation (via OpenCV HSV jitter)
        - Gaussian Blur + additive Gaussian noise
        - Sharper rotation/flip/zoom variants
  2. Focal Loss  (gamma=2, alpha per-class)
  3. Computed class weights  (sklearn compute_class_weight)
  4. Better hyperparams:
        - Phase 2a  LR = 1e-4  |  30 epochs
        - Phase 2b  LR = 1e-5  |  35 epochs
        - EarlyStopping  min_delta = 1e-4
        - ReduceLROnPlateau  factor 0.4
  5. Full sklearn classification report + confusion matrix
  6. Saves .keras + .h5

Output: stainscan_model_v2_optimized.keras / .h5
"""

import os, sys, random, time, warnings
warnings.filterwarnings("ignore")

def _require(mod, hint=""):
    import importlib
    try: return importlib.import_module(mod)
    except ImportError:
        print(f"[ERROR] Missing '{mod}'. {hint}"); sys.exit(1)

_require("rawpy",      "pip install rawpy")
_require("cv2",        "pip install opencv-python-headless")
_require("tensorflow", "pip install tensorflow")
_require("sklearn",    "pip install scikit-learn")

import cv2
import numpy as np
import rawpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageEnhance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR     = r"DataSet/StainScan/dataset"
IMG_SIZE        = (128, 128)
BATCH_SIZE      = 32
TARGET_GREY     = 300
AUGMENTS_PER    = 9
MODEL_H5_OUT    = "stainscan_model_v2_optimized.h5"
MODEL_KERAS_OUT = "stainscan_model_v2_optimized.keras"
SEED            = 42

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

CLASS_FOLDERS = {
    "Ballpen Ink_cotton": os.path.join(DATASET_DIR, "Ballpen Ink_cotton"),
    "Cooking Oil_cotton": os.path.join(DATASET_DIR, "Cooking Oil_cotton"),
    "Mud_cotton":         os.path.join(DATASET_DIR, "Mud_cotton"),
}

# ── PART 1: ENHANCED GREY FABRIC AUGMENTATION ─────────────────────────────────

def _load_cr2(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    return Image.fromarray(rgb)

def _hsv_jitter(bgr, h_shift, s_scale, v_scale):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + h_shift, 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * s_scale,  0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * v_scale,  0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def _add_noise(bgr, sigma=6.0):
    noise = np.random.normal(0, sigma, bgr.shape).astype(np.float32)
    return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _augment_grey(img, idx):
    # Spatial
    angles = [-15,-10,-5,5,10,15,-12,12,-8]
    img = img.rotate(angles[idx%9], resample=Image.BILINEAR, expand=False)
    if idx in (1,3,5,7): img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if idx in (2,4,6,8): img = img.transpose(Image.FLIP_TOP_BOTTOM)
    zooms = [0.85,0.90,0.95,1.05,1.10,1.15,0.80,1.20,0.88]
    z = zooms[idx%9]; w,h = img.size
    if z < 1.0:
        nw,nh = int(w*z),int(h*z)
        c = Image.new("RGB",(w,h),0); c.paste(img.resize((nw,nh),Image.BILINEAR),((w-nw)//2,(h-nh)//2)); img=c
    else:
        cw,ch = int(w/z),int(h/z)
        img = img.crop(((w-cw)//2,(h-ch)//2,(w-cw)//2+cw,(h-ch)//2+ch)).resize((w,h),Image.BILINEAR)
    # Brightness + Contrast
    bf = [0.75,0.82,0.88,0.92,1.0,1.08,1.15,1.22,0.70][idx%9]
    cf = [0.80,0.90,1.05,1.15,0.85,1.20,0.90,1.10,1.25][idx%9]
    img = ImageEnhance.Brightness(img).enhance(bf)
    img = ImageEnhance.Contrast(img).enhance(cf)
    # HSV jitter + Blur + Noise via OpenCV
    arr = np.array(img); bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h_shifts  = [5,-5,10,-10,0,8,-8,12,-12]
    s_scales  = [0.9,1.1,0.85,1.15,1.0,0.8,1.2,0.95,1.05]
    v_scales  = [0.85,0.90,0.95,1.05,1.10,0.80,1.20,0.88,1.15]
    bgr = _hsv_jitter(bgr, h_shifts[idx%9], s_scales[idx%9], v_scales[idx%9])
    blur_ks = [3,3,5,3,0,5,3,5,3][idx%9]
    if blur_ks > 0: bgr = cv2.GaussianBlur(bgr, (blur_ks,blur_ks), 0)
    if idx % 2 == 1: bgr = _add_noise(bgr)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def run_augmentation():
    print("\n"+"="*70+"\n  PHASE 1 — ENHANCED GREY FABRIC AUGMENTATION\n"+"="*70)
    all_done = all(
        len([f for f in os.listdir(fp) if f.startswith("aug_grey_")]) +
        len([f for f in os.listdir(fp) if f.upper().endswith(".CR2")]) >= TARGET_GREY
        for fp in CLASS_FOLDERS.values() if os.path.isdir(fp)
    )
    summary = {}
    if all_done:
        print("  [SKIP] All classes already meet TARGET_GREY.")
        for cls,fp in CLASS_FOLDERS.items():
            files=os.listdir(fp); jpgs=[f for f in files if f.upper().endswith(".JPG")]
            cr2s=[f for f in files if f.upper().endswith(".CR2")]; augs=[f for f in files if f.startswith("aug_grey_")]
            summary[cls]={"white_jpg":len(jpgs)-len(augs),"grey_cr2":len(cr2s),"aug_generated":0,"total":len(jpgs)}
    else:
        for cls,fp in CLASS_FOLDERS.items():
            print(f"\n  [CLASS] {cls}")
            files=os.listdir(fp); jpgs=[f for f in files if f.upper().endswith(".JPG")]
            cr2s=[f for f in files if f.upper().endswith(".CR2")]; augs=[f for f in files if f.startswith("aug_grey_")]
            need=max(0,TARGET_GREY-len(cr2s)-len(augs))
            print(f"  White: {len(jpgs)-len(augs)} | CR2: {len(cr2s)} | Aug: {len(augs)} | Need: {need}")
            generated=0; cr2_paths=[os.path.join(fp,f) for f in cr2s]
            if need>0 and cr2_paths:
                for i in range(need):
                    try:
                        src=cr2_paths[i%len(cr2_paths)]
                        aug=_augment_grey(_load_cr2(src),(i//len(cr2_paths))%AUGMENTS_PER)
                        aug.save(os.path.join(fp,f"aug_grey_{cls[:3].replace(' ','')}_{i:04d}.jpg"),"JPEG",quality=92)
                        generated+=1
                        if generated%30==0 or generated==need: print(f"    {generated}/{need}")
                    except Exception as e: print(f"  [WARN] {e}")
            files_after=os.listdir(fp)
            summary[cls]={"white_jpg":len(jpgs)-len(augs),"grey_cr2":len(cr2s),"aug_generated":generated,
                          "total":len([f for f in files_after if f.upper().endswith(".JPG")])}
    print(f"\n  {'Class':<25} {'White':>8} {'CR2':>6} {'AugGen':>8} {'Total':>8}")
    for cls,info in summary.items():
        print(f"  {cls:<25} {info['white_jpg']:>8} {info['grey_cr2']:>6} {info['aug_generated']:>8} {info['total']:>8}")
    return summary

# ── PART 2: FOCAL LOSS ─────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        ce = -y_true * K.log(y_pred)
        pt = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        return K.mean(K.sum(alpha * K.pow(1.0-pt, gamma) * ce, axis=-1))
    loss_fn.__name__ = f"focal_g{gamma}"
    return loss_fn

# ── PART 3: MODEL ──────────────────────────────────────────────────────────────

def build_model(num_classes, input_shape=(128,128,3)):
    inputs = keras.Input(shape=input_shape, name="input_image")
    base   = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.50, name="drop1")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Dropout(0.40, name="drop2")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.30, name="drop3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="StainScan_v2_Opt"), base

# ── PART 4: JPG-ONLY GENERATOR ────────────────────────────────────────────────

class JPGOnlyGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, **kwargs):
        gen = super().flow_from_directory(directory, **kwargs)
        mask = np.array([f.lower().endswith((".jpg",".jpeg")) for f in gen.filenames])
        gen.filenames=[f for f,ok in zip(gen.filenames,mask) if ok]
        gen.classes=gen.classes[mask]; gen.samples=len(gen.filenames); gen.n=gen.samples
        print(f"  [JPGOnly] {gen.samples} JPGs loaded ({(~mask).sum()} non-JPG skipped)")
        return gen

# ── PART 5: TRAINING ──────────────────────────────────────────────────────────

def run_training(aug_summary):
    print("\n"+"="*70+"\n  PHASE 2 — OPTIMIZED MODEL TRAINING\n"+"="*70)
    train_datagen = JPGOnlyGenerator(
        rescale=1.0/255, validation_split=0.20,
        rotation_range=20, horizontal_flip=True, vertical_flip=True,
        zoom_range=0.20, brightness_range=[0.70,1.30],
        shear_range=0.10, channel_shift_range=25.0,
        width_shift_range=0.10, height_shift_range=0.10, fill_mode="nearest",
    )
    val_datagen = JPGOnlyGenerator(rescale=1.0/255, validation_split=0.20)
    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", seed=SEED)
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", seed=SEED)

    num_classes = len(train_gen.class_indices)
    class_map   = {v:k for k,v in train_gen.class_indices.items()}
    print(f"  Classes: {train_gen.class_indices} | Train: {train_gen.samples} | Val: {val_gen.samples}")

    cw = compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
    class_weight_dict = {i:float(cw[i]) for i in range(num_classes)}
    print(f"  Class weights: {class_weight_dict}")

    model, base_model = build_model(num_classes, IMG_SIZE+(3,))
    model.summary()
    fl = focal_loss(gamma=2.0, alpha=0.25)
    t0 = time.time()

    # Phase 2a
    print("\n  [Phase 2a] HEAD TRAINING  LR=1e-4 | 30 epochs | patience=6")
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=fl, metrics=["accuracy"])
    cb_a = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, min_delta=1e-4, verbose=1),
        ModelCheckpoint(MODEL_H5_OUT, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-7, verbose=1),
    ]
    hist_a = model.fit(train_gen, validation_data=val_gen, epochs=30,
                       callbacks=cb_a, class_weight=class_weight_dict, verbose=1)
    print(f"  Phase 2a done — {(time.time()-t0)/60:.1f} min")

    # Phase 2b
    print("\n  [Phase 2b] FINE-TUNING top-40 layers  LR=1e-5 | 35 epochs | patience=10")
    base_model.trainable = True
    for layer in base_model.layers[:-40]: layer.trainable = False
    print(f"  Unfrozen: {sum(1 for l in base_model.layers if l.trainable)}/{len(base_model.layers)}")
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=fl, metrics=["accuracy"])
    cb_b = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, min_delta=1e-4, verbose=1),
        ModelCheckpoint(MODEL_H5_OUT, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=4, min_lr=1e-8, verbose=1),
    ]
    train_gen.reset(); val_gen.reset(); t1=time.time()
    hist_b = model.fit(train_gen, validation_data=val_gen, epochs=35,
                       callbacks=cb_b, class_weight=class_weight_dict, verbose=1)
    print(f"  Phase 2b done — {(time.time()-t1)/60:.1f} min | Total: {(time.time()-t0)/60:.1f} min")

    merged = {"accuracy":hist_a.history.get("accuracy",[])+hist_b.history.get("accuracy",[]),
              "val_accuracy":hist_a.history.get("val_accuracy",[])+hist_b.history.get("val_accuracy",[]),
              "val_loss":hist_a.history.get("val_loss",[])+hist_b.history.get("val_loss",[])}
    class _H:
        def __init__(self,h): self.history=h
    return model, _H(merged), train_gen, val_gen, class_map

# ── PART 6: EVALUATION ────────────────────────────────────────────────────────

def run_evaluation(model, history, val_gen, class_map, aug_summary):
    print("\n"+"="*70+"\n  PHASE 3 — EVALUATION & REPORTING\n"+"="*70)
    val_accs=history.history.get("val_accuracy",[]); train_accs=history.history.get("accuracy",[])
    val_losses=history.history.get("val_loss",[])
    best_ep=int(np.argmax(val_accs)) if val_accs else 0
    print(f"\n  Best Epoch: {best_ep+1} | Val Acc: {max(val_accs)*100:.2f}% | Val Loss: {val_losses[best_ep]:.6f}")

    val_gen.reset(); all_preds=[]; all_labels=[]
    for _ in range(len(val_gen)):
        xb,yb=next(val_gen)
        all_preds.extend(np.argmax(model.predict(xb,verbose=0),axis=1))
        all_labels.extend(np.argmax(yb,axis=1))
    all_preds=np.array(all_preds); all_labels=np.array(all_labels)
    nc=len(class_map); class_names=[class_map[i] for i in range(nc)]

    print("\n"+"="*70+"\n  CLASSIFICATION REPORT\n"+"="*70)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("="*70+"\n  CONFUSION MATRIX\n"+"="*70)
    hdr="".join(f"  {n[:12]:>14}" for n in class_names)
    print(f"  {'':>22}{hdr}")
    for t in range(nc):
        row="".join(f"  {cm[t,p]:>14}" for p in range(nc))
        print(f"  {class_names[t][:20]:>22}{row}")

    print("\n"+"="*70+"\n  PER-CLASS ACCURACY (target >85%)\n"+"="*70)
    all_ok=True
    for i in range(nc):
        pct=100.0*cm[i,i]/np.sum(all_labels==i) if np.sum(all_labels==i) else 0.0
        flag="OK" if pct>=85.0 else "BELOW TARGET"
        if pct<85.0: all_ok=False
        print(f"  {class_names[i]:<35} {pct:>6.2f}%  [{flag}]")
    print("\n  >>> " + ("All classes >= 85% - TARGET MET!" if all_ok else "Some classes below 85%."))

    # Save .keras
    try:
        model.save(MODEL_KERAS_OUT)
        print(f"\n  Saved .keras : {MODEL_KERAS_OUT}")
    except Exception as e: print(f"  [WARN] .keras save: {e}")
    print(f"  Best .h5 already saved by ModelCheckpoint: {MODEL_H5_OUT}")
    print("\n  Pipeline complete.\n")

if __name__ == "__main__":
    print("\n"+"="*70)
    print(f"  StainScan v2 Optimized | TF {tf.__version__} | GPU: {bool(tf.config.list_physical_devices('GPU'))}")
    print("="*70)
    aug_summary = run_augmentation()
    model, history, train_gen, val_gen, class_map = run_training(aug_summary)
    run_evaluation(model, history, val_gen, class_map, aug_summary)
