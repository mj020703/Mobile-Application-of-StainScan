"""
StainScan Hybrid Dataset — Augmentation + Retraining Pipeline
=============================================================
Pipeline:
  1. Offline augmentation for grey fabric (CR2 RAW) images
     - Skipped automatically if target count already met
  2. Transfer-learning model training (MobileNetV2 + custom head)
     Phase 2a: Train classification head only (frozen base, LR=1e-3)
     Phase 2b: Fine-tune top-30 MobileNetV2 layers (LR=1e-5)
  3. Validation reporting: confusion matrix, per-class accuracy, image counts

Saved model: stainscan_model_v2_hybrid.h5
"""

import os
import sys
import random
import time
import warnings
warnings.filterwarnings("ignore")

# ── Dependency check ──────────────────────────────────────────────────────────
def _require(module_name, hint=""):
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"[ERROR] Missing module '{module_name}'. {hint}")
        sys.exit(1)

_require("rawpy",      "Install: pip install rawpy")
_require("cv2",        "Install: pip install opencv-python-headless")
_require("tensorflow", "Install: pip install tensorflow")

from PIL import Image, ImageEnhance
import numpy as np
import rawpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR  = r"DataSet/StainScan/dataset"
IMG_SIZE     = (128, 128)     # fast on CPU; sufficient for 3-class stain CNN
BATCH_SIZE   = 64
TARGET_GREY  = 300            # target grey images per class
AUGMENTS_PER = 9              # augmentation variants per source CR2
MODEL_OUT    = "stainscan_model_v2_hybrid.h5"
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_FOLDERS = {
    "Ballpen Ink_cotton": os.path.join(DATASET_DIR, "Ballpen Ink_cotton"),
    "Cooking Oil_cotton": os.path.join(DATASET_DIR, "Cooking Oil_cotton"),
    "Mud_cotton":         os.path.join(DATASET_DIR, "Mud_cotton"),
}

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — OFFLINE GREY FABRIC AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_cr2(path: str) -> Image.Image:
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    return Image.fromarray(rgb)


def _augment(img: Image.Image, idx: int) -> Image.Image:
    img = img.copy()
    # Rotation +-15 deg
    angles = [-15, -10, -5, 0, 5, 10, 15, -12, 12]
    img = img.rotate(angles[idx % 9], resample=Image.BILINEAR, expand=False)
    # Flip
    if idx in (1, 3, 5, 7): img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if idx in (2, 4, 6, 8): img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # Zoom
    zooms = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 0.80, 1.20]
    z = zooms[idx % 9]
    w, h = img.size
    if z < 1.0:
        nw, nh = int(w*z), int(h*z)
        canvas = Image.new("RGB", (w, h), 0)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((w-nw)//2, (h-nh)//2))
        img = canvas
    else:
        cw, ch = int(w/z), int(h/z)
        img = img.crop(((w-cw)//2, (h-ch)//2, (w-cw)//2+cw, (h-ch)//2+ch))
        img = img.resize((w, h), Image.BILINEAR)
    # Brightness & contrast
    bfs = [0.80, 0.85, 0.90, 1.0, 1.05, 1.10, 1.15, 1.20, 0.95]
    cfs = [0.85, 0.90, 1.0,  1.10, 1.15, 0.90, 1.05, 0.85, 1.10]
    img = ImageEnhance.Brightness(img).enhance(bfs[idx % 9])
    img = ImageEnhance.Contrast(img).enhance(cfs[idx % 9])
    return img


def run_augmentation():
    print("\n" + "=" * 65)
    print("  PHASE 1 -- OFFLINE GREY FABRIC AUGMENTATION")
    print("=" * 65)

    # Fast skip: check if all classes already meet target
    all_done = all(
        len([f for f in os.listdir(fp) if f.startswith("aug_grey_")]) +
        len([f for f in os.listdir(fp) if f.upper().endswith(".CR2")]) >= TARGET_GREY
        for fp in CLASS_FOLDERS.values() if os.path.isdir(fp)
    )

    summary = {}

    if all_done:
        print("  [SKIP] All classes already meet TARGET_GREY. No augmentation needed.")
        for cls, fp in CLASS_FOLDERS.items():
            files   = os.listdir(fp)
            jpgs    = [f for f in files if f.upper().endswith(".JPG")]
            cr2s    = [f for f in files if f.upper().endswith(".CR2")]
            augs    = [f for f in files if f.startswith("aug_grey_")]
            summary[cls] = {"white_jpg": len(jpgs)-len(augs), "grey_cr2": len(cr2s),
                            "aug_generated": 0, "total": len(jpgs)}
    else:
        for cls, fp in CLASS_FOLDERS.items():
            print(f"\n[CLASS] {cls}")
            files  = os.listdir(fp)
            jpgs   = [f for f in files if f.upper().endswith(".JPG")]
            cr2s   = [f for f in files if f.upper().endswith(".CR2")]
            augs   = [f for f in files if f.startswith("aug_grey_")]
            need   = max(0, TARGET_GREY - len(cr2s) - len(augs))
            print(f"  White JPGs: {len(jpgs)-len(augs)} | Grey CR2s: {len(cr2s)} | "
                  f"Existing aug: {len(augs)} | Need: {need}")

            generated = 0
            cr2_paths = [os.path.join(fp, f) for f in cr2s]
            if need > 0 and cr2_paths:
                print(f"  Generating {need} augmented images...")
                for i in range(need):
                    try:
                        src = cr2_paths[i % len(cr2_paths)]
                        aug = _augment(_load_cr2(src), (i // len(cr2_paths)) % AUGMENTS_PER)
                        out = os.path.join(fp, f"aug_grey_{cls[:3].replace(' ','')}_{i:04d}.jpg")
                        aug.save(out, "JPEG", quality=92)
                        generated += 1
                        if generated % 30 == 0 or generated == need:
                            print(f"    Progress: {generated}/{need}")
                    except Exception as e:
                        print(f"  [WARN] {e}")
                print(f"  Done. Generated {generated} images.")

            files_after = os.listdir(fp)
            summary[cls] = {
                "white_jpg": len(jpgs) - len(augs),
                "grey_cr2":  len(cr2s),
                "aug_generated": generated,
                "total": len([f for f in files_after if f.upper().endswith(".JPG")])
            }

    print("\n" + "=" * 65)
    print("  AUGMENTATION SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<25} {'White JPG':>10} {'Grey CR2':>10} {'Aug Gen':>10} {'Total JPG':>10}")
    print("  " + "-" * 63)
    for cls, info in summary.items():
        print(f"  {cls:<25} {info['white_jpg']:>10} {info['grey_cr2']:>10} "
              f"{info['aug_generated']:>10} {info['total']:>10}")
    print("=" * 65)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — MODEL: MobileNetV2 + Custom Sequential-Style Classification Head
# ─────────────────────────────────────────────────────────────────────────────

def build_transfer_model(num_classes: int, input_shape=(128, 128, 3)):
    """
    Transfer learning model:
      - MobileNetV2 (ImageNet weights) as frozen feature extractor
      - Custom head: GAP -> BN -> Dropout -> Dense(256,relu) -> Dropout
                     -> Dense(128,relu) -> Dropout -> Dense(n, softmax)
    Returns (model, base_model) for 2-phase training.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    # MobileNetV2 expects inputs in [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)

    # Sequential-style classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.40, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.30, name="drop2")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.20, name="drop3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="StainScan_CNN_v2_Hybrid")
    return model, base


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — DATA LOADING & 2-PHASE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class JPGOnlyImageDataGenerator(ImageDataGenerator):
    """Subclass that filters out non-JPG files (e.g. CR2 RAW) after directory scan."""
    def flow_from_directory(self, directory, **kwargs):
        gen = super().flow_from_directory(directory, **kwargs)
        mask = np.array([f.lower().endswith(('.jpg', '.jpeg')) for f in gen.filenames])
        gen.filenames = [f for f, ok in zip(gen.filenames, mask) if ok]
        gen.classes   = gen.classes[mask]
        gen.samples   = len(gen.filenames)
        gen.n         = gen.samples
        skipped = (~mask).sum()
        print(f"  [JPGOnly] {gen.samples} JPG images loaded ({skipped} non-JPG skipped)")
        return gen


def run_training(aug_summary: dict):
    print("\n" + "=" * 65)
    print("  PHASE 2 -- MODEL RETRAINING (Transfer Learning)")
    print("=" * 65)

    train_datagen = JPGOnlyImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.20,
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.80, 1.20],
        fill_mode="nearest",
    )
    val_datagen = JPGOnlyImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.20,
    )

    print(f"\n  Loading data from: {DATASET_DIR}")
    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", seed=SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", seed=SEED,
    )

    num_classes = len(train_gen.class_indices)
    class_map   = {v: k for k, v in train_gen.class_indices.items()}

    print(f"\n  Classes     : {train_gen.class_indices}")
    print(f"  Train set   : {train_gen.samples} images")
    print(f"  Val set     : {val_gen.samples} images")
    print("\n  Per-class totals:")
    for cls, info in aug_summary.items():
        print(f"    {cls:<32} -> {info['total']} JPGs")

    model, base_model = build_transfer_model(num_classes, input_shape=IMG_SIZE + (3,))
    model.summary()

    t0 = time.time()

    # ── Phase 2a: Head-only training ──────────────────────────────────────────
    print("\n  [Phase 2a] Head-only training (MobileNetV2 frozen)")
    print("  LR=1e-3 | max 15 epochs | EarlyStopping patience=5")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb_a = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-6, verbose=1),
    ]
    hist_a = model.fit(train_gen, validation_data=val_gen, epochs=15,
                       callbacks=cb_a, verbose=1)
    print(f"  Phase 2a complete — {(time.time()-t0)/60:.1f} min elapsed.")

    # ── Phase 2b: Fine-tune top-30 MobileNetV2 layers ────────────────────────
    print("\n  [Phase 2b] Fine-tuning top-30 MobileNetV2 layers")
    print("  LR=1e-5 | max 25 epochs | EarlyStopping patience=8")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Unfrozen MobileNetV2 layers: {n_trainable}/{len(base_model.layers)}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb_b = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-8, verbose=1),
    ]
    train_gen.reset(); val_gen.reset()
    t1 = time.time()
    hist_b = model.fit(train_gen, validation_data=val_gen, epochs=25,
                       callbacks=cb_b, verbose=1)
    print(f"  Phase 2b complete — {(time.time()-t1)/60:.1f} min for fine-tuning.")

    # Merge histories
    merged_h = {
        "accuracy":     hist_a.history.get("accuracy", []) + hist_b.history.get("accuracy", []),
        "val_accuracy": hist_a.history.get("val_accuracy", []) + hist_b.history.get("val_accuracy", []),
        "val_loss":     hist_a.history.get("val_loss", []) + hist_b.history.get("val_loss", []),
    }
    class _Hist:
        def __init__(self, h): self.history = h
    print(f"\n  Total training time: {(time.time()-t0)/60:.1f} minutes.")
    return model, _Hist(merged_h), train_gen, val_gen, class_map


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — EVALUATION & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, history, val_gen, class_map, aug_summary):
    print("\n" + "=" * 65)
    print("  PHASE 3 -- VALIDATION & REPORTING")
    print("=" * 65)

    val_accs   = history.history.get("val_accuracy", [])
    train_accs = history.history.get("accuracy", [])
    val_losses = history.history.get("val_loss", [])

    best_ep        = int(np.argmax(val_accs)) if val_accs else 0
    best_val_acc   = max(val_accs) * 100 if val_accs else 0.0
    best_train_acc = train_accs[best_ep] * 100 if train_accs else 0.0

    print(f"\n  Best Epoch          : {best_ep + 1}")
    print(f"  Training Accuracy   : {best_train_acc:.2f}%")
    print(f"  Validation Accuracy : {best_val_acc:.2f}%")
    print(f"  Best Val Loss       : {val_losses[best_ep]:.4f}")

    print("\n  Running final evaluation on validation set...")
    val_gen.reset()
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"  Final Eval Accuracy : {acc*100:.2f}%")
    print(f"  Final Eval Loss     : {loss:.4f}")

    # Per-class predictions
    val_gen.reset()
    all_preds, all_labels = [], []
    for _ in range(len(val_gen)):
        xb, yb = next(val_gen)
        preds = model.predict(xb, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(np.argmax(yb, axis=1))
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    nc = len(class_map)

    # Confusion matrix
    print("\n" + "=" * 65)
    print("  CONFUSION MATRIX (Validation Set)")
    print("=" * 65)
    short = [class_map[i][:14] for i in range(nc)]
    hdr = "  {:>16}  " + "  {:>14}" * nc
    print(hdr.format("Actual \\ Pred", *short))
    print("  " + "-" * (18 + 16 * nc))
    for t in range(nc):
        row = [int(((all_labels==t) & (all_preds==p)).sum()) for p in range(nc)]
        print(("  {:>16}  " + "  {:>14}" * nc).format(short[t], *row))

    # Per-class accuracy
    print("\n" + "=" * 65)
    print("  CLASS-WISE VALIDATION ACCURACY")
    print("=" * 65)
    for i in range(nc):
        mask = all_labels == i
        correct = int((all_preds[mask] == i).sum())
        total   = int(mask.sum())
        pct = 100.0 * correct / total if total else 0.0
        print(f"  {class_map[i]:<32} -> {pct:.2f}%  ({correct}/{total})")

    # Dataset summary
    print("\n" + "=" * 65)
    print("  DATASET IMAGE COUNTS PER CLASS")
    print("=" * 65)
    print(f"  {'Class':<30} {'White JPG':>10} {'Grey CR2':>10} {'Aug Gen':>10} {'Total JPG':>10}")
    print("  " + "-" * 65)
    total_all = 0
    for cls, info in aug_summary.items():
        print(f"  {cls:<30} {info['white_jpg']:>10} {info['grey_cr2']:>10} "
              f"{info['aug_generated']:>10} {info['total']:>10}")
        total_all += info["total"]
    print("  " + "-" * 65)
    print(f"  {'GRAND TOTAL':<30} {'':>10} {'':>10} {'':>10} {total_all:>10}")
    print("=" * 65)
    print(f"\n  Model saved to: {MODEL_OUT}")
    print("  Pipeline complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  StainScan Hybrid Dataset -- Augmentation + Retraining Pipeline")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU        : {bool(tf.config.list_physical_devices('GPU'))}")
    print("=" * 65)

    aug_summary = run_augmentation()
    model, history, train_gen, val_gen, class_map = run_training(aug_summary)
    run_evaluation(model, history, val_gen, class_map, aug_summary)
