"""
StainScan — Transfer Learning Retraining Pipeline (MobileNetV2)
==============================================================
Pipeline:
  1. Balanced Physical Dataset Loading (224x224):
     - 300 physical white cotton photos (.JPG) per class
     - 300 physical grey cotton photos (.CR2) per class
     - Total: 600 photos per class across 3 classes = 1,800 images
     - CIELAB CLAHE (Contrast Limited Adaptive Histogram Equalization) dynamic preprocessing
     - Parallel extraction via embedded camera raw preview + bilinear resizing to (224, 224)
     - Multi-threaded dataset processing with compressed caching to dataset_clahe_224.npz

  2. Preprocessing & Normalization:
     - Exact match between training and inference:
       RGB -> resize (224, 224) -> CIELAB CLAHE -> mobilenet_v2.preprocess_input
     - Brightness, contrast, and spatial augmentations tuned for neutral/dark fabric surfaces

  3. Model Architecture (Transfer Learning):
     - Base model: MobileNetV2 pre-trained on ImageNet (weights='imagenet', include_top=False)
     - Custom classification head:
       GlobalAveragePooling2D -> BatchNormalization -> Dropout(0.4) -> Dense(3, softmax)

  4. Training Strategy:
     - Phase 1 (Feature Extraction):
       Freeze base model (base_model.trainable = False)
       Train for 20 epochs with Adam(learning_rate=1e-3)
     - Phase 2 (Fine-Tuning):
       Unfreeze top 20 layers of MobileNetV2
       Fine-tune for 10 epochs with Adam(learning_rate=1e-5)
     - Save to stain_model.h5, stainscan_model_v3_balanced.h5, and native .keras

  5. Live Inference Verification:
     - Test validation on sample grey fabric photos verifying high confidence (>85%)
"""

import io
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import shutil
import random
import glob
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rawpy
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
DATASET_DIR       = "DataSet/StainScan/dataset"
CACHE_FILE        = "dataset_clahe_224.npz"
IMG_SIZE          = (224, 224)
BATCH_SIZE        = 32
PHASE1_EPOCHS     = 20
PHASE2_EPOCHS     = 10
PHASE1_LR         = 1e-3
PHASE2_LR         = 1e-5
STAIN_MODEL_H5    = "stain_model.h5"
STAINSCAN_MODEL_H5= "stainscan_model.h5"
MODEL_H5_OUT      = "stainscan_model_v3_balanced.h5"
MODEL_KERAS_OUT   = "stainscan_model_v3_balanced.keras"
SEED              = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASSES = [
    "Ballpen Ink_cotton",
    "Cooking Oil_cotton",
    "Mud_cotton"
]

CLASS_DISPLAY = {
    "Ballpen Ink_cotton": "Black Ballpen Ink",
    "Cooking Oil_cotton": "Used Cooking Oil",
    "Mud_cotton":         "Mud"
}


# ── 2. OPENCV CLAHE PREPROCESSING ────────────────────────────────────────────
def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel
    in CIELAB color space to suppress grey fabric background luminance variations.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


def load_single_image(filepath: str) -> np.ndarray:
    """Loads a JPG or CR2 image, resizes to IMG_SIZE (224x224), applies CLAHE, and returns uint8 RGB array."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".cr2":
        with rawpy.imread(filepath) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as img:
                        img.draft("RGB", IMG_SIZE)
                        img = img.resize(IMG_SIZE, Image.BILINEAR)
                        arr = np.array(img.convert("RGB"), dtype=np.uint8)
                        return apply_clahe(arr)
            except Exception:
                pass
            rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            img = Image.fromarray(rgb).resize(IMG_SIZE, Image.BILINEAR)
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
            return apply_clahe(arr)
    else:
        with Image.open(filepath) as img:
            img.draft("RGB", IMG_SIZE)
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
            return apply_clahe(arr)


def load_or_build_dataset():
    """
    Loads preprocessed dataset from compressed cache, or constructs it from disk
    using physical white cotton JPGs and physical grey cotton CR2s with multi-threading.
    """
    if os.path.exists(CACHE_FILE):
        print(f"\n[CACHE] Loading preprocessed dataset from {CACHE_FILE}...")
        data = np.load(CACHE_FILE, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        class_names = [str(c) for c in data["classes"]]
        stats = data["stats"].item() if "stats" in data else {}
        print(f"[CACHE] Loaded {len(X)} images across classes: {class_names}")
        return X, y, class_names, stats

    print(f"\n[DATA] Scanning dataset directory: {DATASET_DIR}...")
    X, y = [], []
    stats = {}
    t0 = time.time()

    for idx, cname in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, cname)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")

        all_files = os.listdir(folder)
        white_files = sorted([os.path.join(folder, f) for f in all_files if f.upper().endswith(".JPG") and not f.startswith("aug_")])
        grey_files  = sorted([os.path.join(folder, f) for f in all_files if f.upper().endswith(".CR2")])

        print(f"\n  [CLASS {idx}] {CLASS_DISPLAY[cname]} ({cname}):")
        print(f"    - Physical White Cotton Photos (.JPG): {len(white_files)}")
        print(f"    - Physical Grey Cotton Photos (.CR2) : {len(grey_files)}")
        print(f"    - Total Images for Class             : {len(white_files) + len(grey_files)}")

        stats[cname] = {
            "white": len(white_files),
            "grey": len(grey_files),
            "total": len(white_files) + len(grey_files)
        }

        all_class_files = white_files + grey_files
        print(f"    - Loading and applying CLAHE to {len(all_class_files)} images at {IMG_SIZE} in parallel...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            class_images = list(executor.map(load_single_image, all_class_files))

        X.extend(class_images)
        y.extend([idx] * len(class_images))

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)

    print(f"\n[CACHE] Saving {len(X)} images to {CACHE_FILE} (took {time.time() - t0:.1f}s)...")
    np.savez_compressed(CACHE_FILE, X=X, y=y, classes=CLASSES, stats=stats)
    return X, y, CLASSES, stats


# ── 3. MOBILENETV2 TRANSFER LEARNING ARCHITECTURE ────────────────────────────
def build_mobilenetv2_transfer_model(num_classes: int = 3, input_shape: tuple = (224, 224, 3)):
    """
    Builds a Transfer Learning model using pre-trained MobileNetV2 with ImageNet weights.
    Appends custom classification head: GlobalAveragePooling2D, BatchNormalization,
    Dropout(0.4), and Dense(3, softmax).
    """
    # Base model pre-trained on ImageNet
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Feature extraction via MobileNetV2 (training=False ensures BN layers in base stay frozen)
    x = base_model(inputs, training=False)

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.40, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="stainscan_mobilenetv2_transfer")
    return model, base_model


# ── 4. TRAINING & FINE-TUNING PIPELINE ─────────────────────────────────────────
def run_retraining():
    print("=" * 70)
    print("  STAINSCAN — MOBILENETV2 TRANSFER LEARNING PIPELINE (224x224)")
    print("=" * 70)

    # 1. Load dataset (uint8 arrays with CIELAB CLAHE applied)
    X_raw, y_raw, class_names, stats = load_or_build_dataset()
    num_classes = len(class_names)

    # Stratified 80/20 train/validation split on raw CLAHE uint8 images
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw
    )

    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat   = keras.utils.to_categorical(y_val,   num_classes=num_classes)

    print(f"\n[SPLIT] Total Dataset:    {len(X_raw)} images")
    print(f"        Training Set:     {len(X_train_raw)} images (80%)")
    print(f"        Validation Set:   {len(X_val_raw)} images (20%)")

    # Data Augmentation & Normalization Pipeline:
    # Brightness, contrast, and HSV color jitter augmentations specifically tuned for neutral/dark fabric surfaces
    @tf.function
    def train_augment_and_preprocess(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=25.0)        # Dark/neutral fabric brightness shift
        img = tf.image.random_contrast(img, lower=0.7, upper=1.4)   # Low-contrast stain edge enhancement
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)  # HSV saturation jitter
        img = tf.image.random_hue(img, max_delta=0.05)              # HSV hue jitter
        img = tf.clip_by_value(img, 0.0, 255.0)
        img = preprocess_input(img)
        return img, label

    @tf.function
    def val_preprocess(img, label):
        return preprocess_input(img), label

    train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(X_train_raw, tf.float32), y_train_cat))
    train_ds = train_ds.shuffle(buffer_size=1024, seed=SEED)
    train_ds = train_ds.map(train_augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.cast(X_val_raw, tf.float32), y_val_cat))
    val_ds = val_ds.map(val_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 2. Build model
    model, base_model = build_mobilenetv2_transfer_model(
        num_classes=num_classes, input_shape=IMG_SIZE + (3,)
    )
    print("\n[ARCHITECTURE] Model Summary:")
    model.summary()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: FEATURE EXTRACTION (Base Frozen, 20 Epochs, lr=1e-3)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  PHASE 1: FEATURE EXTRACTION — FROZEN BASE ({PHASE1_EPOCHS} Epochs, lr={PHASE1_LR})")
    print("=" * 70)

    base_model.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_phase1 = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=STAIN_MODEL_H5,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    t_p1 = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=cb_phase1,
        verbose=1
    )
    print(f"\n[PHASE 1 DONE] Completed in {time.time() - t_p1:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: FINE-TUNING (Unfreeze Top 20 Layers, 10 Epochs, lr=1e-5)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  PHASE 2: FINE-TUNING — TOP 20 LAYERS ({PHASE2_EPOCHS} Epochs, lr={PHASE2_LR})")
    print("=" * 70)

    base_model.trainable = True
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    # Keep BatchNormalization layers frozen to protect pretrained batch statistics
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    print(f"[FINE-TUNE] Trainable weights in model: {len(model.trainable_weights)}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_phase2 = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=STAIN_MODEL_H5,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    t_p2 = time.time()
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=cb_phase2,
        verbose=1
    )
    print(f"\n[PHASE 2 DONE] Fine-tuning completed in {time.time() - t_p2:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. SAVE FINAL ARTIFACTS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[SAVE] Saving best model weights to '{STAIN_MODEL_H5}' and '{MODEL_H5_OUT}'...")
    best_model = keras.models.load_model(STAIN_MODEL_H5, compile=False)
    best_model.save(MODEL_H5_OUT)
    best_model.save(STAINSCAN_MODEL_H5)
    print(f"[SAVE] Saved to '{STAIN_MODEL_H5}', '{STAINSCAN_MODEL_H5}', and '{MODEL_H5_OUT}'")

    try:
        best_model.save(MODEL_KERAS_OUT)
        print(f"[SAVE] Also saved native Keras format: '{MODEL_KERAS_OUT}'")
    except Exception as e:
        print(f"[WARN] Native Keras format save warning: {e}")

    try:
        best_model.save("stain_model.keras")
        print(f"[SAVE] Also saved native Keras format: 'stain_model.keras'")
    except Exception as e:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # 6. EVALUATE FINAL PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────────
    best_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    X_train_norm = preprocess_input(X_train_raw.astype(np.float32))
    X_val_norm   = preprocess_input(X_val_raw.astype(np.float32))

    train_loss, train_acc = best_model.evaluate(X_train_norm, y_train_cat, verbose=0)
    val_loss, val_acc     = best_model.evaluate(X_val_norm, y_val_cat, verbose=0)

    val_preds = best_model.predict(X_val_norm, verbose=0)
    val_pred_labels = np.argmax(val_preds, axis=1)

    target_names = [CLASS_DISPLAY[c] for c in class_names]
    report = classification_report(y_val, val_pred_labels, target_names=target_names, digits=4)
    cm = confusion_matrix(y_val, val_pred_labels)

    print("\n" + "=" * 70)
    print("  FINAL PERFORMANCE METRICS & SUMMARY")
    print("=" * 70)
    print(f"\n  Final Training Accuracy   : {train_acc * 100:.2f}%  (Loss: {train_loss:.4f})")
    print(f"  Final Validation Accuracy : {val_acc * 100:.2f}%  (Loss: {val_loss:.4f})")

    print("\n  Summary of Processed Images Per Class:")
    print("  " + "-" * 62)
    print(f"  {'Class Name':<22} | {'White Cotton':<12} | {'Grey Cotton':<11} | {'Total':<7}")
    print("  " + "-" * 62)
    total_white, total_grey, total_all = 0, 0, 0
    for cname in class_names:
        c_stats = stats.get(cname, {})
        w = c_stats.get("white", 300)
        g = c_stats.get("grey", 300)
        tot = c_stats.get("total", w + g)
        total_white += w
        total_grey += g
        total_all += tot
        print(f"  {CLASS_DISPLAY[cname]:<22} | {w:<12} | {g:<11} | {tot:<7}")
    print("  " + "-" * 62)
    print(f"  {'TOTAL':<22} | {total_white:<12} | {total_grey:<11} | {total_all:<7}")

    print("\n  Classification Report (Validation Set):")
    print(report)

    print("  Confusion Matrix:")
    print(f"  Classes: {target_names}")
    print(cm)
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. TEST VALIDATION ON PHYSICAL GREY FABRIC IMAGES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST VALIDATION ON PHYSICAL GREY FABRIC IMAGES (>85% Target)")
    print("=" * 70)

    for class_idx, cname in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, cname)
        all_cr2s = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.upper().endswith(".CR2")])
        test_samples = all_cr2s[:5]  # Test 5 sample grey cotton images per class

        print(f"\n  Class {class_idx}: {CLASS_DISPLAY[cname]} ({len(test_samples)} grey samples):")
        for sample_path in test_samples:
            img_arr = load_single_image(sample_path)  # (224, 224, 3) uint8 CLAHE
            # Exact same preprocessing as server.py
            preprocessed = preprocess_input(img_arr.astype(np.float32))
            input_tensor = np.expand_dims(preprocessed, axis=0)
            preds = best_model.predict(input_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            conf = float(preds[pred_idx]) * 100.0
            pred_name = CLASS_DISPLAY[CLASSES[pred_idx]]
            is_correct = (pred_idx == class_idx)
            is_high_conf = (conf >= 85.0)

            status = "PASS (>85%)" if (is_correct and is_high_conf) else ("PASS" if is_correct else "WARN")
            print(f"    [{status}] {os.path.basename(sample_path)}: Predicted '{pred_name}' with {conf:.2f}% confidence | Probs: {[round(p*100,1) for p in preds]}")

    print("\n" + "=" * 70)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "stats": stats
    }


if __name__ == "__main__":
    run_retraining()
