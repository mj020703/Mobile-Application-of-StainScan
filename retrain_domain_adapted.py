"""
StainScan — Domain-Adapted MobileNetV2 Retraining Pipeline
==========================================================
Eliminates the Studio DSLR vs. Real Smartphone domain gap via:
1. Low-key exposure gain jitter (0.35x - 1.25x)
2. Additive illumination bias & contrast/saturation jitter
3. Direct real smartphone scan injection
4. Two-phase transfer learning (Warmup -> Top-30 layer Fine-Tuning)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import time
import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CACHE_FILE       = "dataset_domain_adapted_224.npz"
MODEL_KERAS_OUT  = "stainscan_model_v3_balanced.keras"
MODEL_H5_OUT     = "stainscan_model_v3_balanced.h5"
CHECKPOINT_KERAS = "domain_best.keras"
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
PHASE1_EPOCHS    = 15
PHASE2_EPOCHS    = 12
SEED             = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASSES = [
    "Ballpen Ink_cotton",
    "Cooking Oil_cotton",
    "Mud_cotton"
]

CLASS_DISPLAY = {
    0: "Black Ballpen Ink",
    1: "Used Cooking Oil",
    2: "Mud"
}

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Applies CIELAB CLAHE to match server preprocessing."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def build_model(num_classes: int = 3, input_shape: tuple = (224, 224, 3)):
    """Matches the exact architecture of stainscan_model_v3_balanced.keras."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.40, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="stainscan_mobilenetv2_transfer")
    return model, base_model

def run():
    print("=" * 70, flush=True)
    print("  STAINSCAN — DOMAIN-ADAPTED MOBILENETV2 RETRAINING PIPELINE", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache file '{CACHE_FILE}' not found! Run prepare_domain_dataset.py first.")

    print(f"\n[1/5] Loading preprocessed dataset from '{CACHE_FILE}'...", flush=True)
    t0 = time.time()
    data = np.load(CACHE_FILE, allow_pickle=True)
    X_raw = data["X"]
    y_raw = data["y"]
    classes = [str(c) for c in data["classes"]]
    num_classes = len(classes)
    print(f"  Loaded {len(X_raw)} images across classes: {classes} in {time.time() - t0:.1f}s", flush=True)

    # Apply CLAHE to all images so domain-adaptation matches runtime inference
    print("\n[2/5] Applying CIELAB CLAHE normalization to dataset...", flush=True)
    t_c = time.time()
    X_clahe = np.empty_like(X_raw)
    for i in range(len(X_raw)):
        X_clahe[i] = apply_clahe(X_raw[i])
    print(f"  CLAHE applied to {len(X_clahe)} images in {time.time() - t_c:.1f}s", flush=True)

    # Stratified 80/20 train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_clahe, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw
    )
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat   = keras.utils.to_categorical(y_val,   num_classes=num_classes)

    print(f"\n[3/5] Dataset Split:")
    print(f"  Training set   : {len(X_train)} samples")
    print(f"  Validation set : {len(X_val)} samples")
    print(f"  Train class counts: {np.bincount(y_train)}")
    print(f"  Val class counts  : {np.bincount(y_val)}", flush=True)

    # Domain Adaptation Data Augmentation:
    # Simulates low-light indoor phone captures, ambient underexposure, and room color variations
    @tf.function
    def train_augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        # 1. Multiplicative lighting gain (0.35x to 1.25x) - breaks the studio luminance shortcut!
        gain = tf.random.uniform([], minval=0.35, maxval=1.25)
        img = img * gain

        # 2. Additive brightness bias (-20 to +15)
        bias = tf.random.uniform([], minval=-20.0, maxval=15.0)
        img = img + bias

        # 3. Contrast variation
        img = tf.image.random_contrast(img, lower=0.6, upper=1.4)

        # 4. Saturation variation (warm vs cool light)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)

        # 5. Hue variation
        img = tf.image.random_hue(img, max_delta=0.04)

        img = tf.clip_by_value(img, 0.0, 255.0)
        img = preprocess_input(img)
        return img, label

    @tf.function
    def val_preprocess(img, label):
        return preprocess_input(img), label

    train_ds = tf.data.Dataset.from_tensor_slices((tf.cast(X_train, tf.float32), y_train_cat))
    train_ds = train_ds.shuffle(buffer_size=1024, seed=SEED)
    train_ds = train_ds.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.cast(X_val, tf.float32), y_val_cat))
    val_ds = val_ds.map(val_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build Model
    model, base_model = build_model(num_classes=num_classes, input_shape=IMG_SIZE + (3,))

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: FEATURE EXTRACTION (Base Frozen)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print(f"  PHASE 1: BASE FROZEN WARMUP ({PHASE1_EPOCHS} Epochs, lr=1e-3)", flush=True)
    print("=" * 70, flush=True)

    base_model.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_p1 = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(filepath=CHECKPOINT_KERAS, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    t_p1 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=cb_p1,
        verbose=1
    )
    print(f"\n[PHASE 1 COMPLETE] in {time.time() - t_p1:.1f}s", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: DOMAIN ADAPTATION FINE-TUNING (Top 30 Layers)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print(f"  PHASE 2: DOMAIN ADAPTATION FINE-TUNING ({PHASE2_EPOCHS} Epochs, lr=3e-5)", flush=True)
    print("=" * 70, flush=True)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    print(f"  Trainable weights: {len(model.trainable_weights)}", flush=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_p2 = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(filepath=CHECKPOINT_KERAS, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    t_p2 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=cb_p2,
        verbose=1
    )
    print(f"\n[PHASE 2 COMPLETE] in {time.time() - t_p2:.1f}s", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE BEST MODEL ARTIFACTS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4/5] Loading best model from '{CHECKPOINT_KERAS}' and saving production artifacts...", flush=True)
    best_model = keras.models.load_model(CHECKPOINT_KERAS, compile=False)

    best_model.save(MODEL_KERAS_OUT)
    best_model.save("stain_model.keras")
    best_model.save(MODEL_H5_OUT)
    best_model.save("stain_model.h5")
    best_model.save("stainscan_model.h5")
    print(f"  Saved to '{MODEL_KERAS_OUT}', '{MODEL_H5_OUT}', and legacy targets.", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION ON REAL SMARTPHONE SCANS (ZERO GUARDRAILS - PURE RAW CNN)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("  [5/5] RAW CNN ACCURACY BENCHMARK ON REAL SMARTPHONE SCANS (NO GUARDRAILS)", flush=True)
    print("=" * 70, flush=True)

    test_scans = [
        ("scratch/scan_147.jpg", 1, "Cooking Oil (Grey Cotton) - Screen 1&2"),
        ("scratch/latest_mobile_scan.jpg", 1, "Cooking Oil (Grey Cotton)"),
        ("scratch/h_20261106075932_165.jpg", 1, "Cooking Oil (White Cotton) - Screen 4"),
        ("scratch/h_20261106075838_162.jpg", 2, "Mud (Grey Cotton)"),
    ]

    for fpath, ground_truth_idx, desc in test_scans:
        if not os.path.exists(fpath):
            continue
        raw_bgr = cv2.imread(fpath)
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        sz = min(h, w)
        l = (w - sz) // 2
        t = (h - sz) // 2
        sq = rgb[t:t+sz, l:l+sz]
        resized = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_LINEAR)
        clahe_img = apply_clahe(resized)
        inp = preprocess_input(clahe_img.astype(np.float32))
        inp = np.expand_dims(inp, axis=0)

        preds = best_model.predict(inp, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        conf = float(preds[pred_idx]) * 100.0
        status = "PASS" if pred_idx == ground_truth_idx else "FAIL"

        print(f"\n  Scan: {desc} ({os.path.basename(fpath)})")
        print(f"    Result     : [{status}] Predicted '{CLASS_DISPLAY[pred_idx]}' ({conf:.2f}%) | True: '{CLASS_DISPLAY[ground_truth_idx]}'")
        print(f"    Raw Probs  : Ink={preds[0]*100:.2f}%, Oil={preds[1]*100:.2f}%, Mud={preds[2]*100:.2f}%", flush=True)

    # Validation Set Evaluation
    X_val_norm = preprocess_input(X_val.astype(np.float32))
    val_preds = best_model.predict(X_val_norm, verbose=0)
    val_pred_labels = np.argmax(val_preds, axis=1)

    print("\n" + "=" * 70, flush=True)
    print("  VALIDATION SET CLASSIFICATION REPORT", flush=True)
    print("=" * 70, flush=True)
    print(classification_report(y_val, val_pred_labels, target_names=[CLASS_DISPLAY[i] for i in range(3)], digits=4), flush=True)
    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(y_val, val_pred_labels), flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run()
