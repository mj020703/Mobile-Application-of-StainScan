import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"
import time
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CACHE_FILE = "dataset_domain_adapted_224.npz"
MODEL_KERAS_OUT = "stainscan_model_v3_balanced.keras"
MODEL_H5_OUT = "stainscan_model_v3_balanced.h5"
BASE_MODEL_PATH = "domain_best.keras"
BATCH_SIZE = 32
EPOCHS = 4
SEED = 42

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
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def run():
    print("=" * 70, flush=True)
    print("  FAST DOMAIN FINE-TUNING (4 EPOCHS)", flush=True)
    print("=" * 70, flush=True)

    data = np.load(CACHE_FILE, allow_pickle=True)
    X_raw = data["X"]
    y_raw = data["y"]
    num_classes = len(CLASSES)

    print(f"Applying CLAHE to {len(X_raw)} images...", flush=True)
    X_clahe = np.empty_like(X_raw)
    for i in range(len(X_raw)):
        X_clahe[i] = apply_clahe(X_raw[i])

    X_train, X_val, y_train, y_val = train_test_split(
        X_clahe, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw
    )
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)

    @tf.function
    def train_augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        gain = tf.random.uniform([], minval=0.40, maxval=1.25)
        img = img * gain
        bias = tf.random.uniform([], minval=-20.0, maxval=15.0)
        img = img + bias
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
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

    print(f"Loading base model from '{BASE_MODEL_PATH}'...", flush=True)
    model = keras.models.load_model(BASE_MODEL_PATH, compile=False)

    # Unfreeze top 30 layers
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            for sub_l in layer.layers[:-30]:
                sub_l.trainable = False
            for sub_l in layer.layers[-30:]:
                if isinstance(sub_l, layers.BatchNormalization):
                    sub_l.trainable = False
                else:
                    sub_l.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    print(f"\n[DONE] Fine-tuning finished in {time.time() - t0:.1f}s", flush=True)

    print(f"Saving production models...", flush=True)
    model.save(MODEL_KERAS_OUT)
    model.save("stain_model.keras")
    model.save(MODEL_H5_OUT)
    model.save("stain_model.h5")
    model.save("stainscan_model.h5")

    # Evaluate on all user scans
    print("\n" + "=" * 70, flush=True)
    print("  RAW CNN INFERENCE ON ALL REAL USER SCANS (NO GUARDRAILS)", flush=True)
    print("=" * 70, flush=True)

    test_scans = [
        ("scratch/scan_147.jpg", 1, "Cooking Oil (Grey Cotton) - Screen 1&2"),
        ("scratch/h_20261106100010_174.jpg", 1, "Cooking Oil (Grey Cotton) - Screen 2 Latest"),
        ("scratch/h_20261106100034_175.jpg", 1, "Cooking Oil (White Cotton) - Screen 1 Latest"),
        ("scratch/h_20261106075932_165.jpg", 1, "Cooking Oil (White Cotton) - Screen 4"),
    ]

    for fpath, ground_truth_idx, desc in test_scans:
        raw_bgr = cv2.imread(fpath)
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        sz = min(h, w)
        l = (w - sz) // 2
        t = (h - sz) // 2
        sq = rgb[t:t+sz, l:l+sz]
        resized = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_LINEAR)
        clahe_img = apply_clahe(resized)
        inp = preprocess_input(clahe_img.astype(np.float32))[None]

        preds = model.predict(inp, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        conf = float(preds[pred_idx]) * 100.0
        status = "PASS" if pred_idx == ground_truth_idx else "FAIL"

        print(f"\n  Scan: {desc}")
        print(f"    Result     : [{status}] Predicted '{CLASS_DISPLAY[pred_idx]}' ({conf:.2f}%) | True: '{CLASS_DISPLAY[ground_truth_idx]}'")
        print(f"    Raw Probs  : Ink={preds[0]*100:.2f}%, Oil={preds[1]*100:.2f}%, Mud={preds[2]*100:.2f}%", flush=True)

if __name__ == "__main__":
    run()
