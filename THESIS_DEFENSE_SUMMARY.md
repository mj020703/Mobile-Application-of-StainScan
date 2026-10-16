# StainScan: Thesis Defense Technical Cheat Sheet
**System Architecture, Machine Learning Pipeline, Dataset Specifications & Defense Guide**  
*Academic Year 2025–2026 | STI West Negros University*

---

## 1. System & Architecture Overview

### 1.1 Machine Learning Architecture & CNN Model Design

StainScan utilizes **Transfer Learning** based on the lightweight **MobileNetV2** convolutional neural network backbone, pre-trained on ImageNet. It is tailored for edge-capable image classification of stains on cotton fabrics.

```
Input Image (RGB)
      │
      ▼
Resize to (224, 224) ── Bilinear Interpolation
      │
      ▼
CIELAB CLAHE Contrast Equalization (L* channel, clipLimit=2.0, tileGrid=(8,8))
      │
      ▼
MobileNetV2 preprocess_input ── Normalize pixels to [-1.0, 1.0]
      │
      ▼
MobileNetV2 Backbone (154 layers, Inverted Residual Blocks, Depthwise Separable Convolutions)
      │  [Output shape: (None, 7, 7, 1280) | 2,257,984 parameters]
      ▼
GlobalAveragePooling2D (GAP) ── Reduces spatial dimensions to (None, 1280)
      │
      ▼
BatchNormalization (bn_head) ── Stabilizes head activations (5,120 parameters)
      │
      ▼
Dropout (rate = 0.40) ── Regularization against fabric weave overfitting
      │
      ▼
Dense Output Layer (units = 3, activation = "softmax") ── (None, 3) (3,843 parameters)
      │
      ▼
Class Probabilities: [Black Ballpen Ink, Used Cooking Oil, Mud]
```

#### Detailed Architecture Specifications

| Component / Layer | Specification / Type | Output Shape | Parameters | Purpose & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Input Layer** | `InputLayer` | `(None, 224, 224, 3)` | 0 | Standard MobileNetV2 high-resolution input tensor |
| **Base Backbone** | `MobileNetV2` (Functional, `include_top=False`, ImageNet) | `(None, 7, 7, 1280)` | 2,257,984 | Low-latency feature extractor utilizing depthwise separable convolutions and linear bottlenecks |
| **Pooling Layer** | `GlobalAveragePooling2D` | `(None, 1280)` | 0 | Converts feature maps to 1D vector while enforcing spatial invariance |
| **Normalization** | `BatchNormalization` | `(None, 1280)` | 5,120 *(2,560 trainable, 2,560 non-trainable)* | Normalizes feature scales to prevent internal covariate shift in the custom classification head |
| **Regularization** | `Dropout` | `(None, 1280)` | 0 | Drop rate of 40% (0.40) prevents co-adaptation of features on fabric textures |
| **Classifier Head** | `Dense` (`softmax`) | `(None, 3)` | 3,843 *(1280 × 3 + 3)* | Generates normalized posterior probability distribution over the 3 target stain classes |

- **Total Parameters:** **2,266,947** (~8.65 MB)
- **Trainable Parameters:** **1,201,283** (during Phase 2 fine-tuning)
- **Non-Trainable Parameters:** **1,065,664** (frozen lower backbone layers and batch statistics)
- **Primary Model Artifacts:** [`stainscan_model_v3_balanced.keras`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/stainscan_model_v3_balanced.keras) and [`stainscan_model_v3_balanced.h5`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/stainscan_model_v3_balanced.h5)

#### Two-Phase Training Strategy

1. **Phase 1: Feature Extraction (Frozen Base)**
   - **Base Status:** Entire MobileNetV2 backbone frozen (`base_model.trainable = False`).
   - **Trained Layers:** Only GAP, BatchNormalization, Dropout, and Dense layers.
   - **Epochs:** 20 epochs.
   - **Optimizer:** `Adam(learning_rate = 1e-3)`.
   - **Loss Function:** `Categorical Crossentropy`.
   - **Callbacks:** `ReduceLROnPlateau` (factor=0.5, patience=3, min_lr=1e-5), `ModelCheckpoint` tracking `val_accuracy`.
2. **Phase 2: Fine-Tuning (Top 20 Layers)**
   - **Base Status:** Top 20 layers of MobileNetV2 unfrozen (`base_model.layers[:-20]` frozen).
   - **Batch Normalization Freeze:** All `BatchNormalization` layers across the base network are strictly kept frozen (`layer.trainable = False`) to prevent distortion of pre-trained ImageNet mean and variance statistics.
   - **Epochs:** 10 epochs.
   - **Optimizer:** `Adam(learning_rate = 1e-5)` (100× lower to avoid catastrophic forgetting of learned weights).
   - **Loss Function:** `Categorical Crossentropy`.
   - **Callbacks:** `ReduceLROnPlateau` (factor=0.5, patience=2, min_lr=1e-7), `ModelCheckpoint` saving optimal weights.

---

### 1.2 Tech Stack Across System Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CLIENT TIER (Mobile & Admin)                          │
│  • HTML5 Semantic Elements + Responsive Glassmorphic Vanilla CSS3           │
│  • Modular JavaScript (ES6+), Async/Await, Canvas, FileReader API           │
│  • Capacitor Hybrid Runtime & WebIntoApp Wrapper (Android APK: com.stainscan)│
│  • Bilingual Localization: English (en) & Hiligaynon / Ilonggo (hil)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTPS / REST (JSON + Base64)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BACKEND API TIER (Cloud Web Service)                  │
│  • Python 3.10-slim in Docker Container hosted on Render                    │
│  • Flask 3.0.3 Web Framework + Flask-CORS 4.0.1 (Wildcard CORS)             │
│  • Gunicorn 22.0.0 WSGI Server (1 Worker, 4 Threads, 120s Timeout)          │
│  • Memory footprint optimized < 512 MB RAM for cloud container execution     │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
                       ▼                               ▼
┌────────────────────────────────────┐ ┌──────────────────────────────────────┐
│       MACHINE LEARNING TIER        │ │            DATABASE TIER             │
│  • TensorFlow 2.16.1 / Keras 3.3.0 │ │  • MongoDB Atlas Cloud (M0 Cluster)  │
│  • OpenCV (cv2) for CIELAB CLAHE   │ │  • PyMongo 4.7.2 + certifi TLS       │
│  • Pillow 10.3.0 (Image decoding)  │ │  • Fallback in-memory/JSON store     │
│  • rawpy 0.21+ (Canon CR2 decode)  │ │  • Collections: users, scans,        │
│  • scikit-learn 1.4.2 (Metrics)    │ │    kb, metrics, logs                 │
└────────────────────────────────────┘ └──────────────────────────────────────┘
```

#### Detailed Technology Matrix

| Layer / Role | Technology / Library | Version | Specific Purpose in StainScan |
| :--- | :--- | :--- | :--- |
| **Backend Framework** | Flask | 3.0.3 | REST API routes (`/predict`, `/api/scans`, `/api/metrics`, `/api/kb`, etc.) |
| **API Middleware** | Flask-CORS | 4.0.1 | Cross-Origin Resource Sharing handling for hybrid mobile client requests |
| **Production WSGI** | Gunicorn | 22.0.0 | Multi-threaded HTTP server handling concurrent requests with low RAM |
| **Deep Learning Engine** | TensorFlow / Keras | 2.16.1 / 3.3.0 | MobileNetV2 architecture definition, training routines, and model inference |
| **Computer Vision** | OpenCV (`cv2`) | 4.9.0+ | CIELAB color space transformation and CLAHE adaptive histogram equalization |
| **Image Processing** | Pillow (PIL) | 10.3.0 | Image format conversion, bilinear resizing, luminance crop computation |
| **RAW Decoding** | rawpy | 0.21.0+ | Direct sensor extraction of Canon DSLR `.CR2` RAW grey cotton captures |
| **Evaluation / Data Split** | scikit-learn | 1.4.2 | Stratified 80/20 data splitting, confusion matrix, precision/recall calculations |
| **Array & Cache Math** | NumPy | 1.26.4 | Numerical tensor operations and compressed `.npz` dataset cache handling |
| **Database Driver** | PyMongo / dnspython | 4.7.2 / 2.6.1 | MongoDB Atlas cluster communication over TLS with DNS SRV lookups |
| **Security Certificates** | certifi | 2024.7.4 | Up-to-date Mozilla CA certificate bundle for secure MongoDB Atlas TLS handshakes |
| **Frontend Languages** | HTML5, Vanilla CSS3, JS | ES6+ | Zero-dependency native web UI with dynamic themes, canvas previews, animations |
| **Mobile Packaging** | Capacitor / WebIntoApp | v5+ / Cloud | Native Android APK packaging (`com.stainscan.app`) via WebView bridge |

---

### 1.3 Mobile App Architecture

1. **Hybrid Client Architecture:**
   - The StainScan mobile application runs as a hybrid application wrapped around web technologies (`index.html`, `style.css`, `app.js`).
   - Package name: `com.stainscan.app`.
   - Packaged either through **Capacitor** (developer CLI with Android Studio Gradle toolchain) or **WebIntoApp** (cloud APK bundling).
2. **Dynamic Host Resolution:**
   - In [`app.js`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/app.js) and [`admin.js`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/admin.js), `CONFIG.API_BASE_URL` sniffs the runtime environment:
     - Desktop development environments strictly on `localhost` or `127.0.0.1` target the local development server (`http://localhost:5000`).
     - Any mobile device (Android/iOS User-Agent, touch tablets, Capacitor/Cordova hybrid WebViews, `file://` or `null` origins) and any non-localhost host automatically routes all API calls to the live production server: `https://stainscan-backend-gmbw.onrender.com`.
3. **Camera & Image Acquisition Pipeline:**
   - Uses the HTML5 Media Capture API (`<input type="file" accept="image/*" capture="environment">`) to interface directly with the Android system camera or gallery.
   - Images are loaded into a client-side HTML5 Canvas, rendered in real time, and encoded into Base64 format (`data:image/jpeg;base64,...`) for asynchronous transmission to the `/predict` endpoint.
4. **Resilience & Local Storage:**
   - Scan history, user sessions, and cleaning recipes are mirrored to the browser's `localStorage` (`stainscan_history`, `stainscan_kb`, `stainscan_session`).
   - If network connectivity is interrupted, the client displays cached recommendations and queues logs.
5. **Localization:**
   - Full bilingual interface supporting **English** and **Hiligaynon (Ilonggo)**, accommodating local community users in Western Visayas.

---

### 1.4 Data Storage & Database Configuration

- **Database Engine:** **MongoDB Atlas** (Cloud-hosted NoSQL cluster, database name: `stainscan_db`).
- **Connection Configuration:**
  - Connection string injected via environment variable: `MONGO_URI` (or `MONGODB_URI` / `DATABASE_URL`).
  - Configured with `serverSelectionTimeoutMS=5000`, `connectTimeoutMS=10000`, `socketTimeoutMS=20000`, `maxIdleTimeMS=45000`, `retryWrites=True`, `retryReads=True`, `tls=True`, and `certifi` CA verification.
- **Failover / Network Drop Resilience:**
  - Database operations wrap queries with automatic reconnection on transient network drops (`AutoReconnect`, `ConnectionFailure`, `ServerSelectionTimeoutError`).
  - If MongoDB Atlas is temporarily unreachable during mobile network switching, operations seamlessly fall back to local in-memory storage and [`scans.json`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/scans.json)/[`metrics.json`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/metrics.json), ensuring uninterrupted mobile user experience.

#### Database Collections Schema

```
stainscan_db
 ├── users       ── User accounts & credentials (admin & general users)
 ├── scans       ── Classification records, confidence scores & treatment statuses
 ├── kb          ── Fabric-specific cleaning procedures, material lists & cautions
 ├── metrics     ── Global counters (total scans, successful treatments, category tallies)
 └── logs        ── Audit trail of application events, logins, and CNN predictions
```

1. **`users` Collection:**
   ```json
   {
     "_id": ObjectId("..."),
     "name": "Admin Manager",
     "email": "admin@stainscan.com",
     "password": "...",
     "role": "admin",
     "avatar": "admin"
   }
   ```
2. **`scans` Collection:**
   ```json
   {
     "_id": ObjectId("..."),
     "id": "h_20261006115000_16",
     "email": "user@stainscan.com",
     "stain": "Used Cooking Oil",
     "fabric": "Grey Cotton (100%)",
     "confidence": 99,
     "timestamp": "2026-10-06T11:50:00.000Z",
     "status": "Treated",
     "image": ""
   }
   ```
3. **`kb` (Knowledge Base) Collection:**
   Stores dynamic cleaning recipes indexed by stain category and fabric shade (White Cotton vs. Grey Cotton).
4. **`metrics` Collection:**
   Stores global aggregated application performance indicators:
   ```json
   {
     "_id": "app_metrics",
     "total_scans": 16,
     "successful_treatments": 7,
     "stain_counts": {
       "Used Cooking Oil": 8,
       "Black Ballpen Ink": 3,
       "Mud": 5
     }
   }
   ```
5. **`logs` Collection:**
   Stores system diagnostics and classification traces with timestamps.

---

## 2. Dataset & Preprocessing Pipeline

### 2.1 Exact Dataset Distribution & Counts

The physical dataset was captured under realistic conditions on genuine cotton fabrics, encompassing two distinct fabric substrates: **White 100% Cotton** and **Grey 100% Cotton**.

| Class Name | Display Label | White Cotton (.JPG) | Grey Cotton (.CR2) | Total Images | Class Percentage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Ballpen Ink_cotton` | Black Ballpen Ink | 300 | 300 | **600** | 33.33% |
| `Cooking Oil_cotton` | Used Cooking Oil | 300 | 300 | **600** | 33.33% |
| `Mud_cotton` | Mud | 300 | 300 | **600** | 33.33% |
| **TOTAL** | — | **900 (50.0%)** | **900 (50.0%)** | **1,800** | **100.0%** |

#### Dataset Split Breakdown (Stratified 80/20)

- **Total Dataset Size:** 1,800 physical images.
- **Training Set (80%):** **1,440 images** (480 per class: 240 white cotton + 240 grey cotton).
- **Validation Set (20%):** **360 images** (120 per class: 60 white cotton + 60 grey cotton).
- **Substrate Balance Ratio:** **1:1** (Exactly 50% white cotton and 50% grey cotton in every subset).
- **Dataset Cache:** Stored in compressed NumPy array [`dataset_clahe_224.npz`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/dataset_clahe_224.npz) containing 1,800 pre-processed tensors of shape `(224, 224, 3)`.

---

### 2.2 Preprocessing & Contrast-Invariant Pipeline

#### The Problem: Substrate Contrast Imbalance
On white cotton, stains exhibit prominent optical contrast against the high-luminance background. On neutral-to-dark grey cotton, however, the reflectance of the substrate drops significantly. For instance, used cooking oil forms a dark, translucent spot that absorbs light similarly to the surrounding dark grey fibers, creating ambiguous pixel intensity distributions that cause standard CNNs to fail.

```
                    ┌────────────────────────────────────────┐
                    │ Raw Image (White or Grey Cotton Fabric)│
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │      Bilinear Resizing to (224, 224)   │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ RGB to CIELAB Color Space Conversion   │
                    │   cv2.cvtColor(img, cv2.COLOR_RGB2LAB) │
                    └───────────────────┬────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌──────────────────────┐
              │ L* Channel          │       │ a*, b* Channels      │
              │ (Luminance: 0 - 100)│       │ (Color Opponents)    │
              └──────────┬──────────┘       └──────────┬───────────┘
                         │                             │
                         ▼                             │
              ┌─────────────────────┐                  │
              │ Apply CLAHE         │                  │
              │  clipLimit = 2.0    │                  │
              │  tileGrid = (8, 8)  │                  │
              └──────────┬──────────┘                  │
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Re-merge & Convert back to RGB Space   │
                    │   cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ MobileNetV2 preprocess_input           │
                    │ (Scales pixel values to [-1.0, 1.0])   │
                    └────────────────────────────────────────┘
```

#### Step-by-Step Technical Implementation:

1. **CIELAB Color Space Decoupling:**
   - Standard histogram equalization on RGB destroys color ratios and causes color distortion.
   - The image is converted into CIELAB (`cv2.COLOR_RGB2LAB`), separating lightness ($L^*$) from green-red ($a^*$) and blue-yellow ($b^*$) color dimensions.
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE):**
   - Applied **strictly to the $L^*$ channel**: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`.
   - The image is partitioned into an $8 \times 8$ grid of contextual tiles.
   - Local histograms are equalized independently, boosting low-contrast boundaries between the stain and the grey fabric weave.
   - The `clipLimit=2.0` prevents over-amplification of sensor noise in homogeneous background regions.
3. **Chrominance Protection:**
   - The enhanced $L^*$ channel is recombined with the untouched $a^*$ and $b^*$ channels, and converted back to RGB.
   - **Critical defense point:** This guarantees that the distinctive spectral hues of stains (the amber-yellow of cooking oil, the blue-violet undertones of ballpen ink, and the earthy brown of mud) remain uncorrupted.
4. **Training-Inference Preprocessing Parity:**
   - The exact same preprocessing function [`apply_clahe()`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/server.py#L682-L696) is embedded in [`server.py`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/server.py#L698-L715), preventing train-serve skew.
5. **Data Augmentation Techniques (Applied via `tf.data` during Training):**
   - **Random Flips:** `tf.image.random_flip_left_right` and `random_flip_up_down` to enforce rotational and spatial invariance.
   - **Brightness Shifts:** `tf.image.random_brightness(max_delta=25.0)` to simulate fluctuating indoor lighting environments.
   - **Contrast Scaling:** `tf.image.random_contrast(lower=0.7, upper=1.4)` to simulate varying stain saturation.
   - **HSV Color Jitter:** `tf.image.random_saturation(lower=0.8, upper=1.2)` and `tf.image.random_hue(max_delta=0.05)` to account for varying ambient light color temperatures.

---

### 2.3 Model Validation Performance Metrics

The fine-tuned MobileNetV2 transfer learning model was evaluated on the independent 360-image validation set (120 images per class):

```
                   Precision    Recall    F1-Score    Support
─────────────────────────────────────────────────────────────
Black Ballpen Ink     0.9917    0.9917      0.9917        120
 Used Cooking Oil     1.0000    0.9917      0.9958        120
              Mud     0.9917    1.0000      0.9959        120
─────────────────────────────────────────────────────────────
         Accuracy                           0.9944        360
        Macro Avg     0.9945    0.9944      0.9944        360
     Weighted Avg     0.9945    0.9944      0.9944        360
```

#### Validation Confusion Matrix (360 Images)

```
                     PREDICTED
                Ink    Oil    Mud
ACTUAL  Ink  [  119      0      1  ]  ──> 99.17% Accuracy (119/120)
        Oil  [    1    119      0  ]  ──> 99.17% Accuracy (119/120)
        Mud  [    0      0    120  ]  ──> 100.0% Accuracy (120/120)
```
- **Overall Validation Accuracy:** **99.44%** (358 / 360 correct classifications).
- **Physical Grey Fabric Verification:** Sample tests on unseen Canon CR2 grey cotton photos confirm high-confidence predictions exceeding 90% across all classes.

---

## 3. Deployment & Hosting Infrastructure

### 3.1 Backend Hosting Architecture & Render Configuration

1. **Hosting Platform:** **Render** (Cloud Application Platform).
   - Live Base URL: `https://stainscan-backend-gmbw.onrender.com`.
2. **Containerization Strategy ([`Dockerfile`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/Dockerfile)):**
   - **Base Image:** `python:3.10-slim`.
   - **System Libraries:** Installs `build-essential` and `libgomp1` for optimized CPU OpenMP math execution.
   - **Web Server Command:**
     ```bash
     gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 server:app
     ```
3. **RAM & Concurrency Optimization:**
   - Free-tier cloud instances impose strict memory limits (512 MB RAM).
   - Loading multiple Gunicorn workers (`--workers 2` or `4`) duplicates the TensorFlow runtime and weights in memory, triggering `OOMKilled` (Out of Memory) termination.
   - **Architectural Solution:** StainScan uses **1 master worker with 4 asynchronous threads** (`--threads 4`). The neural network weights are loaded into memory once during application startup, and concurrent HTTP requests share the single model instance safely via thread pooling.
4. **Environment Variables:**
   - `PORT`: Dynamically bound by Render (defaults to 5000).
   - `MONGO_URI`: Encrypted connection string for MongoDB Atlas.
   - `TF_ENABLE_ONEDNN_OPTS=0`: Disables oneDNN floating-point order variance for deterministic inference.

---

### 3.2 Client-Server Communication & API Protocol

The mobile application communicates with the inference engine over secure **HTTPS / REST**:

```
[ Mobile App / Android APK ]
             │
             │  1. POST /predict  { image: "base64...", fabric_color: "Grey Cotton" }
             ▼
[ Flask Backend / Gunicorn ]
             │
             ├──> 2. Decodes Base64 to PIL Image & crops center for luminance check
             ├──> 3. Executes apply_clahe() & preprocess_input()
             ├──> 4. Model.predict() -> Softmax probabilities: [0.01, 0.98, 0.01]
             ├──> 5. Look up fabric-specific recipe from RECOMMENDATION_DATABASE
             ├──> 6. Asynchronously updates scan record & metrics in MongoDB Atlas
             │
             │  7. Returns JSON with classification, confidence, steps, materials, caution
             ▼
[ Mobile App / Android APK ] (Renders animated cleaning checklist & warnings)
```

#### Primary API Endpoints Specification

| Method & Route | Request Payload / Params | Response Payload / Status | Purpose in Application |
| :--- | :--- | :--- | :--- |
| `POST /predict` | `{ "image": "<base64>", "email": "...", "fabric_color": "..." }` | `JSON (stain, confidence, steps, materials, caution)` `[200 OK]` | Core CNN inference, fabric color detection, and dynamic cleaning recipe resolution |
| `GET /health` | None | `{ "status": "healthy", "model_loaded": true, "backend": "TensorFlow/Keras" }` `[200 OK]` | Health probe and container monitoring endpoint |
| `GET /api/model-info` | None | `{ "model_file": "...", "input_shape": [null, 224, 224, 3], "classes": {...} }` `[200 OK]` | Admin diagnostics and model parameter verification |
| `POST /api/register` | `{ "name": "...", "email": "...", "password": "..." }` | `{ "name": "...", "email": "...", "role": "user" }` `[201 Created]` | User registration and profile creation |
| `POST /api/login` | `{ "email": "...", "password": "..." }` | `{ "name": "...", "email": "...", "role": "admin"\|"user" }` `[200 OK]` | User and administrator authentication |
| `GET /api/scans` | `?email=user@stainscan.com` | `Array of scan objects [{ id, stain, fabric, confidence, status }]` `[200 OK]` | History retrieval for user profile and admin tables |
| `PUT /api/scans/<id>` | `{ "status": "Treated" }` | `{ "success": true, "message": "Scan updated" }` `[200 OK]` | Status tracking (Pending vs Treated) |
| `POST /api/treatment-complete` | `{ "scan_id": "h_..." }` | `{ "success": true, "successful_treatments": 8 }` `[200 OK]` | Treatment completion increment with duplicate guard |
| `GET /api/metrics` | None | `{ "total_scans": 16, "successful_treatments": 7, "stain_counts": {...} }` `[200 OK]` | Real-time administrative metrics dashboard |
| `GET /api/kb` & `POST /api/kb` | Admin JSON recipe modifications | `{ "success": true }` `[200 OK]` | CMS management of stain cleaning instructions |
| `GET /api/logs` & `POST /api/logs`| `{ "type": "info", "text": "..." }` | Array of log objects `[200 OK]` | System auditing and error tracking |

---

### 3.3 Dynamic Fabric-Aware Stain Recommendation Engine

StainScan pairs machine learning classification with a **Fabric-Aware Rule-Based Cleaning Engine**. The cleaning procedures adapt according to fabric color to protect garment dyes and fibers:

```
                                  CNN Classifier Result
                                            │
                                            ▼
                                   Detected Stain Class
                       (Cooking Oil / Ballpen Ink / Mud)
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
           Fabric: White Cotton                           Fabric: Grey Cotton
  ┌─────────────────────────────────────┐       ┌─────────────────────────────────────┐
  │ • Aggressive chemical agents        │       │ • Color-safe, non-bleaching agents  │
  │ • Oxygen bleach permitted           │       │ • Strictly NO chlorine/oxygen bleach│
  │ • Hot / warm water allowed          │       │ • Cool / lukewarm water preferred   │
  │ • Direct pre-soak treatments        │       │ • Reverse-side blotting to avoid nap│
  └─────────────────────────────────────┘       └─────────────────────────────────────┘
```

1. **Black Ballpen Ink:**
   - **White Cotton:** 70% Isopropyl alcohol blotting, followed by liquid detergent pre-treatment and machine wash with oxygen bleach.
   - **Grey Cotton:** Reverse-side alcohol blotting using a clean cloth, followed by color-safe detergent. *Strict caution: Never use chlorine bleach or acetone, which strip colored dyes.*
2. **Used Cooking Oil:**
   - **White Cotton:** Baking soda or cornstarch surface absorption (15 mins), concentrated liquid dish soap scrub, followed by hot water rinse.
   - **Grey Cotton:** Mild liquid dish soap applied gently with a soft-bristled brush, followed by lukewarm water rinse. *Strict caution: Avoid abrasive scrubbing to prevent color fading and surface pilling.*
3. **Mud:**
   - **White Cotton:** Allow mud to dry fully, scrape surface crust, pre-soak in warm water with oxygen bleach, and machine wash warm.
   - **Grey Cotton:** Allow mud to dry fully, brush off dry crust, pre-soak in cool water with color-safe liquid detergent. *Strict caution: Use cool water pre-soaks to avoid setting microscopic clay minerals into colored cotton weaves.*

---

## 4. Panel Defense FAQ & Answers

### Question 1: Why did you choose MobileNetV2 over deeper architectures like ResNet-50 or custom shallow CNNs?
> **Defense Answer:**  
> "We selected MobileNetV2 because of its optimal trade-off between feature extraction capacity, computational complexity, and inference latency on mobile and cloud-constrained hardware. Architectures like ResNet-50 contain over 25.6 million parameters and require substantial memory bandwidth, which would exceed the 512 MB RAM constraint of our containerized cloud hosting and induce noticeable latency on mobile clients. Conversely, training a shallow custom CNN from scratch on 1,800 images would risk severe overfitting and struggle to capture subtle fabric textures.  
> MobileNetV2 uses **Inverted Residual Blocks** with **Depthwise Separable Convolutions**, which factorize standard convolutions into spatial filtering and channel-wise point convolutions. This reduces parameter count to only ~2.27 million and computation to ~300 million FLOPs. Furthermore, by initializing with ImageNet pre-trained weights and fine-tuning only the top 20 layers, we transferred robust low- and mid-level edge, texture, and contour filters while achieving an inference latency under 150 milliseconds on CPU."

---

### Question 2: Stains on grey cotton have significantly lower visual contrast than on white cotton. How does your pipeline prevent the model from confusing fabric color with stain identity?
> **Defense Answer:**  
> "We addressed the low-contrast challenge through a two-part approach involving dataset balancing and color-space-aware preprocessing:  
> First, at the dataset level, we eliminated substrate representation bias by curating an exact **1:1 physical balance**: 900 physical photos on white cotton and 900 physical photos on grey cotton, with an identical 300:300 distribution across all three classes.  
> Second, at the preprocessing level, we designed a **CIELAB CLAHE pipeline**. Standard RGB histogram equalization distorts stain colors because red, green, and blue channels are correlated. We convert images into the CIELAB color space to decouple the luminance channel ($L^*$) from the chromatic channels ($a^*$ and $b^*$). We then apply Contrast Limited Adaptive Histogram Equalization with a clip limit of 2.0 over an $8 \times 8$ grid strictly to the $L^*$ channel. This enhances localized contrast around stain borders on grey fabric without causing noise over-amplification, while leaving the chrominance channels untouched. As a result, the model learns the invariant chromatic and textural signature of the stain rather than relying on background fabric brightness."

---

### Question 3: How does your system distinguish dark used cooking oil from mud on dark grey cotton, given their visual similarity?
> **Defense Answer:**  
> "Although used cooking oil and mud can both appear as dark patches on grey cotton to the naked eye, their underlying physical and optical properties create distinct feature maps in a convolutional neural network:  
> 1. **Texture and Specular Reflectance:** Used cooking oil creates a semi-translucent lipid saturation that alters the index of refraction of the cotton weave, creating a smoother texture with subtle specular highlights along thread ridges. Mud, by contrast, is a particulate mineral suspension that dries into an opaque, matte crust with high-frequency surface roughness, micro-cracks, and distinct edge borders.  
> 2. **Chrominance Signatures in CIELAB:** Even on grey fabric, cooking oil exhibits slight warm amber-yellow shifts in the positive $b^*$ channel, whereas dried mud exhibits a distinct earthy brown hue with different $a^*$ and $b^*$ coordinates.  
> MobileNetV2’s depthwise convolutional kernels capture these fine micro-textural gradients and chromatic transitions. In our validation confusion matrix, out of 120 mud samples and 120 oil samples, there was zero cross-misclassification between mud and oil on grey fabric."

---

### Question 4: How did you prevent overfitting and data leakage during model training and evaluation?
> **Defense Answer:**  
> "We enforced strict methodological controls throughout the machine learning pipeline:  
> 1. **Stratified Partitioning:** We performed a stratified 80/20 train/validation split using `scikit-learn` with a fixed seed (42). This guaranteed that each subset contained an exact balance of both stain classes and fabric substrates (240 train / 60 val per class per substrate), preventing class distribution skew.  
> 2. **Data Leakage Prevention:** Data augmentations—including random horizontal/vertical flips, brightness adjustments ($\pm 25$), contrast scaling (0.7 to 1.4), and HSV jitter—were executed dynamically strictly on the training pipeline via `tf.data`. The validation set was never augmented; it received only standard bilinear resizing, CLAHE, and MobileNetV2 normalization.  
> 3. **Controlled Fine-Tuning:** During Phase 1, the entire base model was frozen. During Phase 2 fine-tuning, only the top 20 layers were trained with a reduced learning rate of $10^{-5}$, while all `BatchNormalization` layers across the base network were explicitly frozen to prevent corruption of the pre-trained running mean and variance. A 40% Dropout layer on the head further prevented co-adaptation of weights."

---

### Question 5: How does the system handle edge cases in the mobile client, such as network failure or server cold starts on Render?
> **Defense Answer:**  
> "Because our backend is deployed on Render's cloud container tier, inactive instances spin down after periods of inactivity, causing a temporary cold start of approximately 30 to 45 seconds when a new request arrives. We addressed this through both backend and frontend engineering:  
> On the backend, we run Gunicorn with 4 threads and pre-warm the model upon module import by executing a dummy inference during startup, ensuring the TensorFlow computational graph is compiled before serving incoming traffic.  
> On the mobile client ([`app.js`](file:///c:/Users/John%20Michael%20Dohinog/Desktop/Mobile%20Application%20of%20StainScan/app.js)), we built an asynchronous polling controller with a multi-step animated progress overlay in both English and Hiligaynon. If a cold-start delay occurs, the client maintains the connection and presents informative status messages rather than timing out. If network connectivity fails completely, the app gracefully degrades: it notifies the user, provides an offline simulation and diagnostic mode, and falls back to locally cached knowledge base recipes stored in `localStorage` so the user is never left without cleaning assistance."

---

## 5. Quick Defense Checklist & Technical Reference

- [x] **Model Backbone:** MobileNetV2 (154 layers, ImageNet weights, Top 20 layers fine-tuned).
- [x] **Classification Head:** `GlobalAveragePooling2D` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.40)` $\rightarrow$ `Dense(3, softmax)`.
- [x] **Loss & Optimizer:** Categorical Crossentropy, Adam ($10^{-3}$ Phase 1, $10^{-5}$ Phase 2).
- [x] **Input Resolution:** $224 \times 224 \times 3$ RGB.
- [x] **Key Preprocessing:** CIELAB CLAHE ($L^*$ channel, clipLimit=2.0, tileGrid=(8,8)) + `preprocess_input` ([-1.0, 1.0]).
- [x] **Total Dataset:** 1,800 physical photos (900 White Cotton, 900 Grey Cotton | 600 per class).
- [x] **Classes:** Used Cooking Oil, Black Ballpoint Pen Ink, Mud.
- [x] **Validation Accuracy:** **99.44%** on 360 unseen test images (358/360 correct).
- [x] **Backend Architecture:** Dockerized Flask 3.0.3 + Gunicorn (1 worker, 4 threads) on Render.
- [x] **Database:** MongoDB Atlas (M0 Cluster) with PyMongo TLS + in-memory/JSON fallback.
- [x] **Mobile Runtime:** Hybrid web client (HTML5/CSS3/ES6 JS) packaged via Capacitor / WebIntoApp into Android APK.
- [x] **Localization:** English and Hiligaynon (Ilonggo).
