import base64
from io import BytesIO
import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import certifi
import numpy as np

IMAGE_SIZE   = (224, 224)                 # Exact match with MobileNetV2 input shape
MODEL_PATH   = "stainscan_model_v3_balanced.keras"  # Primary retrained model file
LOADED_MODEL_FILE = None
LAST_MODEL_ERROR = None
CLASS_MAPPING_RAW = {
    0: "ballpen ink_cotton",
    1: "Cooking Oil_cotton",
    2: "Mud_cotton"
}

CLASS_MAPPING_DISPLAY = {
    0: "Black Ballpen Ink",
    1: "Used Cooking Oil",
    2: "Mud"
}

# Try importing TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    HAS_TF = True
except ModuleNotFoundError:
    print("TensorFlow not found. Prediction API will run in demo/fallback mode.")
    HAS_TF = False



# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def index():
    return app.send_static_file('index.html')

# --- Dynamic Fabric-Specific Stain Recommendations Database ---
RECOMMENDATION_DATABASE = {
    "ballpen ink": {
        "white": {
            "stain": "Black Ballpen Ink",
            "fabric_color": "White Cotton",
            "materials": ["Paper Towels / Clean Cloth", "70% Isopropyl Alcohol", "Cotton Swab", "Liquid Laundry Detergent", "Oxygen Bleach"],
            "steps": [
                "Place a paper towel or clean cloth underneath the stained area.",
                "Dab 70% Isopropyl Alcohol onto a cotton swab and blot gently from the outer edges inward.",
                "Apply liquid laundry detergent directly to the stain and let sit for 5-10 minutes.",
                "Rinse thoroughly with cold water, then machine wash in warm water with oxygen bleach."
            ],
            "caution": "Do not use chlorine bleach directly on ink spots, as it can fix the pigment into cotton fibers."
        },
        "grey": {
            "stain": "Black Ballpen Ink",
            "fabric_color": "Grey Cotton",
            "materials": ["Paper Towels", "70% Isopropyl Alcohol", "Clean Cloth", "Color-Safe Liquid Detergent"],
            "steps": [
                "Turn garment inside out and place paper towel under the front.",
                "Dab 70% Isopropyl Alcohol on a cloth and gently blot the reverse side to push ink out.",
                "Apply color-safe liquid detergent directly to the spot.",
                "Rinse with cool water and wash on a cool cycle with color-safe detergent."
            ],
            "caution": "Do not use acetone or chlorine bleach on grey fabric, as it will strip the dye."
        }
    },
    "cooking oil": {
        "white": {
            "stain": "Used Cooking Oil",
            "fabric_color": "White Cotton",
            "materials": ["Baking Soda / Cornstarch", "Concentrated Liquid Dish Soap", "Hot Water", "Clean Brush / Cloth"],
            "steps": [
                "Sprinkle baking soda or cornstarch on fresh oil for 10-15 minutes to absorb surface lipid, then brush away.",
                "Apply concentrated liquid dish soap directly onto the stain.",
                "Gently work soap into fibers and let sit for 15 minutes.",
                "Rinse with hot water, then wash in warm water."
            ],
            "caution": "Ensure stain is fully removed before machine drying; heat permanently sets oil."
        },
        "grey": {
            "stain": "Used Cooking Oil",
            "fabric_color": "Grey Cotton",
            "materials": ["Mild Liquid Dish Soap", "Soft-Bristled Brush / Clean Cloth", "Lukewarm Water", "Color-Safe Detergent"],
            "steps": [
                "Dab mild liquid dish soap directly onto the grease spot.",
                "Softly work soap into the fabric using a soft-bristled brush or clean cloth.",
                "Let sit for 10-15 minutes to break down grease.",
                "Rinse with lukewarm water and wash in a standard warm cycle with color-safe detergent."
            ],
            "caution": "Avoid aggressive scrubbing to prevent color fading or fuzzing on grey cotton."
        }
    },
    "mud": {
        "white": {
            "stain": "Mud",
            "fabric_color": "White Cotton",
            "materials": ["Dull Knife or Brush", "Oxygen Bleach", "Warm Water", "Liquid Laundry Detergent"],
            "steps": [
                "Allow mud to dry completely (never clean wet mud).",
                "Gently scrape off hardened top crust with a dull knife or brush.",
                "Pre-soak in warm water with oxygen bleach for 30 minutes.",
                "Apply liquid detergent directly to remaining marks and machine wash warm."
            ],
            "caution": "Washing wet mud forces fine clay deep into fabric weave."
        },
        "grey": {
            "stain": "Mud",
            "fabric_color": "Grey Cotton",
            "materials": ["Soft-Bristle Brush", "Color-Safe Liquid Detergent", "Cool Water"],
            "steps": [
                "Allow mud to dry completely.",
                "Gently brush off dry surface crust using a soft-bristle brush.",
                "Pre-soak in cool water mixed with color-safe liquid detergent for 20-30 minutes.",
                "Lightly rub remaining marks with liquid detergent and wash on a cool cycle."
            ],
            "caution": "Use cool water pre-soaks to prevent fine soil particles from setting into colored fabric."
        }
    }
}

def detect_fabric_color_from_image(pil_image: Image.Image) -> str:
    """
    Detects whether the cotton fabric is White or Grey based on central crop luminance.
    """
    try:
        w, h = pil_image.size
        crop_box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
        cropped = pil_image.crop(crop_box).convert("L")
        mean_lum = float(np.mean(np.array(cropped)))
        return "Grey Cotton" if mean_lum < 165 else "White Cotton"
    except Exception:
        return "White Cotton"

def get_recommendation(stain_name: str, fabric_color_or_type: str = None, image: Image.Image = None) -> dict:
    """
    Looks up fabric-specific cleaning steps and safety caution based on stain class
    and fabric color/type.
    """
    stain_str = str(stain_name or "").lower()
    if "ballpen" in stain_str or "ink" in stain_str:
        stain_key = "ballpen ink"
    elif "oil" in stain_str or "cooking" in stain_str:
        stain_key = "cooking oil"
    elif "mud" in stain_str:
        stain_key = "mud"
    else:
        stain_key = "cooking oil"

    color_str = str(fabric_color_or_type or "").lower().strip()
    if "grey" in color_str or "gray" in color_str:
        color_key = "grey"
    elif "white" in color_str:
        color_key = "white"
    elif image is not None:
        detected = detect_fabric_color_from_image(image)
        color_key = "grey" if "grey" in detected.lower() else "white"
    else:
        color_key = "white"

    rec = RECOMMENDATION_DATABASE[stain_key][color_key]
    return rec

DEFAULT_KB = {
    "Black Ballpen Ink (White Cotton)": RECOMMENDATION_DATABASE["ballpen ink"]["white"],
    "Black Ballpen Ink (Grey Cotton)": RECOMMENDATION_DATABASE["ballpen ink"]["grey"],
    "Used Cooking Oil (White Cotton)": RECOMMENDATION_DATABASE["cooking oil"]["white"],
    "Used Cooking Oil (Grey Cotton)": RECOMMENDATION_DATABASE["cooking oil"]["grey"],
    "Mud (White Cotton)": RECOMMENDATION_DATABASE["mud"]["white"],
    "Mud (Grey Cotton)": RECOMMENDATION_DATABASE["mud"]["grey"],
    "Black Ballpen Ink": RECOMMENDATION_DATABASE["ballpen ink"]["white"],
    "Used Cooking Oil": RECOMMENDATION_DATABASE["cooking oil"]["white"],
    "Mud": RECOMMENDATION_DATABASE["mud"]["white"]
}

# Initialize MongoDB
MONGO_URI = os.environ.get("MONGO_URI", "")
db = None
is_demo_mode = True

if MONGO_URI:
    if "tlsAllowInvalidCertificates" not in MONGO_URI:
        separator = "&" if "?" in MONGO_URI else "?"
        MONGO_URI = f"{MONGO_URI}{separator}tlsAllowInvalidCertificates=true"
        
    try:
        client = MongoClient(
            MONGO_URI, 
            serverSelectionTimeoutMS=5000, 
            tls=True, 
            tlsAllowInvalidCertificates=True
        )
        db = client["stainscan_db"]
        client.server_info() # Trigger quick connection check
        is_demo_mode = False
        print("Connected to MongoDB Atlas successfully.")
        
        # Populate Default Knowledge Base if collection is empty
        if db["kb"].count_documents({}) == 0:
            db["kb"].insert_one({"_id": "current_kb", "data": DEFAULT_KB})
            print("Populated default knowledge base in MongoDB.")
            
        # Create default Admin if empty
        if db["users"].count_documents({}) == 0:
            default_users = [
                { "name": "Admin Manager", "email": "admin@stainscan.com", "password": "admin123", "role": "admin", "avatar": "admin" },
                { "name": "John Doe", "email": "user@stainscan.com", "password": "user123", "role": "user", "avatar": "John" }
            ]
            db["users"].insert_many(default_users)
            print("Populated default users in MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}. Running in memory fallback mode.")
        db = None
        is_demo_mode = True
else:
    print("MONGO_URI not configured. Running in memory fallback mode.")
    is_demo_mode = True

SCANS_FILE = "scans.json"
METRICS_FILE = "metrics.json"

def _load_initial_scans():
    if os.path.exists(SCANS_FILE):
        try:
            with open(SCANS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading initial scans from file: {e}")
    return []

# Fallback memory stores
if is_demo_mode:
    mock_db = {
        "users": [
            { "name": "Admin Manager", "email": "admin@stainscan.com", "password": "admin123", "role": "admin", "avatar": "admin" },
            { "name": "John Doe", "email": "user@stainscan.com", "password": "user123", "role": "user", "avatar": "John" }
        ],
        "scans": _load_initial_scans(),
        "logs": [
            { "time": datetime.utcnow().isoformat(), "type": "info", "text": "Database initialized successfully (Demo Mode)." }
        ],
        "kb": DEFAULT_KB,
        "metrics": {}
    }

def get_metrics():
    if not is_demo_mode and db is not None:
        try:
            doc = db["metrics"].find_one({"_id": "app_metrics"})
            if doc:
                return {
                    "total_scans": int(doc.get("total_scans", 0)),
                    "successful_treatments": int(doc.get("successful_treatments", 0)),
                    "stain_counts": doc.get("stain_counts", {})
                }
        except Exception as e:
            print(f"Error fetching metrics from MongoDB: {e}")

    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    all_scans = get_scans()
    stain_counts = {"Used Cooking Oil": 0, "Black Ballpen Ink": 0, "Mud": 0}
    for s in all_scans:
        st = s.get("stain")
        if st in stain_counts:
            stain_counts[st] += 1

    metrics = {
        "total_scans": len(all_scans),
        "successful_treatments": len([s for s in all_scans if s.get("status") == "Treated"]),
        "stain_counts": stain_counts
    }
    save_metrics(metrics)
    return metrics

def save_metrics(metrics):
    if not is_demo_mode and db is not None:
        try:
            db["metrics"].replace_one({"_id": "app_metrics"}, {"_id": "app_metrics", **metrics}, upsert=True)
        except Exception as e:
            print(f"Error saving metrics to MongoDB: {e}")

    try:
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Error saving metrics to file: {e}")

    if is_demo_mode and "metrics" in mock_db:
        mock_db["metrics"] = metrics

def increment_scan_count(stain_name=None, fabric_name=None, confidence=None, email=None):
    metrics = get_metrics()
    metrics["total_scans"] = metrics.get("total_scans", 0) + 1

    if stain_name:
        stain_counts = metrics.setdefault("stain_counts", {})
        stain_counts[stain_name] = stain_counts.get(stain_name, 0) + 1

    save_metrics(metrics)

    # Also log a persistent scan record in the scans table
    scan_id = "h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + str(metrics["total_scans"])
    new_scan = {
        "id": scan_id,
        "email": email or "mobile_user@stainscan.com",
        "stain": stain_name or "Unknown",
        "fabric": fabric_name or "Cotton Fabric (100%)",
        "confidence": int(confidence or 90),
        "timestamp": datetime.utcnow().isoformat(),
        "status": "Pending",
        "image": ""
    }
    try:
        add_scan(new_scan)
        add_log("info", f"CNN Scan #{metrics['total_scans']} classified: {stain_name} ({confidence}%)")
    except Exception as e:
        print(f"Error logging scan record: {e}")

    return metrics

def increment_treatment_count(scan_id=None):
    metrics = get_metrics()
    metrics["successful_treatments"] = metrics.get("successful_treatments", 0) + 1
    save_metrics(metrics)

    if scan_id:
        try:
            if not is_demo_mode and db is not None:
                from bson.objectid import ObjectId
                query = {"$or": [{"_id": ObjectId(scan_id)}, {"id": scan_id}]} if ObjectId.is_valid(scan_id) else {"id": scan_id}
                db["scans"].update_one(query, {"$set": {"status": "Treated"}})
            else:
                s = next((x for x in mock_db["scans"] if x.get("id") == scan_id or x.get("_id") == scan_id), None)
                if s:
                    s["status"] = "Treated"
                    _save_scans_file()
        except Exception as e:
            print(f"Error updating scan status in increment_treatment_count: {e}")

    add_log("info", f"Treatment #{metrics['successful_treatments']} completed successfully (Scan ID: {scan_id or 'N/A'})")
    return metrics

def _save_scans_file():
    try:
        with open(SCANS_FILE, "w", encoding="utf-8") as f:
            json.dump(mock_db["scans"], f, indent=2)
    except Exception as e:
        print(f"Error saving scans file: {e}")

def get_users():
    if not is_demo_mode and db is not None:
        users = list(db["users"].find({}))
        for u in users:
            u["_id"] = str(u["_id"])
        return users
    return mock_db["users"]

def add_user(user):
    if not is_demo_mode and db is not None:
        result = db["users"].insert_one(user)
        user["_id"] = str(result.inserted_id)
        return user
    mock_db["users"].append(user)
    return user

def find_user_by_email(email):
    if not is_demo_mode and db is not None:
        user = db["users"].find_one({"email": email})
        if user:
            user["_id"] = str(user["_id"])
        return user
    return next((u for u in mock_db["users"] if u["email"] == email), None)

def get_kb():
    if not is_demo_mode and db is not None:
        doc = db["kb"].find_one({"_id": "current_kb"})
        return doc["data"] if doc else DEFAULT_KB
    return mock_db["kb"]

def save_kb(kb_data):
    if not is_demo_mode and db is not None:
        db["kb"].replace_one({"_id": "current_kb"}, {"_id": "current_kb", "data": kb_data}, upsert=True)
    else:
        mock_db["kb"] = kb_data

def get_scans(email=None):
    if not is_demo_mode and db is not None:
        query = {"email": email} if email else {}
        scans = list(db["scans"].find(query).sort("timestamp", -1))
        for s in scans:
            s["_id"] = str(s["_id"])
        return scans
    if email:
        return [s for s in mock_db["scans"] if s["email"] == email]
    return sorted(mock_db["scans"], key=lambda s: s.get("timestamp", ""), reverse=True)

def add_scan(scan):
    if not is_demo_mode and db is not None:
        result = db["scans"].insert_one(scan)
        scan["_id"] = str(result.inserted_id)
        return scan
    if not scan.get("id"):
        scan["id"] = "h_" + str(len(mock_db["scans"]) + 1)
    mock_db["scans"].insert(0, scan)
    _save_scans_file()
    return scan

def get_logs():
    if not is_demo_mode and db is not None:
        logs = list(db["logs"].find({}).sort("time", -1).limit(100))
        for l in logs:
            l["_id"] = str(l["_id"])
        return logs
    return mock_db["logs"][:100]

def add_log(log_type, text):
    log_entry = {
        "time": datetime.utcnow().isoformat(),
        "type": log_type,
        "text": text
    }
    if not is_demo_mode and db is not None:
        db["logs"].insert_one(log_entry)
    else:
        mock_db["logs"].insert(0, log_entry)

# --- DATABASE API ROUTES ---

@app.route("/api/kb", methods=["GET", "POST", "OPTIONS"])
def api_kb():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "No recipe data provided"}), 400
        save_kb(data)
        add_log("info", "Knowledge base recipes updated by admin")
        return jsonify({"success": True, "message": "Knowledge base updated successfully"})
    return jsonify(get_kb())

@app.route("/api/recommendation", methods=["GET", "POST", "OPTIONS"])
def api_recommendation():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    if request.method == "POST":
        data = request.get_json() or {}
        stain = data.get("stain") or data.get("stain_class") or "Used Cooking Oil"
        fabric = data.get("fabric_color") or data.get("fabric_type") or data.get("fabric") or "White Cotton"
    else:
        stain = request.args.get("stain") or request.args.get("stain_class") or "Used Cooking Oil"
        fabric = request.args.get("fabric_color") or request.args.get("fabric_type") or request.args.get("fabric") or "White Cotton"
    return jsonify(get_recommendation(stain, fabric))


@app.route("/api/register", methods=["POST", "OPTIONS"])
def api_register():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    data = request.get_json()
    if not data or "email" not in data or "password" not in data or "name" not in data:
        return jsonify({"error": "Required registration parameters missing"}), 400
    
    email = data["email"].strip().lower()
    name = data["name"].strip()
    password = data["password"]
    avatar = data.get("avatar", name)
    
    existing = find_user_by_email(email)
    if existing:
        return jsonify({"error": "Email address already registered"}), 409
        
    new_user = {
        "name": name,
        "email": email,
        "password": password,
        "role": "user",
        "avatar": avatar
    }
    
    saved_user = add_user(new_user)
    add_log("info", f"New user registered: {email}")
    return jsonify({
        "name": saved_user["name"],
        "email": saved_user["email"],
        "role": saved_user["role"],
        "avatar": saved_user["avatar"]
    }), 201

@app.route("/api/login", methods=["POST", "OPTIONS"])
def api_login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    data = request.get_json()
    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Missing login credentials"}), 400
        
    email = data["email"].strip().lower()
    password = data["password"]
    
    user = find_user_by_email(email)
    if user and user["password"] == password:
        add_log("info", f"User login successful: {email}")
        return jsonify({
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "avatar": user.get("avatar", user["name"])
        })
    
    add_log("warning", f"Failed login attempt for: {email}")
    return jsonify({"error": "Invalid email credentials or password"}), 401

@app.route("/api/users", methods=["GET", "OPTIONS"])
def api_users():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    users = get_users()
    public_users = []
    for u in users:
        public_users.append({
            "name": u["name"],
            "email": u["email"],
            "role": u["role"],
            "avatar": u.get("avatar", u["name"])
        })
    return jsonify(public_users)

@app.route("/api/scans", methods=["GET", "POST", "OPTIONS"])
def api_scans():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    
    if request.method == "POST":
        data = request.get_json()
        if not data or "email" not in data or "stain" not in data or "fabric" not in data or "confidence" not in data:
            return jsonify({"error": "Missing scan properties"}), 400
            
        new_scan = {
            "id": data.get("id", ""),
            "email": data["email"].strip().lower(),
            "stain": data["stain"],
            "fabric": data["fabric"],
            "confidence": int(data["confidence"]),
            "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
            "status": data.get("status", "Pending"),
            "image": data.get("image", "")
        }
        
        saved_scan = add_scan(new_scan)
        add_log("info", f"New stain scan logged for: {new_scan['email']} ({new_scan['stain']})")
        return jsonify(saved_scan), 201
        
    email = request.args.get("email")
    return jsonify(get_scans(email))

@app.route("/api/scans/<scan_id>", methods=["PUT", "PATCH", "OPTIONS"])
def api_update_scan(scan_id):
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
        
    data = request.get_json()
    if not data or "status" not in data:
        return jsonify({"error": "Missing status parameter"}), 400
        
    new_status = data["status"]
    
    if not is_demo_mode and db is not None:
        from bson.objectid import ObjectId
        try:
            query = {}
            if ObjectId.is_valid(scan_id):
                query = {"$or": [{"_id": ObjectId(scan_id)}, {"id": scan_id}]}
            else:
                query = {"$or": [{"id": scan_id}, {"_id": scan_id}]}
                
            result = db["scans"].update_one(query, {"$set": {"status": new_status}})
            if result.matched_count > 0:
                if new_status == "Treated":
                    increment_treatment_count(scan_id=scan_id)
                add_log("info", f"Scan {scan_id} status updated to {new_status}")
                return jsonify({"success": True, "message": f"Scan updated to {new_status}"})
        except Exception as e:
            return jsonify({"error": f"Failed to update scan: {str(e)}"}), 500
    else:
        # Mock/Demo database fallback update
        scan = next((s for s in mock_db["scans"] if s.get("id") == scan_id or s.get("_id") == scan_id), None)
        if scan:
            scan["status"] = new_status
            _save_scans_file()
            if new_status == "Treated":
                increment_treatment_count(scan_id=scan_id)
            add_log("info", f"Scan {scan_id} status updated to {new_status} (Demo Mode)")
            return jsonify({"success": True, "message": f"Scan updated to {new_status}"})
            
    return jsonify({"error": "Scan not found"}), 404

@app.route("/api/treatment-complete", methods=["POST", "OPTIONS"])
def api_treatment_complete():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    data = request.get_json(silent=True) or {}
    scan_id = data.get("scan_id") or data.get("id")
    metrics = increment_treatment_count(scan_id=scan_id)
    return jsonify({
        "success": True,
        "message": "Treatment marked complete and metrics updated",
        "successful_treatments": metrics.get("successful_treatments", 0),
        "metrics": metrics
    }), 200

@app.route("/api/metrics", methods=["GET", "OPTIONS"])
def api_metrics():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    metrics = get_metrics()
    all_scans = get_scans()
    metrics["total_scans"] = max(metrics.get("total_scans", 0), len(all_scans))
    metrics["successful_treatments"] = max(metrics.get("successful_treatments", 0), len([s for s in all_scans if s.get("status") == "Treated"]))
    return jsonify(metrics), 200

@app.route("/api/logs", methods=["GET", "POST", "OPTIONS"])
def api_logs():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    if request.method == "POST":
        data = request.get_json()
        if not data or "type" not in data or "text" not in data:
            return jsonify({"error": "Missing log parameters"}), 400
        add_log(data["type"], data["text"])
        return jsonify({"success": True})
    return jsonify(get_logs())

# ── Global model handle ───────────────────────────────────────────────────────
model = None

def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel
    in CIELAB color space, matching the dynamic contrast training pipeline.
    """
    try:
        import cv2
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:
        return img_rgb


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image to match the MobileNetV2 transfer learning pipeline:
      1. Convert to RGB color space
      2. Resize to IMAGE_SIZE (224x224) using Bilinear interpolation
      3. Apply CIELAB CLAHE dynamic contrast equalization
      4. Apply MobileNetV2 preprocess_input to scale pixel values to [-1.0, 1.0]
      5. Expand batch dimension -> shape (1, 224, 224, 3)
    """
    img = pil_image.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    enhanced = apply_clahe(arr)
    if HAS_TF:
        arr_norm = preprocess_input(enhanced.astype(np.float32))
    else:
        arr_norm = (enhanced.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(arr_norm, axis=0)   # (1, 224, 224, 3)


if HAS_TF:
    class SafeGlorotUniform(keras.initializers.GlorotUniform):
        def __init__(self, seed=None, **kwargs):
            super().__init__(seed=seed)

    CUSTOM_MODEL_OBJECTS = {
        "GlorotUniform": SafeGlorotUniform,
        "GlorotNormal": keras.initializers.GlorotNormal
    }
    try:
        keras.saving.get_custom_objects()["GlorotUniform"] = SafeGlorotUniform
        keras.saving.get_custom_objects()["GlorotNormal"] = keras.initializers.GlorotNormal
    except Exception:
        pass

    try:
        def _patch_init(cls):
            orig_init = cls.__init__
            def patched_init(self, *args, **kwargs):
                kwargs.pop("input_axes", None)
                kwargs.pop("output_axes", None)
                orig_init(self, *args, **kwargs)
            cls.__init__ = patched_init

        _patch_init(keras.initializers.GlorotUniform)
        _patch_init(keras.initializers.GlorotNormal)
        import keras.src.initializers.random_initializers as kri
        _patch_init(kri.GlorotUniform)
        _patch_init(kri.GlorotNormal)
    except Exception as e:
        print(f"Warning setting initializer patch: {e}")
else:
    CUSTOM_MODEL_OBJECTS = {}

CANDIDATE_ERRORS = {}

def load_model(force_reload=False):
    global model, LOADED_MODEL_FILE, LAST_MODEL_ERROR, CANDIDATE_ERRORS
    if model is not None and not force_reload:
        return

    if not HAS_TF:
        print("Warning: TensorFlow not installed. Prediction API will run in demo/fallback mode.")
        return

    candidate_paths = [
        MODEL_PATH,
        "stainscan_model_v3_balanced.keras",
        "stain_model.keras",
        "stainscan_model_v3_balanced.h5",
        "stain_model.h5",
        "stainscan_model.h5"
    ]

    loaded_any = False
    for p in candidate_paths:
        if not os.path.exists(p):
            CANDIDATE_ERRORS[p] = "File does not exist"
            continue
        print(f"Attempting model load from '{p}'...")
        try:
            loaded = keras.models.load_model(p, compile=False, custom_objects=CUSTOM_MODEL_OBJECTS)
            dummy = np.zeros((1,) + IMAGE_SIZE + (3,), dtype=np.float32)
            loaded.predict(dummy, verbose=0)
            model = loaded
            LOADED_MODEL_FILE = p
            LAST_MODEL_ERROR = None
            CANDIDATE_ERRORS[p] = "SUCCESS"
            loaded_any = True
            print(f"Model loaded successfully from '{p}'. Input shape: {model.input_shape}")
            break
        except Exception as e:
            print(f"Error loading model from '{p}': {e}")
            err_msg = f"{type(e).__name__}: {str(e)}"
            CANDIDATE_ERRORS[p] = err_msg
            LAST_MODEL_ERROR = f"Failed '{p}': {err_msg}"

    if not loaded_any:
        print(f"Warning: No valid model could be loaded. Running in demo/fallback mode.")
        model = None
        LOADED_MODEL_FILE = None

# Pre-load model at module import so workers are warm
try:
    load_model()
except Exception as e:
    print(f"Startup model pre-load exception: {e}")

# CORS headers handled dynamically by Flask-CORS middleware

@app.route("/health", methods=["GET"])
def health_check():
    global model, LOADED_MODEL_FILE
    if model is None:
        load_model()
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_file": LOADED_MODEL_FILE or MODEL_PATH,
        "configured_model_path": MODEL_PATH,
        "input_shape": list(model.input_shape) if (model is not None and hasattr(model, "input_shape")) else [None, 224, 224, 3],
        "backend": "TensorFlow/Keras" if HAS_TF else "demo",
        "load_error": LAST_MODEL_ERROR
    })

@app.route("/api/model-info", methods=["GET", "OPTIONS"])
def api_model_info():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    global model, LOADED_MODEL_FILE
    if model is None:
        load_model()
    
    file_to_check = LOADED_MODEL_FILE if (LOADED_MODEL_FILE and os.path.exists(LOADED_MODEL_FILE)) else (MODEL_PATH if os.path.exists(MODEL_PATH) else None)
    file_size = os.path.getsize(file_to_check) if (file_to_check and os.path.exists(file_to_check)) else None

    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_file": LOADED_MODEL_FILE or MODEL_PATH,
        "configured_model_path": MODEL_PATH,
        "input_shape": list(model.input_shape) if (model is not None and hasattr(model, "input_shape")) else [None, 224, 224, 3],
        "output_shape": list(model.output_shape) if (model is not None and hasattr(model, "output_shape")) else [None, 3],
        "classes": CLASS_MAPPING_DISPLAY,
        "backend": "TensorFlow/Keras" if HAS_TF else "demo",
        "model_size_bytes": file_size,
        "load_error": LAST_MODEL_ERROR,
        "candidate_errors": CANDIDATE_ERRORS,
        "tf_version": tf.__version__ if HAS_TF else None,
        "keras_version": keras.__version__ if HAS_TF else None
    }), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    global model
    if model is None:
        load_model()

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided in the request payload."}), 400

    base64_str = data["image"]
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # Check fabric color / type in request context
    fabric_context = data.get("fabric_color") or data.get("fabric_type") or data.get("fabric") or ""

    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Demo/fallback mode when model is unavailable
        if model is None:
            print("Server running in Demo/Simulation Fallback mode.")
            img_w, img_h = image.size
            pseudo_seed = (img_w + img_h) % 3
            stains = ["Black Ballpen Ink", "Used Cooking Oil", "Mud"]
            raw_stains = ["ballpen ink_cotton", "Cooking Oil_cotton", "Mud_cotton"]
            confidences = [89, 92, 85]
            predicted_raw = raw_stains[pseudo_seed]
            predicted_label = stains[pseudo_seed]
            confidence_percent = confidences[pseudo_seed]

            rec = get_recommendation(predicted_label, fabric_context, image=image)
            increment_scan_count(
                stain_name=predicted_label,
                fabric_name=f"{rec['fabric_color']} (100%)",
                confidence=confidence_percent,
                email=data.get("email")
            )
            return jsonify({
                "stain_classification": predicted_raw,
                "fabric_material":      "Cotton",
                "fabric_color":         rec["fabric_color"],
                "confidence":           confidence_percent,
                "stain":                predicted_label,
                "fabric":               f"{rec['fabric_color']} (100%)",
                "steps":                rec["steps"],
                "caution":              rec["caution"],
                "materials":            rec["materials"],
                "recommendation": {
                    "steps": rec["steps"],
                    "caution": rec["caution"],
                    "materials": rec["materials"],
                    "fabric_color": rec["fabric_color"]
                },
                "message":              "Demo prediction (model file not found)"
            })

        # Preprocess and run TensorFlow inference
        input_arr   = preprocess_image(image)              # (1, 224, 224, 3)
        predictions = model.predict(input_arr, verbose=0)  # (1, 3) softmax probabilities
        probs       = predictions[0]                        # (3,)
        predicted_idx    = int(np.argmax(probs))
        confidence_percent = int(round(float(probs[predicted_idx]) * 100))
        predicted_raw    = CLASS_MAPPING_RAW.get(predicted_idx, "Unknown")
        predicted_label  = CLASS_MAPPING_DISPLAY.get(predicted_idx, predicted_raw)

        # Retrieve dynamic recommendation based on stain and fabric context
        rec = get_recommendation(predicted_label, fabric_context, image=image)
        increment_scan_count(
            stain_name=predicted_label,
            fabric_name=f"{rec['fabric_color']} (100%)",
            confidence=confidence_percent,
            email=data.get("email")
        )

        print(f"Prediction: {predicted_raw} / {predicted_label} ({confidence_percent}% confidence) | "
              f"Fabric: {rec['fabric_color']} | Probs: {[round(float(p)*100,1) for p in probs]}")

        # Return actual model prediction with fabric-specific steps and caution
        return jsonify({
            "stain_classification": predicted_raw,
            "fabric_material":      "Cotton",
            "fabric_color":         rec["fabric_color"],
            "confidence":           confidence_percent,
            "stain":                predicted_label,
            "fabric":               f"{rec['fabric_color']} (100%)",
            "steps":                rec["steps"],
            "caution":              rec["caution"],
            "materials":            rec["materials"],
            "recommendation": {
                "steps": rec["steps"],
                "caution": rec["caution"],
                "materials": rec["materials"],
                "fabric_color": rec["fabric_color"]
            }
        })

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    load_model(force_reload=True)
    # Run server dynamically binding to port from environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
