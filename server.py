import base64
from io import BytesIO
import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datetime import datetime
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import (
    PyMongoError,
    AutoReconnect,
    ServerSelectionTimeoutError,
    NetworkTimeout,
    ConnectionFailure
)
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

# Configure CORS to allow cross-origin requests from any origin (*) across all API routes (/*)
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept"],
            "expose_headers": ["Content-Type", "Authorization"],
            "max_age": 86400
        }
    }
)

@app.before_request
def handle_preflight_options():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
        return response

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
    return response

@app.route("/")
def index():
    return app.send_static_file('index.html')

@app.route("/admin")
@app.route("/admin.html")
def admin_page():
    return app.send_static_file('admin.html')

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

# Initialize MongoDB Connection & Resilient Database Handling
MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI") or os.environ.get("DATABASE_URL") or ""
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "stainscan_db")

client = None
db = None
is_demo_mode = True
_last_connect_attempt = 0

SCANS_FILE = "scans.json"
METRICS_FILE = "metrics.json"
USERS_FILE = "users.json"

def _load_initial_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list) and len(content) > 0:
                    return content
        except Exception as e:
            print(f"Error loading initial users from file: {e}")
    return [
        { "name": "John Michael Dohinog", "email": "admin@stainscan.com", "password": "admin123", "role": "admin", "avatar": "System Root" }
    ]

def _save_users_file():
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(mock_db["users"], f, indent=2)
    except Exception as e:
        print(f"Error saving users file: {e}")

def _load_initial_scans():
    if os.path.exists(SCANS_FILE):
        try:
            with open(SCANS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading initial scans from file: {e}")
    return []

# Persistent in-memory and local file store (strictly initialized without mock offsets)
mock_db = {
    "users": _load_initial_users(),
    "scans": _load_initial_scans(),
    "logs": [
        { "time": datetime.utcnow().isoformat(), "type": "info", "text": "Database initialized successfully." }
    ],
    "kb": DEFAULT_KB,
    "metrics": {
        "total_scans": 0,
        "successful_treatments": 0,
        "stain_counts": {"Used Cooking Oil": 0, "Black Ballpen Ink": 0, "Mud": 0}
    }
}

def _save_scans_file():
    try:
        with open(SCANS_FILE, "w", encoding="utf-8") as f:
            json.dump(mock_db["scans"], f, indent=2)
    except Exception as e:
        print(f"Error saving scans file: {e}")

def create_mongo_client(uri):
    """
    Creates a robust PyMongo MongoClient instance optimized for Atlas cloud clusters
    and graceful mobile network reconnection (socket timeouts, idle connection closing, TLS).
    """
    formatted_uri = uri
    if "tlsAllowInvalidCertificates" not in formatted_uri:
        separator = "&" if "?" in formatted_uri else "?"
        formatted_uri = f"{formatted_uri}{separator}tlsAllowInvalidCertificates=true"

    try:
        ca_cert = certifi.where()
    except Exception:
        ca_cert = None

    client_options = {
        "serverSelectionTimeoutMS": 5000,
        "connectTimeoutMS": 10000,
        "socketTimeoutMS": 20000,
        "maxIdleTimeMS": 45000,          # Close idle sockets before Atlas drops them
        "retryWrites": True,             # Retry writes on transient network glitches
        "retryReads": True,              # Retry reads on transient network drops
        "tls": True,
        "tlsAllowInvalidCertificates": True
    }
    if ca_cert:
        client_options["tlsCAFile"] = ca_cert

    return MongoClient(formatted_uri, **client_options)

def init_mongodb():
    """
    Establishes connection to MongoDB Atlas, verifies ping, and initializes
    default knowledge base and admin collections.
    """
    global client, db, is_demo_mode, _last_connect_attempt
    import time
    _last_connect_attempt = time.time()

    if not MONGO_URI:
        print("MONGO_URI not configured. Running in local resilient fallback mode.")
        is_demo_mode = True
        db = None
        return None

    try:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

        new_client = create_mongo_client(MONGO_URI)
        # Verify cluster accessibility via ping
        new_client.admin.command('ping')
        client = new_client
        db = client[MONGO_DB_NAME]
        is_demo_mode = False
        print(f"Connected to MongoDB Atlas successfully (DB: {MONGO_DB_NAME}).")

        # Populate Default Knowledge Base if collection is empty
        try:
            if db["kb"].count_documents({}) == 0:
                db["kb"].insert_one({"_id": "current_kb", "data": DEFAULT_KB})
                print("Populated default knowledge base in MongoDB.")
        except Exception as e:
            print(f"Notice: Initial KB seeding skipped: {e}")

        # Seed initial admin user if collection is completely empty
        try:
            if db["users"].count_documents({}) == 0:
                default_users = [
                    { "name": "John Michael Dohinog", "email": "admin@stainscan.com", "password": "admin123", "role": "admin", "avatar": "System Root" }
                ]
                db["users"].insert_many(default_users)
                print("Populated root admin user in MongoDB.")
        except Exception as e:
            print(f"Notice: Initial users seeding skipped: {e}")

        return db
    except Exception as e:
        print(f"Notice: MongoDB Atlas connection error ({e}). Running in resilient fallback mode.")
        db = None
        is_demo_mode = True
        return None

def get_db():
    """
    Returns the active database instance. If a temporary network disconnect
    occurred, attempts a graceful reconnect if cooldown has elapsed.
    """
    global client, db, is_demo_mode, _last_connect_attempt
    import time

    if not MONGO_URI:
        return None

    if db is not None:
        return db

    # Throttle reconnect attempts to once every 10 seconds to avoid blocking requests
    if time.time() - _last_connect_attempt > 10:
        return init_mongodb()

    return None

def execute_db_operation(op, fallback_fn=None):
    """
    Executes a database operation with automatic single retry on transient
    network drops (AutoReconnect, ConnectionFailure, ServerSelectionTimeoutError).
    If connection is unavailable or fails, invokes fallback_fn gracefully.
    """
    active_db = get_db()
    if active_db is not None:
        try:
            return op(active_db)
        except (AutoReconnect, ConnectionFailure, ServerSelectionTimeoutError, NetworkTimeout) as net_err:
            print(f"Transient MongoDB network drop detected ({net_err}). Attempting reconnect & retry...")
            reconnected_db = init_mongodb()
            if reconnected_db is not None:
                try:
                    return op(reconnected_db)
                except Exception as retry_err:
                    print(f"MongoDB retry failed after network drop: {retry_err}. Falling back gracefully.")
            else:
                print("MongoDB reconnection failed. Falling back gracefully.")
        except Exception as general_err:
            print(f"MongoDB operation error: {general_err}. Falling back gracefully.")

    if fallback_fn is not None:
        return fallback_fn()
    return None

# Initial connection attempt on boot
init_mongodb()

def get_metrics():
    all_scans = get_scans()
    mobile_users = get_mobile_user_count()

    # 1. If scans collection is empty or dropped, return clean 0 values
    if not all_scans:
        clean_zero_metrics = {
            "total_scans": 0,
            "total_cnn_scans_run": 0,
            "totalScansCount": 0,
            "successful_treatments": 0,
            "totalTreatedCount": 0,
            "total_users": mobile_users,
            "registered_mobile_users": mobile_users,
            "totalUsersCount": mobile_users,
            "stain_counts": {"Used Cooking Oil": 0, "Black Ballpen Ink": 0, "Mud": 0}
        }
        save_metrics(clean_zero_metrics)
        return clean_zero_metrics

    # 2. Scans exist: compute real metrics directly from the scans and history collections
    stain_counts = {"Used Cooking Oil": 0, "Black Ballpen Ink": 0, "Mud": 0}
    treated_count = 0
    for s in all_scans:
        st = s.get("stain")
        if st in stain_counts:
            stain_counts[st] += 1
        if s.get("status") == "Treated":
            treated_count += 1

    metrics = {
        "total_scans": len(all_scans),
        "total_cnn_scans_run": len(all_scans),
        "totalScansCount": len(all_scans),
        "successful_treatments": treated_count,
        "totalTreatedCount": treated_count,
        "total_users": mobile_users,
        "registered_mobile_users": mobile_users,
        "totalUsersCount": mobile_users,
        "stain_counts": stain_counts,
        "status": "ok"
    }
    save_metrics(metrics)
    return metrics

def save_metrics(metrics):
    def _op(d):
        d["metrics"].replace_one({"_id": "app_metrics"}, {"_id": "app_metrics", **metrics}, upsert=True)
        return True
    execute_db_operation(_op, fallback_fn=lambda: None)

    try:
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Error saving metrics to file: {e}")

    if "metrics" in mock_db:
        mock_db["metrics"] = metrics

def increment_scan_count(stain_name=None, fabric_name=None, confidence=None, email=None, image=None):
    metrics = get_metrics()
    metrics["total_scans"] = metrics.get("total_scans", 0) + 1

    conf_num = int(confidence or 0)
    # Strictly enforce 70% confidence threshold: override stain to Unidentified and fabric to Unidentified / N/A if below 70%
    if conf_num < 70:
        stain_name = "Unidentified"
        fabric_name = "Unidentified / N/A"

    if stain_name and stain_name != "Unidentified":
        stain_counts = metrics.setdefault("stain_counts", {})
        stain_counts[stain_name] = stain_counts.get(stain_name, 0) + 1

    save_metrics(metrics)

    # Also log a persistent scan record in the scans table
    scan_id = "h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + str(metrics["total_scans"])
    new_scan = {
        "id": scan_id,
        "email": email or "mobile_user@stainscan.com",
        "stain": stain_name or "Unidentified",
        "fabric": fabric_name or ("Unidentified / N/A" if conf_num < 70 else "Cotton Fabric (100%)"),
        "confidence": conf_num,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "Pending",
        "image": image or ""
    }
    try:
        add_scan(new_scan)
        if conf_num < 70:
            add_log("warning", f"CNN Scan #{metrics['total_scans']} below 70% threshold: Unidentified ({conf_num}%)")
        else:
            add_log("info", f"CNN Scan #{metrics['total_scans']} classified: {stain_name} ({conf_num}%)")
    except Exception as e:
        print(f"Error logging scan record: {e}")

    return metrics, scan_id

def increment_treatment_count(scan_id=None):
    # Guard against double counting: check if scan is already marked Treated
    if scan_id:
        def _op(d):
            from bson.objectid import ObjectId
            query = {"$or": [{"_id": ObjectId(scan_id)}, {"id": scan_id}]} if ObjectId.is_valid(scan_id) else {"id": scan_id}
            existing_scan = d["scans"].find_one(query)
            if existing_scan:
                if existing_scan.get("status") == "Treated":
                    return "already_treated"
                d["scans"].update_one(query, {"$set": {"status": "Treated"}})
                return "updated"
            return None
        def _fb():
            s = next((x for x in mock_db["scans"] if x.get("id") == scan_id or x.get("_id") == scan_id), None)
            if s:
                if s.get("status") == "Treated":
                    return "already_treated"
                s["status"] = "Treated"
                _save_scans_file()
                return "updated"
            return None

        status = execute_db_operation(_op, fallback_fn=_fb)
        if status == "already_treated":
            print(f"Scan {scan_id} is already marked Treated. Skipping duplicate increment.")
            return get_metrics()

    metrics = get_metrics()
    metrics["successful_treatments"] = metrics.get("successful_treatments", 0) + 1
    save_metrics(metrics)
    add_log("info", f"Treatment #{metrics['successful_treatments']} completed successfully (Scan ID: {scan_id or 'N/A'})")
    return metrics

def get_users():
    def _op(d):
        users = list(d["users"].find({}))
        for u in users:
            u["_id"] = str(u["_id"])
        return users
    return execute_db_operation(_op, fallback_fn=lambda: mock_db["users"])

def get_user_count():
    """
    Counts total user documents in MongoDB 'users' collection.
    """
    def _op(d):
        return d["users"].count_documents({})
    return execute_db_operation(_op, fallback_fn=lambda: len(get_users()))

def get_mobile_user_count():
    """
    Strictly counts actual mobile user accounts where role == 'user' (or role != 'admin')
    in MongoDB without adding any hardcoded offsets or mock users.
    """
    def _op(d):
        return d["users"].count_documents({"$or": [{"role": "user"}, {"role": {"$ne": "admin"}}]})
    def _fb():
        return len([u for u in get_users() if u.get("role") != "admin"])
    return execute_db_operation(_op, fallback_fn=_fb)

def add_user(user):
    def _op(d):
        result = d["users"].insert_one(user)
        user["_id"] = str(result.inserted_id)
        return user
    def _fb():
        mock_db["users"].append(user)
        _save_users_file()
        return user
    res = execute_db_operation(_op, fallback_fn=_fb)
    if user not in mock_db["users"]:
        mock_db["users"].append(user)
        _save_users_file()
    return res

def find_user_by_email(email):
    def _op(d):
        user = d["users"].find_one({"email": email})
        if user:
            user["_id"] = str(user["_id"])
        return user
    def _fb():
        return next((u for u in mock_db["users"] if u["email"] == email), None)
    return execute_db_operation(_op, fallback_fn=_fb)

def get_kb():
    def _op(d):
        doc = d["kb"].find_one({"_id": "current_kb"})
        return doc["data"] if doc else DEFAULT_KB
    return execute_db_operation(_op, fallback_fn=lambda: mock_db.get("kb", DEFAULT_KB))

def save_kb(kb_data):
    def _op(d):
        d["kb"].replace_one({"_id": "current_kb"}, {"_id": "current_kb", "data": kb_data}, upsert=True)
        return True
    execute_db_operation(_op, fallback_fn=lambda: None)
    mock_db["kb"] = kb_data

def get_scans(email=None):
    def _op(d):
        query = {"email": email} if email else {}
        scans = list(d["scans"].find(query).sort("timestamp", -1))
        # Ensure completed scans from 'history' collection are also merged if present
        try:
            colls = d.list_collection_names()
            if "history" in colls:
                history_scans = list(d["history"].find(query).sort("timestamp", -1))
                existing_ids = {str(s.get("id") or s.get("_id")) for s in scans}
                for hs in history_scans:
                    hs_id = str(hs.get("id") or hs.get("_id"))
                    if hs_id not in existing_ids:
                        scans.append(hs)
                        existing_ids.add(hs_id)
        except Exception as e:
            print(f"Notice: checking history collection: {e}")
        for s in scans:
            s["_id"] = str(s["_id"])
        return scans
    def _fb():
        if email:
            return [s for s in mock_db["scans"] if s["email"] == email]
        return sorted(mock_db["scans"], key=lambda s: s.get("timestamp", ""), reverse=True)
    return execute_db_operation(_op, fallback_fn=_fb)

def add_scan(scan):
    # Enforce 75% confidence persistence check
    conf_val = scan.get("confidence")
    if conf_val is not None:
        try:
            if int(conf_val) < 75 or scan.get("stain") in ["Unidentified", "Unidentified Stain", "Unknown"]:
                scan["stain"] = "Unidentified"
                scan["fabric"] = "Unidentified / N/A"
        except (ValueError, TypeError):
            pass

    def _op(d):
        result = d["scans"].insert_one(scan)
        scan["_id"] = str(result.inserted_id)
        return scan
    def _fb():
        if not scan.get("id"):
            scan["id"] = "h_" + str(len(mock_db["scans"]) + 1)
        mock_db["scans"].insert(0, scan)
        _save_scans_file()
        return scan
    res = execute_db_operation(_op, fallback_fn=_fb)
    # Ensure local in-memory fallback list stays synced
    if scan not in mock_db["scans"]:
        mock_db["scans"].insert(0, scan)
        _save_scans_file()
    return res

def get_logs():
    def _op(d):
        logs = list(d["logs"].find({}).sort("time", -1).limit(100))
        for l in logs:
            l["_id"] = str(l["_id"])
        return logs
    return execute_db_operation(_op, fallback_fn=lambda: mock_db["logs"][:100])

def add_log(log_type, text):
    log_entry = {
        "time": datetime.utcnow().isoformat(),
        "type": log_type,
        "text": text
    }
    def _op(d):
        d["logs"].insert_one(log_entry)
        return True
    execute_db_operation(_op, fallback_fn=lambda: None)
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
            
        conf_num = int(data["confidence"])
        stain_name = "Unidentified" if conf_num < 70 else data["stain"]
        fabric_name = "Unidentified / N/A" if conf_num < 70 else data["fabric"]
        new_scan = {
            "id": data.get("id", ""),
            "email": data["email"].strip().lower(),
            "stain": stain_name,
            "fabric": fabric_name,
            "confidence": conf_num,
            "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
            "status": data.get("status", "Pending"),
            "image": data.get("image", "")
        }
        
        # Check if scan with this id already exists to prevent duplicate insertion
        scan_id = data.get("id", "")
        if scan_id:
            def _mongo_check(d):
                from bson.objectid import ObjectId
                query = {"$or": [{"id": scan_id}, {"_id": ObjectId(scan_id)}]} if ObjectId.is_valid(scan_id) else {"id": scan_id}
                ex = d["scans"].find_one(query)
                if ex:
                    ex["_id"] = str(ex["_id"])
                    return ex
                return None
            def _fb_check():
                return next((s for s in mock_db["scans"] if s.get("id") == scan_id), None)

            existing = execute_db_operation(_mongo_check, fallback_fn=_fb_check)
            if existing:
                return jsonify(existing), 200

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
    
    def _op(d):
        from bson.objectid import ObjectId
        query = {"$or": [{"_id": ObjectId(scan_id)}, {"id": scan_id}]} if ObjectId.is_valid(scan_id) else {"$or": [{"id": scan_id}, {"_id": scan_id}]}
        res = d["scans"].update_one(query, {"$set": {"status": new_status}})
        return res.matched_count > 0

    def _fb():
        scan = next((s for s in mock_db["scans"] if s.get("id") == scan_id or s.get("_id") == scan_id), None)
        if scan:
            scan["status"] = new_status
            _save_scans_file()
            return True
        return False

    updated = execute_db_operation(_op, fallback_fn=_fb)
    if updated:
        if new_status == "Treated":
            increment_treatment_count(scan_id=scan_id)
        add_log("info", f"Scan {scan_id} status updated to {new_status}")
        return jsonify({"success": True, "message": f"Scan updated to {new_status}"})
            
    return jsonify({"error": "Scan not found"}), 404

@app.route("/api/scans/save", methods=["POST", "OPTIONS"])
@app.route("/api/treatment-complete", methods=["POST", "OPTIONS"])
@app.route("/api/treatment/complete", methods=["POST", "OPTIONS"])
def api_save_treatment_scan():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    data = request.get_json(silent=True) or {}
    scan_id = data.get("scan_id") or data.get("id") or ("h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    confidence = int(data.get("confidence") or 90)
    stain = "Unidentified" if confidence < 70 else (data.get("stain") or "Used Cooking Oil")
    fabric = "Unidentified / N/A" if confidence < 70 else (data.get("fabric") or data.get("fabric_color") or "Cotton Fabric (100%)")
    status = data.get("status") or data.get("treatment_status") or "Treated"
    timestamp = data.get("timestamp") or datetime.utcnow().isoformat()
    image = data.get("image") or ""
    email = (data.get("user_email") or data.get("email") or "mobile_user@stainscan.com").strip().lower()

    scan_doc = {
        "id": scan_id,
        "email": email,
        "stain": stain,
        "fabric": fabric,
        "confidence": confidence,
        "status": status,
        "timestamp": timestamp,
        "image": image
    }

    def _op(d):
        from bson.objectid import ObjectId
        query = {"$or": [{"_id": ObjectId(scan_id)}, {"id": scan_id}]} if ObjectId.is_valid(scan_id) else {"id": scan_id}
        existing = d["scans"].find_one(query)
        if existing:
            d["scans"].update_one(query, {"$set": {
                "status": status,
                "stain": stain,
                "fabric": fabric,
                "confidence": confidence,
                "timestamp": timestamp,
                "image": image or existing.get("image", "")
            }})
            scan_doc["_id"] = str(existing["_id"])
        else:
            res = d["scans"].insert_one(scan_doc)
            scan_doc["_id"] = str(res.inserted_id)

        # Also sync to 'history' collection if present
        try:
            if "history" in d.list_collection_names():
                d["history"].update_one(query, {"$set": scan_doc}, upsert=True)
        except Exception as e:
            print(f"Notice: history collection sync: {e}")

        return scan_doc

    def _fb():
        existing = next((s for s in mock_db["scans"] if s.get("id") == scan_id or s.get("_id") == scan_id), None)
        if existing:
            existing["status"] = status
            existing["stain"] = stain
            existing["fabric"] = fabric
            existing["confidence"] = confidence
            existing["timestamp"] = timestamp
            if image:
                existing["image"] = image
        else:
            mock_db["scans"].insert(0, scan_doc)
        _save_scans_file()
        return scan_doc

    saved = execute_db_operation(_op, fallback_fn=_fb)

    # Keep mock_db list in sync
    existing_mock = next((s for s in mock_db["scans"] if s.get("id") == scan_id or s.get("_id") == scan_id), None)
    if existing_mock:
        existing_mock["status"] = status
        existing_mock["stain"] = stain
        existing_mock["fabric"] = fabric
        existing_mock["confidence"] = confidence
        existing_mock["timestamp"] = timestamp
        if image:
            existing_mock["image"] = image
    else:
        mock_db["scans"].insert(0, scan_doc)
    _save_scans_file()

    add_log("info", f"Scan {scan_id} saved and treatment completed: {email} ({stain})")
    current_metrics = get_metrics()

    return jsonify({
        "success": True,
        "message": f"Scan {scan_id} saved and treatment marked complete",
        "scan": saved or scan_doc,
        "successful_treatments": current_metrics.get("successful_treatments", 0),
        "metrics": current_metrics
    }), 200

@app.route("/api/stats", methods=["GET", "OPTIONS"])
@app.route("/api/admin/overview", methods=["GET", "OPTIONS"])
@app.route("/api/metrics", methods=["GET", "OPTIONS"])
def api_admin_stats():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    metrics = get_metrics()
    return jsonify(metrics), 200

@app.route("/api/users/count", methods=["GET", "OPTIONS"])
def api_users_count():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    mobile_count = get_mobile_user_count()
    total_count = execute_db_operation(lambda d: d["users"].count_documents({}), fallback_fn=lambda: len(get_users()))
    return jsonify({
        "count": mobile_count,
        "total_users": mobile_count,
        "registered_mobile_users": mobile_count,
        "totalUsersCount": mobile_count,
        "total_accounts": total_count
    }), 200

@app.route("/api/metrics/reset", methods=["POST", "OPTIONS"])
def api_metrics_reset():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    clean_zero_metrics = {
        "total_scans": 0,
        "successful_treatments": 0,
        "stain_counts": {"Used Cooking Oil": 0, "Black Ballpen Ink": 0, "Mud": 0}
    }
    save_metrics(clean_zero_metrics)
    add_log("info", "Metrics counters reset to 0 by Administrator.")
    return jsonify({"success": True, "message": "Metrics reset to 0", "metrics": clean_zero_metrics}), 200

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
      1. Correct EXIF orientation (for mobile camera/gallery photos)
      2. Convert to RGB color space
      3. Resize to target input shape (224x224) using Bilinear interpolation
      4. Apply CIELAB CLAHE dynamic contrast equalization
      5. Apply MobileNetV2 preprocess_input to scale pixel values to [-1.0, 1.0]
      6. Expand batch dimension -> shape (1, 224, 224, 3)
    """
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass

    target_size = IMAGE_SIZE
    if model is not None and hasattr(model, "input_shape") and model.input_shape:
        shape = model.input_shape
        if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
            target_size = (int(shape[1]), int(shape[2]))

    img = pil_image.convert("RGB").resize(target_size, Image.BILINEAR)
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
        try:
            import keras.src.initializers.random_initializers as kri
            _patch_init(kri.GlorotUniform)
            _patch_init(kri.GlorotNormal)
        except Exception:
            pass

        def _patch_bn(cls):
            orig_init = cls.__init__
            def patched_bn_init(self, *args, **kwargs):
                kwargs.pop("renorm", None)
                kwargs.pop("renorm_clipping", None)
                kwargs.pop("renorm_momentum", None)
                orig_init(self, *args, **kwargs)
            cls.__init__ = patched_bn_init

        _patch_bn(keras.layers.BatchNormalization)
        try:
            import keras.src.layers.normalization.batch_normalization as kbn
            _patch_bn(kbn.BatchNormalization)
        except Exception:
            pass

        def _patch_input(cls):
            orig_init = cls.__init__
            def patched_input_init(self, *args, **kwargs):
                kwargs.pop("optional", None)
                orig_init(self, *args, **kwargs)
            cls.__init__ = patched_input_init

        _patch_input(keras.layers.InputLayer)
        try:
            import keras.src.layers.core.input_layer as kil
            _patch_input(kil.InputLayer)
        except Exception:
            pass
        def _patch_dense(cls):
            orig_init = cls.__init__
            def patched_dense_init(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                orig_init(self, *args, **kwargs)
            cls.__init__ = patched_dense_init

        _patch_dense(keras.layers.Dense)
        try:
            import keras.src.layers.core.dense as kd
            _patch_dense(kd.Dense)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning setting compatibility patches: {e}")
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
        raw_image = Image.open(BytesIO(image_bytes))
        try:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
        except Exception:
            image = raw_image.convert("RGB")

        # Ensure model is ready; never return fake mock predictions
        if model is None:
            load_model()
            if model is None:
                err_detail = LAST_MODEL_ERROR or "CNN model weights file not loaded on server."
                print(f"Prediction requested but model is unavailable: {err_detail}")
                return jsonify({
                    "error": "CNN model unavailable on server. Please ensure the model file is deployed.",
                    "status": "error",
                    "details": err_detail
                }), 503

        # 1. Image Quality & Out-of-Distribution Plausibility Checks
        # Prevent CLAHE contrast equalization from inflating flat/uniform sensor noise into artificial "Mud"
        arr_raw = np.array(image.resize((224, 224), Image.BILINEAR), dtype=np.uint8)
        gray_raw = arr_raw.mean(axis=2)
        contrast_std = float(np.std(gray_raw))

        # Check HSV color saturation to reject vivid non-fabric objects (e.g., Persian rugs, neon objects)
        try:
            import cv2
            hsv = cv2.cvtColor(arr_raw, cv2.COLOR_RGB2HSV)
            mean_sat = float(np.mean(hsv[:, :, 1]))
        except Exception:
            mean_sat = 0.0

        is_flat_surface = (contrast_std < 4.0)
        is_vivid_non_fabric = (mean_sat > 75.0)

        # 2. Preprocess uploaded camera/gallery image and run real TensorFlow CNN inference
        input_arr   = preprocess_image(image)              # (1, 224, 224, 3)
        predictions = model.predict(input_arr, verbose=0)  # (1, 3) softmax probabilities
        probs       = predictions[0]                        # (3,)
        
        # Sort predictions to evaluate top-1 vs top-2 class probabilities
        order              = np.argsort(probs)[::-1]
        top1_idx           = int(order[0])
        top2_idx           = int(order[1])
        max_confidence     = float(probs[top1_idx])
        top2_confidence    = float(probs[top2_idx])
        margin             = max_confidence - top2_confidence
        confidence_percent = int(round(max_confidence * 100))
        margin_percent     = round(margin * 100, 2)
        predicted_raw      = CLASS_MAPPING_RAW.get(top1_idx, "Unknown")
        predicted_label    = CLASS_MAPPING_DISPLAY.get(top1_idx, predicted_raw)

        # Build authentic probability distribution across all 3 trained classes
        probabilities = {
            "Black Ballpen Ink": round(float(probs[0]) * 100, 2),
            "Used Cooking Oil":  round(float(probs[1]) * 100, 2),
            "Mud":               round(float(probs[2]) * 100, 2)
        }

        # Retrieve dynamic recommendation based on stain and fabric context
        rec = get_recommendation(predicted_label, fabric_context, image=image)
        fabric_color = rec.get("fabric_color", "White Cotton")

        # ROBUST CONFIDENCE THRESHOLD & MARGIN / OUT-OF-DISTRIBUTION VALIDATION CHECK
        CONFIDENCE_THRESHOLD = 0.75  # 75% safe confidence threshold
        MARGIN_THRESHOLD     = 0.25  # 25% minimum separation between top-1 and top-2 classes

        is_low_confidence = (max_confidence < CONFIDENCE_THRESHOLD or confidence_percent < 75)
        is_narrow_margin  = (margin < MARGIN_THRESHOLD)
        is_unidentified   = is_low_confidence or is_narrow_margin or is_flat_surface or is_vivid_non_fabric

        if is_unidentified:
            unidentified_title = "Unidentified Stain"
            if is_flat_surface:
                caution_message = "No visible stain pattern detected. Garment surface appears clean or uniform."
            elif is_vivid_non_fabric:
                caution_message = "Pattern or color distribution not recognized as white or grey cotton fabric."
            elif is_narrow_margin:
                caution_message = f"Ambiguous Scan. Model uncertainty between stain types (narrow separation of {round(margin*100, 1)}%). Please capture a clearer, centered photo."
            else:
                caution_message = f"Inconclusive Scan (<75%). Prediction confidence ({confidence_percent}%) is below the safety threshold. Please scan a clear, well-lit photo of a stained cotton garment."

            metrics, scan_id = increment_scan_count(
                stain_name="Unidentified",
                fabric_name="Unidentified / N/A",
                confidence=confidence_percent,
                email=data.get("email"),
                image=data.get("image", "")
            )

            print(f"Prediction overridden to Unidentified: conf={confidence_percent}%, margin={margin_percent}%, flat={is_flat_surface}, vivid={is_vivid_non_fabric} | Probs: {probabilities}")

            return jsonify({
                "id":                   scan_id,
                "status":               "Unidentified",
                "is_classified":        False,
                "stain_classification": "Unidentified",
                "fabric_material":      "Unidentified / N/A",
                "fabric_color":         "N/A",
                "fabric_type":          "Unidentified / N/A",
                "confidence":           confidence_percent,
                "margin":               margin_percent,
                "probabilities":        probabilities,
                "stain":                unidentified_title,
                "fabric":               "Unidentified / N/A",
                "steps":                [],
                "caution":              caution_message,
                "materials":            [],
                "recommendation": {
                    "id":            scan_id,
                    "status":        "inconclusive",
                    "is_classified": False,
                    "steps":         [],
                    "caution":       caution_message,
                    "materials":     [],
                    "fabric_color":  "N/A",
                    "fabric_type":   "Unidentified / N/A"
                }
            })

        # Confident classification (>= 75% confidence, >= 25% margin, genuine stained fabric)
        metrics, scan_id = increment_scan_count(
            stain_name=predicted_label,
            fabric_name=f"{fabric_color} (100%)",
            confidence=confidence_percent,
            email=data.get("email"),
            image=data.get("image", "")
        )

        print(f"Prediction successful (>=75%, margin={margin_percent}%): {predicted_raw} / {predicted_label} ({confidence_percent}% confidence) | "
              f"Fabric: {fabric_color} | Probs: {probabilities}")

        # Return authentic CNN model prediction with probability distribution, fabric-specific steps and caution
        return jsonify({
            "id":                   scan_id,
            "status":               "success",
            "is_classified":        True,
            "stain_classification": predicted_raw,
            "fabric_material":      "Cotton",
            "fabric_color":         fabric_color,
            "confidence":           confidence_percent,
            "margin":               margin_percent,
            "probabilities":        probabilities,
            "stain":                predicted_label,
            "fabric":               f"{fabric_color} (100%)",
            "steps":                rec["steps"],
            "caution":              rec["caution"],
            "materials":            rec["materials"],
            "recommendation": {
                "id":           scan_id,
                "status":       "success",
                "is_classified": True,
                "steps":        rec["steps"],
                "caution":      rec["caution"],
                "materials":    rec["materials"],
                "fabric_color": fabric_color
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
