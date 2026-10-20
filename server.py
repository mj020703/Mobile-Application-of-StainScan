import sys
import functools
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# Ensure immediate flush for real-time terminal and log monitoring
print = functools.partial(print, flush=True)
import base64
from io import BytesIO
import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"
from datetime import datetime
import traceback
from PIL import Image, ImageOps, UnidentifiedImageError
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
import cv2

IMAGE_SIZE           = (224, 224)                 # Exact match with MobileNetV2 input shape
MODEL_PATH           = "stainscan_model_v3_balanced.keras"  # Primary retrained model file
CONFIDENCE_THRESHOLD = 0.85                       # Strict 85% confidence threshold for Out-of-Distribution guardrail
LOADED_MODEL_FILE    = None
LAST_MODEL_ERROR     = None

# Initialize Haar cascade face detector if available
FACE_CASCADE = None
try:
    cascade_filename = "haarcascade_frontalface_default.xml"
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        getattr(getattr(cv2, "data", None), "haarcascades", ""),
        "."
    ]
    if hasattr(cv2, "CascadeClassifier"):
        for s_dir in search_dirs:
            if s_dir:
                candidate = os.path.join(s_dir, cascade_filename)
                if os.path.exists(candidate):
                    clf = cv2.CascadeClassifier(candidate)
                    if not clf.empty():
                        FACE_CASCADE = clf
                        print(f"Haar cascade face detector loaded successfully from '{candidate}'.")
                        break
        if FACE_CASCADE is None:
            clf = cv2.CascadeClassifier(cascade_filename)
            if not clf.empty():
                FACE_CASCADE = clf
except Exception as face_init_err:
    print(f"Warning: Haar cascade face detector initialization: {face_init_err}")
    FACE_CASCADE = None

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
def handle_preflight_and_log():
    # Handle CORS preflight requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
        return response

    path = request.path
    # Ignore noisy static asset requests (css, js, images, favicon)
    if any(path.endswith(ext) for ext in [".css", ".js", ".png", ".jpg", ".ico", ".svg", ".woff", ".woff2"]) and not path.startswith("/api/"):
        return

    # Extract client IP and device information
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1")
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    user_agent = request.headers.get("User-Agent", "Unknown Device")
    origin = request.headers.get("Origin") or request.headers.get("Referer") or "Direct Request"

    # Log incoming mobile API endpoints with high visibility
    if path.startswith("/api/") or path == "/predict" or path == "/health":
        payload_summary = ""
        if request.is_json:
            try:
                data = request.get_json(silent=True) or {}
                sanitized = {}
                for k, v in data.items():
                    if k in ["image", "photo", "file"] and isinstance(v, str) and len(v) > 80:
                        sanitized[k] = f"<Base64 Image Data: {len(v):,} characters>"
                    elif k in ["password"]:
                        sanitized[k] = "••••••••"
                    else:
                        sanitized[k] = v
                payload_summary = json.dumps(sanitized)
            except Exception:
                payload_summary = "<unparseable json payload>"
        elif request.args:
            payload_summary = f"Query: {dict(request.args)}"

        print(f"\n{'='*75}")
        print(f"[INCOMING MOBILE / API REQUEST] {request.method} {path}")
        print(f"   * Client IP : {client_ip}")
        print(f"   * Origin    : {origin}")
        print(f"   * User-Agent: {user_agent[:90]}")
        if payload_summary:
            print(f"   * Payload   : {payload_summary}")
        print(f"{'='*75}\n")

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

# Configure dnspython DNS resolver fallback for Windows local environments
try:
    import dns.resolver
    _default_resolver = dns.resolver.get_default_resolver()
    if _default_resolver:
        for _ns in ["8.8.8.8", "1.1.1.1"]:
            if _ns not in _default_resolver.nameservers:
                _default_resolver.nameservers.append(_ns)
except Exception:
    pass

# Load .env file automatically if present in workspace
def load_env_file():
    # 1. Attempt standard python-dotenv first if installed
    try:
        from dotenv import load_dotenv
        for candidate in [".env", ".env.local"]:
            p1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), candidate)
            if os.path.exists(p1):
                load_dotenv(p1, override=True)
            p2 = os.path.join(os.getcwd(), candidate)
            if os.path.exists(p2):
                load_dotenv(p2, override=True)
    except Exception:
        pass

    # 2. Resilient manual parser fallback in case dotenv is unavailable
    env_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"),
        os.path.join(os.getcwd(), ".env.local")
    ]
    loaded = False
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("export "):
                            line = line[7:].strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and (k not in os.environ or not os.environ[k]):
                                os.environ[k] = v
                print(f"Loaded environment variables from '{env_path}'.")
                loaded = True
                break
            except Exception as e:
                print(f"Notice: Failed to load .env file '{env_path}': {e}")
    if not loaded:
        print("Notice: No local .env file found. Checking system environment variables.")

load_env_file()

# Initialize MongoDB Connection & Resilient Database Handling
MONGO_URI = (
    os.environ.get("MONGO_URI") or 
    os.environ.get("MONGODB_URI") or 
    os.environ.get("DATABASE_URL") or 
    ""
).strip()
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "stainscan_db").strip()

client = None
db = None
is_demo_mode = True
_last_connect_attempt = 0
LAST_MONGO_ERROR = None

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
    and graceful network reconnection (socket timeouts, idle connection closing, TLS).
    """
    formatted_uri = uri.strip()
    # Check if TLS certificate bypass is needed
    if "tlsallowinvalidcertificates" not in formatted_uri.lower():
        separator = "&" if "?" in formatted_uri else "?"
        formatted_uri = f"{formatted_uri}{separator}tlsAllowInvalidCertificates=true"

    try:
        ca_cert = certifi.where()
    except Exception:
        ca_cert = None

    client_options = {
        "serverSelectionTimeoutMS": 6000,
        "connectTimeoutMS": 10000,
        "socketTimeoutMS": 20000,
        "maxIdleTimeMS": 45000,          # Close idle sockets before Atlas drops them
        "retryWrites": True,             # Retry writes on transient network glitches
        "retryReads": True,              # Retry reads on transient network drops
        "tls": True,
        "tlsAllowInvalidCertificates": True
    }
    if ca_cert and os.path.exists(ca_cert):
        client_options["tlsCAFile"] = ca_cert

    return MongoClient(formatted_uri, **client_options)

def init_mongodb():
    """
    Establishes connection to MongoDB Atlas, verifies ping, and initializes
    default knowledge base and admin collections.
    """
    global client, db, is_demo_mode, _last_connect_attempt, LAST_MONGO_ERROR
    global MONGO_URI, MONGO_DB_NAME
    import time
    _last_connect_attempt = time.time()

    # Re-evaluate MONGO_URI from environment variables in case .env was modified
    current_uri = (
        os.environ.get("MONGO_URI") or 
        os.environ.get("MONGODB_URI") or 
        os.environ.get("DATABASE_URL") or 
        MONGO_URI or 
        ""
    ).strip()

    if not current_uri:
        LAST_MONGO_ERROR = "MONGO_URI is not set. Add your MongoDB Atlas connection string to .env (see .env.example)."
        print("Notice: MONGO_URI not configured. Running in local resilient fallback mode (Local JSON).")
        is_demo_mode = True
        db = None
        return None

    MONGO_URI = current_uri
    if not MONGO_DB_NAME:
        MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "stainscan_db").strip() or "stainscan_db"

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
        LAST_MONGO_ERROR = None
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
        LAST_MONGO_ERROR = str(e)
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

    if db is not None:
        return db

    current_uri = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI") or os.environ.get("DATABASE_URL") or MONGO_URI or ""
    if not current_uri.strip():
        return None

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
    """
    Fetches live statistics and counters directly from MongoDB Atlas collections:
    1. Checks the 'metrics' collection for a persisted 'app_metrics' document.
    2. If absent or newly initialized, dynamically counts live documents across the
       'scans' and 'users' collections without loading full Base64 images into memory.
    3. Falls back to mock_db / metrics.json only when running in offline/local demo mode.
    """
    def _op(d):
        # 1. Check if metrics document exists in MongoDB Atlas 'metrics' collection
        doc = d["metrics"].find_one({"_id": "app_metrics"})
        if not doc:
            doc = d["metrics"].find_one({})

        # Count live mobile users from 'users' collection
        mobile_users = d["users"].count_documents({"$or": [{"role": "user"}, {"role": {"$ne": "admin"}}]})

        # Count live scans and treated scans directly from 'scans' collection
        total_scans = d["scans"].count_documents({})
        treated_count = d["scans"].count_documents({"status": "Treated"})
        oil_count = d["scans"].count_documents({"stain": "Used Cooking Oil"})
        ink_count = d["scans"].count_documents({"stain": "Black Ballpen Ink"})
        mud_count = d["scans"].count_documents({"stain": "Mud"})

        if doc:
            doc["_id"] = str(doc["_id"])
            doc["total_users"] = mobile_users
            doc["registered_mobile_users"] = mobile_users
            doc["totalUsersCount"] = mobile_users

            # Sync total scans (respecting any higher historical counter)
            max_scans = max(int(doc.get("total_scans", 0)), total_scans)
            doc["total_scans"] = max_scans
            doc["total_cnn_scans_run"] = max_scans
            doc["totalScansCount"] = max_scans

            # Sync treated scans directly with live database collection
            max_treated = max(int(doc.get("successful_treatments", 0)), treated_count)
            doc["successful_treatments"] = max_treated
            doc["totalTreatedCount"] = max_treated

            # Sync stain distribution
            stain_counts = doc.setdefault("stain_counts", {})
            stain_counts["Used Cooking Oil"] = max(stain_counts.get("Used Cooking Oil", 0), oil_count)
            stain_counts["Black Ballpen Ink"] = max(stain_counts.get("Black Ballpen Ink", 0), ink_count)
            stain_counts["Mud"] = max(stain_counts.get("Mud", 0), mud_count)

            doc["database_source"] = "mongodb_atlas"
            doc["is_live"] = True
            doc["status"] = "ok"

            # Persist updated synchronized numbers back to MongoDB Atlas 'metrics' collection
            d["metrics"].update_one(
                {"_id": "app_metrics"},
                {"$set": {
                    "total_users": mobile_users,
                    "registered_mobile_users": mobile_users,
                    "totalUsersCount": mobile_users,
                    "total_scans": max_scans,
                    "total_cnn_scans_run": max_scans,
                    "totalScansCount": max_scans,
                    "successful_treatments": max_treated,
                    "totalTreatedCount": max_treated,
                    "stain_counts": stain_counts,
                    "database_source": "mongodb_atlas",
                    "is_live": True,
                    "status": "ok"
                }},
                upsert=True
            )
            return doc

        # 2. If no metrics document exists yet, aggregate live counts directly from MongoDB collections
        computed = {
            "_id": "app_metrics",
            "total_scans": total_scans,
            "total_cnn_scans_run": total_scans,
            "totalScansCount": total_scans,
            "successful_treatments": treated_count,
            "totalTreatedCount": treated_count,
            "total_users": mobile_users,
            "registered_mobile_users": mobile_users,
            "totalUsersCount": mobile_users,
            "stain_counts": {
                "Used Cooking Oil": oil_count,
                "Black Ballpen Ink": ink_count,
                "Mud": mud_count
            },
            "database_source": "mongodb_atlas",
            "is_live": True,
            "status": "ok"
        }
        d["metrics"].replace_one({"_id": "app_metrics"}, computed, upsert=True)
        return computed

    def _fb():
        all_scans = mock_db["scans"]
        mobile_users = len([u for u in mock_db["users"] if u.get("role") != "admin"])
        treated = len([s for s in all_scans if s.get("status") == "Treated"])
        counts = {"Used Cooking Oil": 0, "Black Ballpen Ink": 0, "Mud": 0}
        for s in all_scans:
            st = s.get("stain")
            if st in counts:
                counts[st] += 1
        return {
            "total_scans": len(all_scans),
            "total_cnn_scans_run": len(all_scans),
            "totalScansCount": len(all_scans),
            "successful_treatments": treated,
            "totalTreatedCount": treated,
            "total_users": mobile_users,
            "registered_mobile_users": mobile_users,
            "totalUsersCount": mobile_users,
            "stain_counts": counts,
            "database_source": "local_fallback",
            "is_live": False,
            "status": "ok (fallback)"
        }

    return execute_db_operation(_op, fallback_fn=_fb)

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
    # Strictly enforce 85% confidence threshold: override stain to Unidentified Object and fabric to Unidentified / N/A if below 85%
    if conf_num < 85 or stain_name in ["Unidentified", "Unidentified Object", "Unidentified Stain"]:
        stain_name = "Unidentified Object"
        fabric_name = "Unidentified / N/A"

    if stain_name and stain_name not in ["Unidentified", "Unidentified Object", "Unidentified Stain"]:
        stain_counts = metrics.setdefault("stain_counts", {})
        stain_counts[stain_name] = stain_counts.get(stain_name, 0) + 1

    save_metrics(metrics)

    # Also log a persistent scan record in the scans table
    scan_id = "h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + str(metrics["total_scans"])
    new_scan = {
        "id": scan_id,
        "email": email or "mobile_user@stainscan.com",
        "stain": stain_name or "Unidentified Object",
        "stain_type": stain_name or "Unidentified Object",
        "is_valid_fabric": conf_num >= 85 and stain_name not in ["Unidentified", "Unidentified Object", "Unidentified Stain"],
        "fabric": fabric_name or ("Unidentified / N/A" if conf_num < 85 else "Cotton Fabric (100%)"),
        "confidence": conf_num,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "Pending",
        "image": image or ""
    }
    try:
        add_scan(new_scan)
        if conf_num < 85:
            add_log("warning", f"CNN Scan #{metrics['total_scans']} below 85% threshold: Unidentified Object ({conf_num}%)")
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
    clean_email = str(email or "").strip().lower()
    if not clean_email:
        return None

    def _op(d):
        import re
        user = d["users"].find_one({"email": clean_email})
        if not user:
            # Case-insensitive fallback query against MongoDB Atlas
            user = d["users"].find_one({"email": {"$regex": f"^{re.escape(clean_email)}$", "$options": "i"}})
        if user:
            user["_id"] = str(user["_id"])
        return user

    def _fb():
        return next((u for u in mock_db["users"] if str(u.get("email", "")).strip().lower() == clean_email), None)

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

def get_scans(email=None, limit=None, include_images=True):
    def _op(d):
        query = {"email": email} if email else {}
        projection = None if include_images else {"image": 0}
        cursor = d["scans"].find(query, projection).sort("timestamp", -1)
        if limit:
            cursor = cursor.limit(limit)
        scans = list(cursor)
        # Ensure completed scans from 'history' collection are also merged if present
        try:
            colls = d.list_collection_names()
            if "history" in colls:
                hist_cursor = d["history"].find(query, projection).sort("timestamp", -1)
                if limit:
                    hist_cursor = hist_cursor.limit(limit)
                history_scans = list(hist_cursor)
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
            items = [s for s in mock_db["scans"] if s["email"] == email]
        else:
            items = sorted(mock_db["scans"], key=lambda s: s.get("timestamp", ""), reverse=True)
        if not include_images:
            items = [{k: v for k, v in s.items() if k != "image"} for s in items]
        if limit:
            items = items[:limit]
        return items
    return execute_db_operation(_op, fallback_fn=_fb)

def add_scan(scan):
    # Enforce 85% confidence persistence check
    conf_val = scan.get("confidence")
    if conf_val is not None:
        try:
            if int(conf_val) < 85 or scan.get("stain") in ["Unidentified", "Unidentified Object", "Unidentified Stain", "Unknown"]:
                if scan.get("stain") not in ["Unidentified", "Unidentified Object", "Unidentified Stain"]:
                    scan["stain"] = "Unidentified Object"
                scan["stain_type"] = scan.get("stain_type") or "Unidentified Object"
                scan["is_valid_fabric"] = False
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
@app.route("/api/signup", methods=["POST", "OPTIONS"])
@app.route("/api/auth/register", methods=["POST", "OPTIONS"])
def api_register():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. JSON object expected."}), 400
        
        raw_name = data.get("name")
        raw_email = data.get("email")
        raw_password = data.get("password")
        
        if raw_name is None or raw_email is None or raw_password is None:
            return jsonify({"error": "Required registration parameters missing. Name, email, and password are required."}), 400
        
        name = str(raw_name).strip()
        email = str(raw_email).strip().lower()
        password = str(raw_password)
        avatar = str(data.get("avatar") or name)
        
        if not name:
            return jsonify({"error": "Full name cannot be blank."}), 400
        if not email:
            return jsonify({"error": "Email address cannot be blank."}), 400
        if "@" not in email or "." not in email:
            return jsonify({"error": "Please enter a valid email address."}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters."}), 400
            
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
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
        print(f"[OK] [API REGISTER] Successfully saved user to database: '{email}' | Name: '{name}' | IP: {client_ip}")
        
        user_name = saved_user.get("name", name) if isinstance(saved_user, dict) else name
        user_email = saved_user.get("email", email) if isinstance(saved_user, dict) else email
        user_role = saved_user.get("role", "user") if isinstance(saved_user, dict) else "user"
        user_avatar = saved_user.get("avatar", avatar) if isinstance(saved_user, dict) else avatar
        
        return jsonify({
            "name": user_name,
            "email": user_email,
            "role": user_role,
            "avatar": user_avatar
        }), 201

    except Exception as e:
        print(f"[ERROR] [API REGISTER] Error during registration: {e}")
        return jsonify({"error": "A server error occurred during registration. Please try again."}), 500

@app.route("/api/login", methods=["POST", "OPTIONS"])
@app.route("/api/signin", methods=["POST", "OPTIONS"])
@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def api_login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request format. JSON object expected."}), 400
        
        raw_email = data.get("email")
        raw_password = data.get("password")
        
        if raw_email is None or raw_password is None:
            return jsonify({"error": "Missing login credentials. Both email and password are required."}), 400
        
        email = str(raw_email).strip().lower()
        password = str(raw_password)
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
        
        if not email or not password:
            return jsonify({"error": "Email address and password cannot be blank."}), 400
            
        user = find_user_by_email(email)
        if user and str(user.get("password", "")) == password:
            add_log("info", f"User login successful: {email}")
            print(f"[OK] [API LOGIN SUCCESS] User authenticated: '{email}' | Role: {user.get('role', 'user')} | IP: {client_ip}")
            return jsonify({
                "name": user.get("name", "User"),
                "email": user.get("email", email),
                "role": user.get("role", "user"),
                "avatar": user.get("avatar") or user.get("name", "User")
            }), 200
        
        add_log("warning", f"Failed login attempt for: {email}")
        print(f"[WARN] [API LOGIN FAILED] Invalid credentials for: '{email}' | IP: {client_ip}")
        return jsonify({"error": "Invalid email credentials or password"}), 401

    except Exception as e:
        print(f"Error during login: {e}")
        return jsonify({"error": "A server error occurred during login. Please try again."}), 500

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
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
        print(f"[SCAN] [API SCANS POST] Mobile scan recorded in database: User='{new_scan['email']}' | Stain='{new_scan['stain']}' | Conf={new_scan['confidence']}% | Fabric='{new_scan['fabric']}' | IP={client_ip}")
        return jsonify(saved_scan), 201
        
    email = request.args.get("email")
    limit = request.args.get("limit", type=int)
    # Default include_images to True only when querying by specific user email; false for dashboard tables
    include_images = request.args.get("include_images", "true" if email else "false").lower() in ["1", "true", "yes"]
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
    results = get_scans(email=email, limit=limit, include_images=include_images)
    print(f"[LIST] [API SCANS GET] Returning {len(results)} scans for User='{email or 'all'}' | IP={client_ip}")
    return jsonify(results)

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
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
    print(f"[TREATMENT] [API TREATMENT COMPLETE] Treatment saved to database: scan_id='{scan_id}' | User='{email}' | Stain='{stain}' | IP={client_ip}")
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

@app.route("/api/db-status", methods=["GET", "OPTIONS"])
def api_db_status():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    global client, db, is_demo_mode, LAST_MONGO_ERROR, MONGO_URI, MONGO_DB_NAME
    active_db = get_db()
    is_connected = False
    details = {
        "status": "connected" if (active_db is not None and not is_demo_mode) else "fallback_mode",
        "is_demo_mode": is_demo_mode or (active_db is None),
        "database_name": MONGO_DB_NAME if (active_db is not None and not is_demo_mode) else "local_json_mock",
        "mongo_configured": bool(MONGO_URI or os.environ.get("MONGO_URI")),
        "last_error": LAST_MONGO_ERROR
    }

    if active_db is not None and client is not None:
        try:
            client.admin.command('ping')
            is_connected = True
            details["collections"] = {
                "users": active_db["users"].count_documents({}),
                "scans": active_db["scans"].count_documents({}),
                "metrics": active_db["metrics"].count_documents({})
            }
        except Exception as e:
            is_connected = False
            details["ping_error"] = str(e)
            details["last_error"] = str(e)

    details["is_connected"] = is_connected
    return jsonify(details), 200

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
    if pil_image is None:
        raise ValueError("Cannot preprocess a None image object.")

    # 1. Correct EXIF orientation
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception as exif_err:
        print(f"[Preprocessing] EXIF orientation warning: {exif_err}")

    # 2. Determine target input shape
    target_size = IMAGE_SIZE
    if model is not None and hasattr(model, "input_shape") and model.input_shape:
        try:
            shape = model.input_shape
            if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                target_size = (int(shape[1]), int(shape[2]))
        except Exception as shape_err:
            print(f"[Preprocessing] Model input_shape lookup warning: {shape_err}")
            target_size = IMAGE_SIZE

    # 3. Convert to RGB and resize
    try:
        img = pil_image.convert("RGB").resize(target_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
    except Exception as img_err:
        raise ValueError(f"Failed converting or resizing image to array: {img_err}")

    # 4. Apply CIELAB CLAHE contrast enhancement
    try:
        enhanced = apply_clahe(arr)
    except Exception as clahe_err:
        print(f"[Preprocessing] CLAHE failed ({clahe_err}), proceeding with raw array.")
        enhanced = arr

    # 5. MobileNetV2 normalization [-1.0, 1.0]
    try:
        if HAS_TF:
            arr_norm = preprocess_input(enhanced.astype(np.float32))
        else:
            arr_norm = (enhanced.astype(np.float32) / 127.5) - 1.0
    except Exception as norm_err:
        print(f"[Preprocessing] Normalization warning ({norm_err}), using direct scaling.")
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
    try:
        # 1. Parse and validate JSON request body
        try:
            data = request.get_json(silent=True)
        except Exception as json_err:
            print(f"[Predict] Malformed JSON error: {json_err}")
            return jsonify({
                "error": "Malformed JSON payload in request.",
                "status": "error",
                "details": str(json_err)
            }), 400

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid request payload. Expected a JSON object.",
                "status": "error"
            }), 400

        raw_b64 = data.get("image")
        if not raw_b64 or not isinstance(raw_b64, str) or not raw_b64.strip():
            return jsonify({
                "error": "No image data provided. Please provide a valid base64 image string in the 'image' field.",
                "status": "error"
            }), 400

        base64_str = raw_b64.strip()
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1].strip()

        if not base64_str:
            return jsonify({
                "error": "Empty base64 image payload.",
                "status": "error"
            }), 400

        # Check fabric color / type in request context
        fabric_context = str(data.get("fabric_color") or data.get("fabric_type") or data.get("fabric") or "").strip()
        user_email = data.get("email")

        # 2. Decode base64 image data
        try:
            padded_b64 = base64_str + '=' * (-len(base64_str) % 4)
            image_bytes = base64.b64decode(padded_b64)
        except Exception as b64_err:
            print(f"[Predict] Base64 decoding failed: {b64_err}")
            return jsonify({
                "error": "Invalid base64-encoded image string. Decoding failed.",
                "status": "error",
                "details": str(b64_err)
            }), 400

        if not image_bytes:
            return jsonify({
                "error": "Decoded image data is empty.",
                "status": "error"
            }), 400

        # 3. Open image with Pillow and handle EXIF orientation
        try:
            raw_image = Image.open(BytesIO(image_bytes))
            raw_image.load()  # Force load pixel data to detect truncation or corruption early
        except (UnidentifiedImageError, OSError, IOError, ValueError) as pil_open_err:
            print(f"[Predict] Image identification/decoding failed: {pil_open_err}")
            return jsonify({
                "error": "Unsupported or corrupted image file. Please upload a valid JPEG, PNG, or WEBP image.",
                "status": "error",
                "details": str(pil_open_err)
            }), 400
        except Exception as pil_err:
            print(f"[Predict] Unexpected image decoding error: {pil_err}")
            return jsonify({
                "error": "Failed to decode the provided image.",
                "status": "error",
                "details": str(pil_err)
            }), 400

        try:
            image = ImageOps.exif_transpose(raw_image)
        except Exception as exif_err:
            print(f"[Predict] EXIF orientation warning: {exif_err}")
            image = raw_image

        try:
            image = image.convert("RGB")
        except Exception as conv_err:
            print(f"[Predict] RGB conversion failed: {conv_err}")
            return jsonify({
                "error": "Failed to convert image to RGB color space.",
                "status": "error",
                "details": str(conv_err)
            }), 400

        # 4. Ensure model is ready; never return fake mock predictions
        if model is None:
            load_model()
            if model is None:
                err_detail = LAST_MODEL_ERROR or "CNN model weights file not loaded on server."
                print(f"[Predict] Prediction requested but model is unavailable: {err_detail}")
                return jsonify({
                    "error": "CNN model unavailable on server. Please ensure the model file is deployed.",
                    "status": "error",
                    "details": err_detail
                }), 503

        # Convert uploaded PIL image to OpenCV BGR format
        cv_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # =========================================================
        # LAYER 1: OOD PRE-CHECK (OpenCV Skin & Face Filter)
        # =========================================================
        # a) YCrCb Human Skin Filter
        # Check human skin pixel ratios using bounds: lower [0, 133, 77], upper [255, 173, 127]
        try:
            cv_ycrcb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2YCrCb)
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)
            raw_skin_mask = cv2.inRange(cv_ycrcb, lower_skin, upper_skin)

            # Exclude near-white / neutral fabric highlights (Y > 215, S < 35) to prevent
            # warm-lit white/grey cotton and golden cooking oil from being falsely flagged as skin
            hsv_img = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)
            y_chan = cv_ycrcb[:, :, 0]
            s_chan = hsv_img[:, :, 1]
            is_fabric_highlight = (y_chan > 215) & (s_chan < 35)
            skin_mask = (raw_skin_mask > 0) & (~is_fabric_highlight)
            skin_ratio = float(np.mean(skin_mask))
        except Exception as skin_err:
            print(f"[Predict] Skin filter warning: {skin_err}")
            skin_ratio = 0.0

        if skin_ratio > 0.30:
            rejection_msg = "Scan rejected: Human skin detected. Please scan a cotton fabric stain."
            scan_id = "h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
            try:
                metrics, scan_id = increment_scan_count(
                    stain_name="Unidentified Object",
                    fabric_name="Unidentified / N/A",
                    confidence=0,
                    email=user_email,
                    image=data.get("image", "")
                )
            except Exception as scan_err:
                print(f"[Predict] Warning: Failed to record skin-rejected scan: {scan_err}")

            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
            print(f"[PREDICT] [OOD SKIN REJECTED] Intercepted by skin filter: skin_ratio={round(skin_ratio*100, 1)}% > 30% | User='{user_email or 'anonymous'}' | IP={client_ip}")

            return jsonify({
                "id":                   scan_id,
                "status":               "success",
                "stain_type":           "Unidentified Object",
                "is_valid_fabric":      False,
                "is_classified":        False,
                "confidence":           0,
                "confidence_score":     0.0,
                "skin_ratio":           round(skin_ratio, 4),
                "margin":               0.0,
                "message":              rejection_msg,
                "rejection_message":    rejection_msg,
                "caution":              rejection_msg,
                "stain":                "Unidentified Object",
                "stain_classification": "Unidentified Object",
                "fabric_material":      "Unidentified / N/A",
                "fabric_color":         "N/A",
                "fabric_type":          "Unidentified / N/A",
                "fabric":               "Unidentified / N/A",
                "steps":                [],
                "materials":            [],
                "probabilities": {
                    "Black Ballpen Ink": 0.0,
                    "Used Cooking Oil":  0.0,
                    "Mud":               0.0
                },
                "recommendation": {
                    "id":               scan_id,
                    "status":           "inconclusive",
                    "is_classified":    False,
                    "is_valid_fabric":  False,
                    "stain_type":       "Unidentified Object",
                    "steps":            [],
                    "caution":          rejection_msg,
                    "materials":        [],
                    "fabric_color":     "N/A",
                    "fabric_type":      "Unidentified / N/A"
                }
            }), 200

        # b) Haar Cascade Face Filter
        face_detected = False
        if FACE_CASCADE is not None:
            try:
                cv_gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
                h, w = cv_gray.shape[:2]
                if max(h, w) > 800:
                    scale = 800.0 / max(h, w)
                    gray_small = cv2.resize(cv_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    gray_small = cv_gray
                faces = FACE_CASCADE.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    face_detected = True
            except Exception as face_err:
                print(f"[Predict] Face detection warning: {face_err}")

        if face_detected:
            rejection_msg = "Scan rejected: Face detected. Please scan a cotton fabric stain."
            scan_id = "h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
            try:
                metrics, scan_id = increment_scan_count(
                    stain_name="Unidentified Object",
                    fabric_name="Unidentified / N/A",
                    confidence=0,
                    email=user_email,
                    image=data.get("image", "")
                )
            except Exception as scan_err:
                print(f"[Predict] Warning: Failed to record face-rejected scan: {scan_err}")

            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
            print(f"[PREDICT] [OOD FACE REJECTED] Intercepted by face detector | User='{user_email or 'anonymous'}' | IP={client_ip}")

            return jsonify({
                "id":                   scan_id,
                "status":               "success",
                "stain_type":           "Unidentified Object",
                "is_valid_fabric":      False,
                "is_classified":        False,
                "confidence":           0,
                "confidence_score":     0.0,
                "margin":               0.0,
                "message":              rejection_msg,
                "rejection_message":    rejection_msg,
                "caution":              rejection_msg,
                "stain":                "Unidentified Object",
                "stain_classification": "Unidentified Object",
                "fabric_material":      "Unidentified / N/A",
                "fabric_color":         "N/A",
                "fabric_type":          "Unidentified / N/A",
                "fabric":               "Unidentified / N/A",
                "steps":                [],
                "materials":            [],
                "probabilities": {
                    "Black Ballpen Ink": 0.0,
                    "Used Cooking Oil":  0.0,
                    "Mud":               0.0
                },
                "recommendation": {
                    "id":               scan_id,
                    "status":           "inconclusive",
                    "is_classified":    False,
                    "is_valid_fabric":  False,
                    "stain_type":       "Unidentified Object",
                    "steps":            [],
                    "caution":          rejection_msg,
                    "materials":        [],
                    "fabric_color":     "N/A",
                    "fabric_type":      "Unidentified / N/A"
                }
            }), 200

        # c) Out-of-Distribution Pre-Check: Laplacian Texture Variance
        # Decode into grayscale to evaluate surface texture weave (threads vs smooth surfaces)
        TEXTURE_THRESHOLD = 15.0
        try:
            cv_gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
            gray_std = cv2.resize(cv_gray, (224, 224), interpolation=cv2.INTER_AREA)
            laplacian_std = float(cv2.Laplacian(gray_std, cv2.CV_64F).var())
            laplacian_raw = float(cv2.Laplacian(cv_gray, cv2.CV_64F).var())
            texture_variance = max(laplacian_std, laplacian_raw)
        except Exception as cv_err:
            print(f"[Predict] Texture variance check warning: {cv_err}")
            texture_variance = 0.0

        # If texture variance is below threshold, immediately intercept smooth non-fabric objects, walls, or blanks
        if texture_variance < TEXTURE_THRESHOLD:
            rejection_message = "Image lacks textile texture or detail. Please scan a clear cotton fabric."
            scan_id = "h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
            try:
                metrics, scan_id = increment_scan_count(
                    stain_name="Unidentified Object",
                    fabric_name="Unidentified / N/A",
                    confidence=0,
                    email=user_email,
                    image=data.get("image", "")
                )
            except Exception as scan_err:
                print(f"[Predict] Warning: Failed to record smooth/texture-less scan: {scan_err}")

            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
            print(f"[PREDICT] [OOD TEXTURE REJECTED] Intercepted by Laplacian texture guardrail: variance={round(texture_variance, 2)} < {TEXTURE_THRESHOLD} | User='{user_email or 'anonymous'}' | IP={client_ip}")

            return jsonify({
                "id":                   scan_id,
                "status":               "success",
                "is_classified":        False,
                "is_valid_fabric":      False,
                "stain_type":           "Unidentified Object",
                "stain":                "Unidentified Object",
                "stain_classification": "Unidentified Object",
                "fabric_material":      "Unidentified / N/A",
                "fabric_color":         "N/A",
                "fabric_type":          "Unidentified / N/A",
                "confidence":           0,
                "confidence_score":     0.0,
                "texture_variance":     round(texture_variance, 2),
                "margin":               0.0,
                "probabilities": {
                    "Black Ballpen Ink": 0.0,
                    "Used Cooking Oil":  0.0,
                    "Mud":               0.0
                },
                "fabric":               "Unidentified / N/A",
                "message":              rejection_message,
                "rejection_message":    rejection_message,
                "caution":              rejection_message,
                "steps":                [],
                "materials":            [],
                "recommendation": {
                    "id":               scan_id,
                    "status":           "inconclusive",
                    "is_classified":    False,
                    "is_valid_fabric":  False,
                    "stain_type":       "Unidentified Object",
                    "steps":            [],
                    "caution":          rejection_message,
                    "materials":        [],
                    "fabric_color":     "N/A",
                    "fabric_type":      "Unidentified / N/A"
                }
            }), 200

        # d) Image Quality & Out-of-Distribution Plausibility Checks
        # Prevent CLAHE contrast equalization from inflating flat/uniform sensor noise into artificial "Mud"
        is_flat_surface = False
        is_vivid_non_fabric = False
        contrast_std = 0.0
        mean_sat = 0.0

        try:
            arr_raw = np.array(image.resize((224, 224), Image.BILINEAR), dtype=np.uint8)
            gray_raw = arr_raw.mean(axis=2)
            contrast_std = float(np.std(gray_raw))
            is_flat_surface = (contrast_std < 4.0)
        except Exception as ood_contrast_err:
            print(f"[Predict] Contrast check warning: {ood_contrast_err}")
            arr_raw = np.zeros((224, 224, 3), dtype=np.uint8)

        # Check HSV color saturation to reject vivid non-fabric objects (e.g., Persian rugs, neon objects)
        try:
            hsv = cv2.cvtColor(arr_raw, cv2.COLOR_RGB2HSV)
            mean_sat = float(np.mean(hsv[:, :, 1]))
            is_vivid_non_fabric = (mean_sat > 75.0)
        except Exception as ood_sat_err:
            print(f"[Predict] Saturation check warning: {ood_sat_err}")

        # Preprocess uploaded camera/gallery image for CNN inference
        try:
            input_arr = preprocess_image(image)  # (1, 224, 224, 3)
        except Exception as prep_err:
            print(f"[Predict] Image preprocessing failed: {prep_err}")
            return jsonify({
                "error": "Image preprocessing failed.",
                "status": "error",
                "details": str(prep_err)
            }), 500

        # Run real TensorFlow CNN inference
        try:
            predictions = model.predict(input_arr, verbose=0)  # (1, 3) softmax probabilities
            probs       = predictions[0]                        # (3,)
        except Exception as inf_err:
            print(f"[Predict] Model inference error: {inf_err}\n{traceback.format_exc()}")
            return jsonify({
                "error": "CNN model inference failed during prediction.",
                "status": "error",
                "details": str(inf_err)
            }), 500
        
        # Calculate max_confidence and sort predictions to evaluate top-1 vs top-2 class probabilities
        max_confidence     = float(np.max(predictions))
        order              = np.argsort(probs)[::-1]
        top1_idx           = int(order[0])
        top2_idx           = int(order[1])
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
        try:
            rec = get_recommendation(predicted_label, fabric_context, image=image)
            fabric_color = rec.get("fabric_color", "White Cotton")
        except Exception as rec_err:
            print(f"[Predict] Recommendation retrieval warning: {rec_err}")
            rec = {
                "fabric_color": "White Cotton",
                "steps": ["Apply mild detergent and rinse thoroughly with clean water."],
                "caution": "Test cleaning agent on an inconspicuous area first.",
                "materials": ["Clean Cloth", "Mild Liquid Detergent", "Water"]
            }
            fabric_color = "White Cotton"

        # =========================================================
        # LAYER 2: SOFTMAX CONFIDENCE THRESHOLD
        # =========================================================
        MARGIN_THRESHOLD = 0.20  # Minimum separation between top-1 and top-2 classes
        is_low_confidence = (max_confidence < CONFIDENCE_THRESHOLD)
        is_narrow_margin  = (margin < MARGIN_THRESHOLD)
        is_unidentified   = is_low_confidence or is_narrow_margin or is_flat_surface or is_vivid_non_fabric

        fallback_scan_id = "h_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")

        if is_unidentified:
            unidentified_title = "Unidentified Object"
            if is_low_confidence:
                rejection_message = "Scan rejected: Unrecognized surface pattern or low model certainty."
            elif is_flat_surface:
                rejection_message = "No visible stain pattern detected. Garment surface appears clean, uniform, or non-fabric."
            elif is_vivid_non_fabric:
                rejection_message = "Pattern or color distribution not recognized as white or grey cotton fabric."
            elif is_narrow_margin:
                rejection_message = f"Ambiguous scan. Model uncertainty between stain types (narrow separation of {round(margin*100, 1)}%). Please capture a clearer, centered photo of a stained cotton garment."
            else:
                rejection_message = "Scan rejected: Unrecognized surface pattern or low model certainty."

            scan_id = fallback_scan_id
            try:
                metrics, scan_id = increment_scan_count(
                    stain_name="Unidentified Object",
                    fabric_name="Unidentified / N/A",
                    confidence=confidence_percent,
                    email=user_email,
                    image=data.get("image", "")
                )
            except Exception as scan_err:
                print(f"[Predict] Warning: Failed to record unidentified scan: {scan_err}")

            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
            print(f"[PREDICT] [OOD REJECTED] Scan rejected by guardrail: stain='Unidentified Object', conf={round(max_confidence, 4)} (<{CONFIDENCE_THRESHOLD}), margin={margin_percent}% | User='{user_email or 'anonymous'}' | IP={client_ip}")

            return jsonify({
                "id":                   scan_id,
                "status":               "success",
                "is_classified":        False,
                "is_valid_fabric":      False,
                "confidence":           max_confidence,
                "confidence_score":     round(max_confidence, 4),
                "stain_type":           "Unidentified Object",
                "stain":                unidentified_title,
                "stain_classification": "Unidentified Object",
                "fabric_material":      "Unidentified / N/A",
                "fabric_color":         "N/A",
                "fabric_type":          "Unidentified / N/A",
                "texture_variance":     round(texture_variance, 2),
                "margin":               margin_percent,
                "probabilities":        probabilities,
                "fabric":               "Unidentified / N/A",
                "message":              rejection_message,
                "rejection_message":    rejection_message,
                "caution":              rejection_message,
                "steps":                [],
                "materials":            [],
                "recommendation": {
                    "id":            scan_id,
                    "status":        "inconclusive",
                    "is_classified": False,
                    "is_valid_fabric": False,
                    "stain_type":    "Unidentified Object",
                    "steps":         [],
                    "caution":       rejection_message,
                    "materials":     [],
                    "fabric_color":  "N/A",
                    "fabric_type":   "Unidentified / N/A"
                }
            }), 200

        # =========================================================
        # VALID PREDICTIONS (Both Layer 1 & Layer 2 checks passed)
        # =========================================================
        scan_id = fallback_scan_id
        try:
            metrics, scan_id = increment_scan_count(
                stain_name=predicted_label,
                fabric_name=f"{fabric_color} (100%)",
                confidence=confidence_percent,
                email=user_email,
                image=data.get("image", "")
            )
        except Exception as scan_err:
            print(f"[Predict] Warning: Failed to record confident scan: {scan_err}")

        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
        print(f"[PREDICT] [SUCCESS] CNN Inference completed: Stain='{predicted_label}' ({confidence_percent}%) | Fabric='{fabric_color}' | User='{user_email or 'anonymous'}' | IP={client_ip}")

        # Return authentic CNN model prediction with probability distribution, fabric-specific steps and caution
        return jsonify({
            "id":                   scan_id,
            "status":               "success",
            "is_classified":        True,
            "is_valid_fabric":      True,
            "stain_type":           predicted_label,
            "stain_classification": predicted_raw,
            "fabric_material":      "Cotton",
            "fabric_color":         fabric_color,
            "confidence":           confidence_percent,
            "confidence_score":     round(max_confidence, 4),
            "texture_variance":     round(texture_variance, 2),
            "margin":               margin_percent,
            "probabilities":        probabilities,
            "stain":                predicted_label,
            "fabric":               f"{fabric_color} (100%)",
            "message":              f"Successfully recognized {predicted_label} on {fabric_color}.",
            "steps":                rec.get("steps", []),
            "caution":              rec.get("caution", ""),
            "materials":            rec.get("materials", []),
            "recommendation": {
                "id":           scan_id,
                "status":       "success",
                "is_classified": True,
                "is_valid_fabric": True,
                "stain_type":   predicted_label,
                "steps":        rec.get("steps", []),
                "caution":      rec.get("caution", ""),
                "materials":    rec.get("materials", []),
                "fabric_color": fabric_color
            }
        }), 200

    except Exception as unhandled_err:
        print(f"[Predict] Unhandled exception in /predict: {unhandled_err}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"An unexpected error occurred during prediction: {str(unhandled_err)}",
            "status": "error"
        }), 500

if __name__ == "__main__":
    load_model(force_reload=True)
    # Run server dynamically binding to port from environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
