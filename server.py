import base64
from io import BytesIO
import os
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import certifi

# Load configuration same as training
IMAGE_SIZE = (224, 224)
MODEL_PATH_PTH = "stainscan_model.pth"
CLASS_MAPPING = {
    0: "Black Ballpen Ink",
    1: "Used Cooking Oil",
    2: "Mud"
}

# Try importing torch/torchvision conditionally to support immediate fallback execution
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ModuleNotFoundError:
    print("PyTorch libraries not found. Backend API will run in Numpy/Demo Fallback mode.")
    HAS_TORCH = False

if HAS_TORCH:
    # Define the CNN architecture (must match train.py)
    class MobileNetV2Stain(nn.Module):
        def __init__(self, num_classes=3, pretrained=False):
            super(MobileNetV2Stain, self).__init__()
            import torchvision.models as models
            self.base_model = models.mobilenet_v2()
            
            # Modify first conv layer to accept 5 input channels
            original_conv = self.base_model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=5,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                dilation=original_conv.dilation,
                groups=original_conv.groups,
                bias=original_conv.bias is not None
            )
            self.base_model.features[0][0] = new_conv
            
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.base_model(x)

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Custom transform block to stack HSV Saturation & Value channels onto RGB channels
    class AddHSVChannels(object):
        def __call__(self, img):
            rgb_tensor = transforms.ToTensor()(img)
            hsv_img = img.convert("HSV")
            hsv_tensor = transforms.ToTensor()(hsv_img)
            s_channel = hsv_tensor[1:2, :, :]
            v_channel = hsv_tensor[2:3, :, :]
            x_5ch = torch.cat([rgb_tensor, s_channel, v_channel], dim=0)
            return x_5ch

    # Validation Transforms (must match training exactly)
    inference_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        AddHSVChannels(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.5, 0.5])
    ])
else:
    device = "cpu"

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Default Knowledge Base Recommendations ---
DEFAULT_KB = {
    "Used Cooking Oil": {
        "materials": ["Liquid Dish Soap", "Warm Water", "Microfiber Cloth", "Baking Soda"],
        "steps": [
            "Blot the excess oil immediately using a clean paper towel. Do not rub, as this spreads the oil.",
            "Apply a generous amount of liquid dish soap directly to the stained area. Dish soap is designed to cut grease.",
            "Gently work the soap into the cotton fabric fibers with a soft cloth or toothbrush in circular motions.",
            "Let it stand for 5-10 minutes to allow the soap to break down the oil structure.",
            "Rinse the area thoroughly with warm water to flush out the grease-soap emulsion.",
            "Launder standardly at the highest safe temperature for the garment, then check the area before machine drying."
        ]
    },
    "Black Ballpen Ink": {
        "materials": ["Isopropyl Alcohol", "Cotton Balls", "Absorbent Towels", "Liquid Detergent"],
        "steps": [
            "Place an absorbent paper towel directly underneath the stained layer of the fabric to catch bleeding ink.",
            "Dab the stain generously using a cotton ball saturated with isopropyl alcohol.",
            "Blot repeatedly, switching to fresh cotton balls as they absorb the ink. Do not scrub, blot only.",
            "Rinse the stained fabric section thoroughly with cold water to remove the alcohol.",
            "Rub a small amount of liquid detergent into any remaining faint ink outline.",
            "Wash immediately in a regular laundry cycle, verifying the stain is gone before applying heat drying."
        ]
    },
    "Mud": {
        "materials": ["Laundry Brush", "Liquid Laundry Detergent", "Warm Water", "White Vinegar"],
        "steps": [
            "Allow the mud to dry completely. Attempting to clean wet mud will rub dirt deeper into cotton fibers.",
            "Scrape or brush off dry mud crust using a stiff-bristled brush.",
            "Pre-treat the remaining dirt spots with a small amount of liquid laundry detergent.",
            "Rub the fabric together gently under warm running water to release dirt particles.",
            "For stubborn brown mud stains, mix equal parts warm water and white vinegar, sponge the area, and let sit for 10 minutes.",
            "Rinse clean and launder normally in a warm wash cycle."
        ]
    }
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

# Fallback memory stores
if is_demo_mode:
    mock_db = {
        "users": [
            { "name": "Admin Manager", "email": "admin@stainscan.com", "password": "admin123", "role": "admin", "avatar": "admin" },
            { "name": "John Doe", "email": "user@stainscan.com", "password": "user123", "role": "user", "avatar": "John" }
        ],
        "scans": [],
        "logs": [
            { "time": datetime.utcnow().isoformat(), "type": "info", "text": "Database initialized successfully (Demo Mode)." }
        ],
        "kb": DEFAULT_KB
    }

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
    scan["id"] = "h_" + str(len(mock_db["scans"]) + 1)
    mock_db["scans"].append(scan)
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

# Global variables for model
model = None

def load_model():
    global model
    if model is not None:
        return
        
    print(f"Loading trained weights from {MODEL_PATH_PTH}...")
    if not HAS_TORCH:
        print("Warning: PyTorch not installed. Prediction API will run in demo/fallback mode.")
        return
        
    if not os.path.exists(MODEL_PATH_PTH):
        print(f"Warning: Trained weights file '{MODEL_PATH_PTH}' not found. Prediction API will run in demo/fallback mode.")
        return
        
    try:
        model = MobileNetV2Stain(num_classes=3)
        # Load weights on the correct device
        state_dict = torch.load(MODEL_PATH_PTH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully and set to evaluation mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# CORS headers handled dynamically by Flask-CORS middleware

@app.route("/health", methods=["GET"])
def health_check():
    global model
    # Reload model if weights became available later
    if model is None and os.path.exists(MODEL_PATH_PTH):
        load_model()
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight options requests
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
        
    global model
    # Proactively check/reload model weights
    if model is None:
        load_model()
        
    # Get request JSON data
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided in the request payload."}), 400
        
    base64_str = data["image"]
    
    # Strip base64 headers if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
        
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Fallback if model hasn't been trained yet
        if model is None:
            print("Server running in Demo/Simulation Fallback mode.")
            # Simple fallback heuristic based on image size to provide simulated variation
            img_w, img_h = image.size
            pseudo_seed = (img_w + img_h) % 3
            stains = ["Black Ballpen Ink", "Used Cooking Oil", "Mud"]
            confidences = [89, 92, 85]
            return jsonify({
                "stain": stains[pseudo_seed],
                "fabric": "Cotton Fabric (100%)",
                "confidence": confidences[pseudo_seed],
                "message": "Demo prediction (model weights not found)"
            })
            
        # Run inference
        image_tensor = inference_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence_val, predicted_idx = torch.max(probabilities, 0)
            
        confidence_percent = int(confidence_val.item() * 100)
        predicted_label = CLASS_MAPPING.get(predicted_idx.item(), "Unknown Stain")
        
        print(f"Prediction successful: {predicted_label} ({confidence_percent}% confidence)")
        
        return jsonify({
            "stain": predicted_label,
            "fabric": "Cotton Fabric (100%)",
            "confidence": confidence_percent
        })
        
    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    load_model()
    # Run server dynamically binding to port from environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
