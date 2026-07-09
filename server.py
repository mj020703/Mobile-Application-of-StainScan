import base64
from io import BytesIO
import os
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

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
CORS(app)

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
    # Run server locally on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
