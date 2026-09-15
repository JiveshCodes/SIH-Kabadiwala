"""
======================================================================
KABADIWALA REST API BACKEND ENGINE (SIH #26229)
======================================================================
Provides the production-grade REST API endpoint `/predict` for:
  1. Image-based scrap material recognition (Vision Model 1)
  2. Pan-India fair market valuation & anomaly detection (Price Model 2)
  3. Vernacular safety hazard warnings (Hindi, Marathi, English)
  4. CPCB/SPCB authorized recycler recommendation & matching

Framework: FastAPI (runs via `uvicorn api_server:app --port 8000`)
Fallback: Flask (runs via `python api_server.py`)
"""

import os
import io
import json
from PIL import Image

# Import existing core modules without replacing anything
from predict_scrap_image_and_price import (
    predict_image_and_price,
    CLASSES,
    SUPER_CATEGORIES,
    resolve_canonical_class
)
from safety_guidance import get_safety_guidance
from recycler_recommender import rank_authorized_recyclers

# -------------------------------------------------------------------
# 1. OPTIONAL TORCH / VISION MODEL SETUP (Lazy Loading)
# -------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ REST API NOTICE: 'torch' is not installed in current Python environment. Running in lightweight mode.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_index.json")

if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        raw_idx = json.load(f)
        MODEL_CLASSES = [raw_idx[str(i)] for i in range(len(raw_idx))]
else:
    MODEL_CLASSES = CLASSES

NUM_CLASSES = len(MODEL_CLASSES)

_vision_model_cache = None
_img_transform_cache = None

def get_vision_model():
    """Lazy loader for MobileNetV3 model to ensure instant server startup."""
    global _vision_model_cache, _img_transform_cache
    if not HAS_TORCH:
        return None, None

    if _vision_model_cache is None:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            m = models.mobilenet_v3_large(weights=None)  # Avoid blocking download
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
            
            mobilenet_path = os.path.join(MODEL_DIR, "mobilenetv3_best.pt")
            if os.path.exists(mobilenet_path):
                m.load_state_dict(torch.load(mobilenet_path, map_location=device, weights_only=True))
                print(f"✅ Loaded MobileNetV3 ({mobilenet_path})")

            m.to(device).eval()
            _vision_model_cache = (m, device)

            _img_transform_cache = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Vision model lazy load error: {e}")
            return None, None

    return _vision_model_cache, _img_transform_cache


# -------------------------------------------------------------------
# 2. CORE PREDICTION LOGIC
# -------------------------------------------------------------------
def process_prediction_request(image_bytes=None, material_name=None, city="Delhi", user_offered_price=None, lang="hi"):
    """
    Unified prediction handler for both image uploads and direct material queries.
    """
    top_material = None
    conf_score = 0.95
    top_3 = []

    model_info, transform_fn = get_vision_model()

    # Case A: Image file provided & PyTorch available -> Run Vision Model
    if image_bytes and model_info and transform_fn:
        try:
            v_model, dev = model_info
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            tensor = transform_fn(img).unsqueeze(0).to(dev)

            with torch.no_grad():
                logits = v_model(tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.dim() > 1:
                    logits = logits[0]

                probs = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
                class_probs = {MODEL_CLASSES[i]: float(probs[i]) for i in range(len(MODEL_CLASSES))}
                sorted_candidates = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)

                top_material, conf_score = sorted_candidates[0]
                top_3 = [
                    {"material": m, "confidence": f"{p * 100:.2f}%", "super_category": SUPER_CATEGORIES.get(m, "General")}
                    for m, p in sorted_candidates[:3]
                ]
        except Exception as e:
            print(f"Vision inference error: {e}")
            top_material = resolve_canonical_class(material_name) if material_name else "mixed_plastic"
    # Case B: Direct material parameter passed or fallback
    elif material_name:
        top_material = resolve_canonical_class(material_name)
    else:
        top_material = "mixed_plastic"

    # Price & Anomaly Valuation
    offered_price_float = float(user_offered_price) if user_offered_price is not None and str(user_offered_price).strip() != "" else None

    price_res = predict_image_and_price(
        material_name=top_material,
        city=city,
        user_offered_price=offered_price_float,
        confidence=conf_score
    )

    if top_3:
        price_res['top_3_candidates'] = top_3

    # Safety Hazard Guidance
    safety_info = get_safety_guidance(top_material, lang=lang)

    # Authorized Recycler Recommendations
    sample_lot = [{"material": top_material, "weight_kg": 10.0, "estimated_rate": 100.0}]
    recommended_recyclers = rank_authorized_recyclers(sample_lot, collector_city=city)

    return {
        "success": True,
        "prediction": price_res,
        "safety_guidance": safety_info,
        "recommended_recyclers": recommended_recyclers[:3]
    }


# -------------------------------------------------------------------
# 3. FASTAPI BACKEND INTEGRATION
# -------------------------------------------------------------------
try:
    from fastapi import FastAPI, File, UploadFile, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="Kabadiwala AI REST API",
        description="Scrap Image Recognition, Price Intelligence, Safety Engine & Authorized Recycler Matching",
        version="2.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    @app.get("/health")
    def health_check():
        return {
            "status": "online",
            "service": "Kabadiwala AI ML Backend",
            "pytorch_available": HAS_TORCH,
            "documentation_ui": "http://127.0.0.1:8000/docs",
            "classes_supported": NUM_CLASSES
        }

    @app.post("/predict")
    async def predict_endpoint(
        file: UploadFile = File(None),
        material_name: str = Form(None),
        city: str = Form("Delhi"),
        user_offered_price: str = Form(None),
        lang: str = Form("hi")
    ):
        image_bytes = None
        if file is not None:
            image_bytes = await file.read()

        result = process_prediction_request(
            image_bytes=image_bytes,
            material_name=material_name,
            city=city,
            user_offered_price=user_offered_price,
            lang=lang
        )
        return JSONResponse(content=result)

    HAS_FASTAPI = True

except ImportError:
    HAS_FASTAPI = False


# -------------------------------------------------------------------
# 4. FLASK BACKEND FALLBACK INTEGRATION
# -------------------------------------------------------------------
if not HAS_FASTAPI:
    try:
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route("/health", methods=["GET"])
        @app.route("/", methods=["GET"])
        def health_check():
            return jsonify({
                "status": "online",
                "service": "Kabadiwala AI ML Backend (Flask)",
                "pytorch_available": HAS_TORCH,
                "classes_supported": NUM_CLASSES
            })

        @app.route("/predict", methods=["POST"])
        def predict_endpoint():
            file = request.files.get("file") or request.files.get("image")
            image_bytes = file.read() if file else None

            data = request.form if request.form else (request.get_json(silent=True) or {})
            material_name = data.get("material_name")
            city = data.get("city", "Delhi")
            user_offered_price = data.get("user_offered_price")
            lang = data.get("lang", "hi")

            result = process_prediction_request(
                image_bytes=image_bytes,
                material_name=material_name,
                city=city,
                user_offered_price=user_offered_price,
                lang=lang
            )
            return jsonify(result)

    except ImportError:
        pass


if __name__ == "__main__":
    print("\n🚀 Starting Kabadiwala REST API Server on http://localhost:8000 ...")
    if HAS_FASTAPI:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        app.run(host="0.0.0.0", port=8000, debug=True)
