import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Import the Price Prediction Engine
from predict_scrap_image_and_price import predict_image_and_price, CLASSES, SUPER_CATEGORIES

# --- 1. DYNAMIC CLASS INDEX LOADING ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_index.json")

if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        raw_idx = json.load(f)
        # Ensure keys are integer-ordered
        MODEL_CLASSES = [raw_idx[str(i)] for i in range(len(raw_idx))]
    print(f" Loaded {len(MODEL_CLASSES)} classes dynamically from {CLASS_INDEX_PATH}:")
    print(MODEL_CLASSES)
else:
    MODEL_CLASSES = CLASSES
    print(f" Falling back to default {len(MODEL_CLASSES)} classes.")

NUM_CLASSES = len(MODEL_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 2. MODEL DEFINITIONS & CHECKPOINT LOADERS ---
class DINOv2ScrapClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)


def build_mobilenet(num_classes=NUM_CLASSES):
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


dinov2_path = os.path.join(MODEL_DIR, "dinov2_vits14_best.pt")
mobilenet_path = os.path.join(MODEL_DIR, "mobilenetv3_best.pt")

loaded_models = {}

# Try loading DINOv2
if os.path.exists(dinov2_path):
    try:
        dino = DINOv2ScrapClassifier(NUM_CLASSES)
        dino.load_state_dict(torch.load(dinov2_path, map_location=device, weights_only=True))
        dino.to(device).eval()
        loaded_models["DINOv2 (High Accuracy)"] = dino
        print(f" Loaded DINOv2 model ({dinov2_path})")
    except Exception as e:
        print(f"Note on DINOv2 load: {e}")

# Try loading MobileNetV3
try:
    mb = build_mobilenet(NUM_CLASSES)
    if os.path.exists(mobilenet_path):
        try:
            mb.load_state_dict(torch.load(mobilenet_path, map_location=device, weights_only=True))
            print(f" Loaded trained MobileNetV3 ({mobilenet_path})")
        except Exception as e:
            print(f" Using initialized MobileNetV3 weights: {e}")
    mb.to(device).eval()
    loaded_models["MobileNetV3 (Fast Edge)"] = mb
except Exception as e:
    print(f"Note on MobileNet load: {e}")

# Image Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def analyze_scrap(image, city, user_offered_price, model_choice):
    if image is None:
        return "Please upload an image first!"

    active_model = loaded_models.get(model_choice)
    if active_model is None:
        if loaded_models:
            active_model = next(iter(loaded_models.values()))
        else:
            return "No vision model is currently loaded!"

    # 1. Image preprocessing
    img = Image.fromarray(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 2. Vision Model Inference (clean logits, no artificial bias subtraction)
    with torch.no_grad():
        logits = active_model(input_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.dim() > 1:
            logits = logits[0]

        probabilities = torch.nn.functional.softmax(logits, dim=0)
        prob_list = probabilities.cpu().numpy()

        class_probs = {MODEL_CLASSES[i]: float(prob_list[i]) for i in range(len(MODEL_CLASSES))}
        sorted_candidates = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        top_material, conf_score = sorted_candidates[0]

    # 3. Price Engine Valuation
    try:
        offered_price_float = float(user_offered_price) if user_offered_price and user_offered_price.strip() else None
    except ValueError:
        offered_price_float = None

    final_result = predict_image_and_price(
        material_name=top_material,
        city=city,
        user_offered_price=offered_price_float,
        confidence=conf_score
    )

    final_result['model_used'] = model_choice
    final_result['super_category'] = SUPER_CATEGORIES.get(top_material, "General Scrap")
    final_result['top_3_predictions'] = [
        {'material': mat, 'confidence': f"{prob * 100:.2f}%", 'super_category': SUPER_CATEGORIES.get(mat, "")}
        for mat, prob in sorted_candidates[:3]
    ]

    return json.dumps(final_result, indent=4, ensure_ascii=False)


# --- 3. GRADIO USER INTERFACE ---
available_choices = list(loaded_models.keys()) if loaded_models else ["MobileNetV3 (Fast Edge)"]
default_choice = "DINOv2 (High Accuracy)" if "DINOv2 (High Accuracy)" in available_choices else available_choices[0]

with gr.Blocks(theme=gr.themes.Soft(), title="Kabadiwala AI Scrap Intelligence") as demo:
    gr.Markdown("# ♻️ Kabadiwala AI — Vision & Pan-India Scrap Valuation")
    gr.Markdown("Upload an image of scrap material (electronics, PCB, copper, cables, plastic, cardboard, wood, appliances). The Vision model identifies the exact grade, and the Price Engine calculates the fair market price.")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label="Upload Scrap Image", type="numpy")
            model_choice_dropdown = gr.Dropdown(
                choices=available_choices,
                value=default_choice,
                label="Select Vision Model"
            )
            city_input = gr.Dropdown(
                choices=["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Kolkata", "Jaipur", "Ahmedabad", "Indore", "Noida"],
                value="Delhi",
                label="City (Local Mandi Adjustment)"
            )
            price_input = gr.Textbox(
                label="Dealer Offered Price (₹/kg) [Optional]",
                placeholder="e.g. 150 (Leave blank for fair valuation only)"
            )
            submit_btn = gr.Button("🔍 Analyze & Value Scrap", variant="primary")

        with gr.Column(scale=1):
            output_json = gr.Code(label="AI Final Output (Vision + Price Intelligence)", language="json")

    submit_btn.click(
        fn=analyze_scrap,
        inputs=[img_input, city_input, price_input, model_choice_dropdown],
        outputs=output_json
    )

if __name__ == "__main__":
    print("\n App is ready! Starting local server...")
    demo.launch()
