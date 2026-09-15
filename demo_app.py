import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import json

# Import the Price Prediction Engine we discussed!
from predict_scrap_image_and_price import predict_image_and_price, CLASSES

# --- 1. SET UP THE VISION MODEL (Model 1) ---
# We must recreate the exact MobileNetV3 brain structure before loading the weights
NUM_CLASSES = len(CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading Vision Model...")
model = models.mobilenet_v3_large(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)

# Path to the model you just downloaded from Google Drive
model_path = os.path.join("models", "mobilenetv3_best.pt")

if not os.path.exists(model_path):
    print(f"ERROR: Could not find {model_path}. Make sure you downloaded it from Drive!")
else:
    # Load the trained brain into the structure
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Vision Model Loaded Successfully!")

# Standard ImageNet transformations that MobileNet expects
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def analyze_scrap(image, city, user_offered_price):
    """
    This function takes the uploaded image from the UI, runs it through Model 1,
    and then passes the result to Model 2.
    """
    if image is None:
        return "Please upload an image first!"

    if not os.path.exists(model_path):
        return f"Model file not found at {model_path}. Please download it from Google Drive."

    # 1. Prepare image
    img = Image.fromarray(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 2. Vision Model Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get the top prediction
        confidence, predicted_idx = torch.max(probabilities, 0)
        detected_material = CLASSES[predicted_idx.item()]
        conf_score = confidence.item()

    # 3. Pass to Price Engine (Model 2)
    # If the user didn't type a price, we pass None
    try:
        offered_price_float = float(user_offered_price) if user_offered_price.strip() else None
    except ValueError:
        offered_price_float = None

    final_result = predict_image_and_price(
        material_name=detected_material,
        city=city,
        user_offered_price=offered_price_float,
        confidence=conf_score
    )

    # Format the dictionary nicely for the UI
    return json.dumps(final_result, indent=4, ensure_ascii=False)


# --- 2. BUILD THE USER INTERFACE ---
print("Building Web Interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ♻️ Kabadiwala AI - End-to-End Test")
    gr.Markdown("Upload a photo of scrap. The Vision Model will classify it, and the Price Engine will calculate its fair market value in your city.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Scrap Image")
            city_input = gr.Dropdown(
                choices=["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Kolkata"], 
                value="Delhi", 
                label="Select City"
            )
            price_input = gr.Textbox(
                label="Dealer Offered Price (₹/kg) [Optional]", 
                placeholder="e.g. 150 (Leave blank just to see fair value)"
            )
            submit_btn = gr.Button("Analyze Scrap", variant="primary")
            
        with gr.Column():
            output_json = gr.Code(label="AI Final Output (Vision + Price)", language="json")

    submit_btn.click(
        fn=analyze_scrap,
        inputs=[img_input, city_input, price_input],
        outputs=output_json
    )

if __name__ == "__main__":
    print("\n🚀 App is ready! Click the local URL below to open it in your browser:")
    demo.launch()
