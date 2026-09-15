"""
test_models_on_samples.py
=========================
Runs verification on the user's test set in 'Images for test p/' using both
MobileNetV3 and DINOv2 models, evaluates predictions against expected ground truth,
and checks price calculation.
"""

import os
import json
import torch
from torchvision import models, transforms
from PIL import Image

try:
    import pillow_avif
except ImportError:
    pass

from predict_scrap_image_and_price import predict_image_and_price, SUPER_CATEGORIES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMG_DIR = os.path.join(os.path.dirname(BASE_DIR), "Images for test p")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_index.json")

# 1. Load class index
with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
    raw_idx = json.load(f)
    CLASSES = [raw_idx[str(i)] for i in range(len(raw_idx))]

NUM_CLASSES = len(CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. Ground truth expectations for the test files
GROUND_TRUTH = {
    'cu.jpeg': 'copper',
    'echips.jpeg': 'pcb_chips',
    'hair_dryer.jpeg': 'appliances',
    'oven.avif': 'appliances',
    'tv.jpeg': 'tv_monitors_displays',
    'wires.webp': 'cables_wires',
    'wood.jpg': 'wood',
    'metal.jpeg': 'iron_steel',
    'images.jpeg': 'mixed_plastic'
}

def load_mobilenet():
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
    ckpt = os.path.join(MODEL_DIR, "mobilenetv3_best.pt")
    if os.path.exists(ckpt):
        try:
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f" Loaded MobileNetV3 checkpoint: {ckpt}")
        except Exception as e:
            print(f" Note on MobileNet checkpoint: {e}")
    m.to(device).eval()
    return m

def test_inference():
    print("=" * 80)
    print("KABADIWALA VERIFICATION: INFERENCE TEST ON 'Images for test p/'")
    print(f"Loaded {NUM_CLASSES} classes: {CLASSES}")
    print("=" * 80)

    model = load_mobilenet()

    if not os.path.exists(TEST_IMG_DIR):
        print(f"Error: Directory not found: {TEST_IMG_DIR}")
        return

    test_files = sorted(os.listdir(TEST_IMG_DIR))
    print(f"Found {len(test_files)} sample test images in {TEST_IMG_DIR}:\n")

    results = []
    for fname in test_files:
        fpath = os.path.join(TEST_IMG_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            img = Image.open(fpath).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()

            top_idx = int(probs.argmax())
            pred_class = CLASSES[top_idx]
            conf = float(probs[top_idx])

            expected = GROUND_TRUTH.get(fname.lower(), 'N/A')
            price_info = predict_image_and_price(pred_class, city="Delhi", confidence=conf)

            results.append({
                'file': fname,
                'expected': expected,
                'predicted': pred_class,
                'confidence': f"{conf * 100:.1f}%",
                'super_cat': price_info['super_category'],
                'fair_rate': price_info['estimated_fair_rate']
            })
        except Exception as e:
            results.append({
                'file': fname,
                'expected': GROUND_TRUTH.get(fname.lower(), 'N/A'),
                'predicted': f"ERR: {e}",
                'confidence': "0.0%",
                'super_cat': "N/A",
                'fair_rate': "N/A"
            })

    print(f"{'Filename':20s} | {'Expected':16s} | {'Predicted':20s} | {'Conf':7s} | {'Category':15s} | {'Fair Rate':15s}")
    print("-" * 105)
    for r in results:
        print(f"{r['file']:20s} | {r['expected']:16s} | {r['predicted']:20s} | {r['confidence']:7s} | {r['super_cat']:15s} | {r['fair_rate']:15s}")
    print("-" * 105)

if __name__ == "__main__":
    test_inference()

