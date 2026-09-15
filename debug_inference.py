import os
import torch
import torch.nn as nn
from torchvision import models

model_path = os.path.join("models", "mobilenetv3_best.pt")
dinov2_path = os.path.join("models", "dinov2_vits14_best.pt")

print("Checking MobileNetV3 checkpoint:")
if os.path.exists(model_path):
    sd = torch.load(model_path, map_location='cpu', weights_only=True)
    print("Keys in state_dict:", list(sd.keys())[-5:])
    if 'classifier.3.weight' in sd:
        w = sd['classifier.3.weight']
        b = sd['classifier.3.bias']
        print("Classifier shape:", w.shape)
        print("Classifier weight mean per class:", w.mean(dim=1))
        print("Classifier bias values:", b)

print("\nChecking DINOv2 checkpoint:")
if os.path.exists(dinov2_path):
    sd_dino = torch.load(dinov2_path, map_location='cpu', weights_only=True)
    print("DINOv2 Keys in state_dict:", list(sd_dino.keys())[-5:])
