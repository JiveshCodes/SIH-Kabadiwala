# Kabadiwala ML Pipeline & Model Package

This repository contains the complete Machine Learning pipeline, pre-trained vision models, price estimation models, safety guidance modules, and interactive demonstration applications for the **Kabadiwala Scrap Recycling System** (SIH).

## 🚀 Key Components & Models Included

- **`models/mobilenetv3.onnx`** (~0.34 MB) — Lightweight deployable MobileNetV3 vision classifier (ONNX runtime format).
- **`models/mobilenetv3_best.pt`** (~17.1 MB) — MobileNetV3 PyTorch checkpoint.
- **`models/dinov2_vits14_best.pt`** (~88.7 MB) — High-accuracy DINOv2 vision model checkpoint.
- **`models/price_model.joblib`** (~1.07 MB) — Local price estimation model pipeline.
- **`models/class_index.json`** — Output index mapping to scrap category names.
- **`models/price_model_metadata.json`** — Price model schema & metadata.

## 🛠️ How to Setup & Run

### 1. Install Dependencies
```bash
pip install torch torchvision onnxruntime joblib pandas numpy scikit-learn streamlit Pillow
```

### 2. Run Scrap Classifier & Price Estimator Demo
```bash
streamlit run demo_app.py
```
*or directly test on sample images:*
```bash
python test_models_on_samples.py
```

### 3. Run End-to-End Pipeline Script
```bash
python kabadiwala_master_pipeline.py
```

## 📁 Repository Structure

```
├── models/                         # Pre-trained vision & price models (tracked in Git)
│   ├── class_index.json
│   ├── dinov2_vits14_best.pt
│   ├── mobilenetv3.onnx
│   ├── mobilenetv3_best.pt
│   ├── price_model.joblib
│   └── price_model_metadata.json
├── demo_app.py                     # Streamlit Web UI for scrap identification & pricing
├── test_models_on_samples.py       # Inference testing script
├── kabadiwala_master_pipeline.py   # Complete end-to-end training & evaluation pipeline
├── 08_price_data_and_model.py      # Price model training & dataset ingestion
├── safety_guidance.py              # E-waste & scrap safety handling instructions
└── recycler_recommender.py         # Recycler matching & location recommendation logic
```

---
*Maintained for Kabadiwala SIH Project.*
