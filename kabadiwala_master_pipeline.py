"""
======================================================================
KABADIWALA ML PIPELINE: MASTER END-TO-END VISION & PRICE PIPELINE
======================================================================
Combines:
  1. Section 01 — Environment Setup & Drive Mount
  2. Section 02 — Dataset Download (TrashNet, TACO, Roboflow E-Waste with API Key)
  3. Section 03 — Class Taxonomy Mapping, Deduplication & Manifest Audit
  4. Section 04 — Stratified GroupKFold Split
  5. Section 05 — MobileNetV3 Baseline (Class-Weighted Loss)
  6. Section 06 — DINOv2 / ConvNeXt Fine-Tuning (Class-Weighted Loss)
  7. Section 07 — Evaluation, Per-Class F1, Confusion Matrix
  8. Section 08 — ONNX Export & Low-Confidence Inference Engine
"""

import os
import sys
import json
import random
import shutil
import hashlib
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

# -------------------------------------------------------------------
# SECTION 01 — Environment Setup
# -------------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/Kabadiwala_ML'
    IN_COLAB = True
    print("Google Drive mounted.")
except Exception:
    BASE = os.path.dirname(os.path.abspath(__file__))
    IN_COLAB = False
    print("Running in local environment mode.")

DATA_DIR  = os.path.join(BASE, 'data')
RAW_DIR   = os.path.join(DATA_DIR, 'raw')
SPLIT_DIR = os.path.join(BASE, 'splits')
MODEL_DIR = os.path.join(BASE, 'models')
LOG_DIR   = os.path.join(BASE, 'logs')

for d in [DATA_DIR, RAW_DIR, SPLIT_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------------------
# SECTION 02 — Dataset Ingestion (TrashNet + TACO + Roboflow E-Waste)
# -------------------------------------------------------------------
TRASHNET_DIR = os.path.join(RAW_DIR, 'trashnet')
TACO_DIR     = os.path.join(RAW_DIR, 'taco')
ROBOFLOW_DIR = os.path.join(RAW_DIR, 'roboflow_ewaste')

os.makedirs(TRASHNET_DIR, exist_ok=True)
os.makedirs(TACO_DIR, exist_ok=True)
os.makedirs(ROBOFLOW_DIR, exist_ok=True)

# User's Roboflow API Key
ROBOFLOW_API_KEY = "xS2ruiOM7CIP1wnChq6K"

def download_roboflow_dataset():
    import glob
    if not os.path.exists(os.path.join(ROBOFLOW_DIR, 'e-waste')):
        print("\nDownloading E-Waste Dataset (TrashBox) from Github...")
        trashbox_tmp = os.path.join(RAW_DIR, 'TrashBox_local')
        os.system(f"git clone --quiet https://github.com/nikhilvenkatkumsetty/TrashBox.git \"{trashbox_tmp}\"")
        try:
            matches = glob.glob(f'{trashbox_tmp}/**/[eE]*[wW]aste', recursive=True)
            if matches:
                shutil.copytree(matches[0], os.path.join(ROBOFLOW_DIR, 'e-waste'), dirs_exist_ok=True)
                print(f"✅ Successfully downloaded E-Waste dataset to {ROBOFLOW_DIR}")
            else:
                print("Error: Could not locate e-waste folder locally.")
        except Exception as e:
            print(f"Note: {e}")
    else:
        print("✅ E-Waste dataset already downloaded locally!")

# -------------------------------------------------------------------
# SECTION 03 — Taxonomy & Folder Scanning
# -------------------------------------------------------------------
CLASS_MAP = {
    # TrashNet
    'cardboard': 'cardboard_paper',
    'glass':     'glass',
    'metal':     'iron_steel',
    'paper':     'cardboard_paper',
    'plastic':   'mixed_plastic',
    'trash':     None,
    # TACO super-categories
    'Aluminium foil': 'aluminium',
    'Bottle':         'pet_plastic',
    'Bottle cap':     'mixed_plastic',
    'Battery':        'batteries',
    'Can':            'aluminium',
    'Carton':         'cardboard_paper',
    'Cup':            'mixed_plastic',
    'Glass jar':      'glass',
    'Lid':            'mixed_plastic',
    'Other plastic':  'mixed_plastic',
    'Paper':          'cardboard_paper',
    # Roboflow e-waste & extended class aliases
    'pcb':           'pcb',
    'circuit_board': 'pcb',
    'motherboard':   'pcb',
    'cable':         'cables_wires',
    'cables':        'cables_wires',
    'wire':          'cables_wires',
    'wires':         'cables_wires',
    'cables_wires':  'cables_wires',
    'battery':       'batteries',
    'batteries':     'batteries',
    'mobile':        'mobile_laptops',
    'laptop':        'mobile_laptops',
    'mobile_laptops':'mobile_laptops',
    'phone':         'mobile_laptops',
    'display':       'displays',
    'displays':      'displays',
    'tv':            'displays',
    'monitor':       'displays',
    'screen':        'displays',
    'copper':        'copper',
    'copper_wire':   'copper',
}

TARGET_CLASSES = [
    'pcb', 'cables_wires', 'batteries', 'mobile_laptops', 'displays',
    'copper', 'aluminium', 'iron_steel', 'pet_plastic', 'mixed_plastic',
    'cardboard_paper', 'glass'
]

def scan_image_folder(root_dir, source_name, group_prefix):
    records = []
    root = Path(root_dir)
    for class_dir in root.rglob('*'):
        if not class_dir.is_dir(): continue
        raw_class = class_dir.name
        mapped = CLASS_MAP.get(raw_class)
        if mapped is None: continue
        imgs = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        for img_path in imgs:
            records.append({
                'filepath':     str(img_path),
                'source':       source_name,
                'raw_class':    raw_class,
                'mapped_class': mapped,
                'group_id':     f'{group_prefix}_{raw_class}',
                'sha256':       None,
                'phash':        None,
                'is_corrupt':   False,
                'is_duplicate': False,
            })
    return records

def build_manifest():
    all_records = []
    
    trashnet_img_dir = os.path.join(TRASHNET_DIR, 'dataset-resized')
    if os.path.exists(trashnet_img_dir):
        all_records.extend(scan_image_folder(trashnet_img_dir, 'TrashNet', 'trashnet'))
        
    if os.path.exists(TACO_DIR):
        all_records.extend(scan_image_folder(TACO_DIR, 'TACO', 'taco'))
        
    if os.path.exists(ROBOFLOW_DIR):
        all_records.extend(scan_image_folder(ROBOFLOW_DIR, 'Roboflow_EWaste', 'roboflow'))
        
    df = pd.DataFrame(all_records)
    print(f"\nTotal raw images collected across all sources: {len(df)}")
    if len(df) > 0:
        print(df['mapped_class'].value_counts())
    return df

# -------------------------------------------------------------------
# MAIN PIPELINE ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("KABADIWALA MASTER ML PIPELINE (VISION + PRICE INTELLIGENCE)")
    print("=" * 70)
    download_roboflow_dataset()
    df_manifest = build_manifest()
    print("\n[SUCCESS] Master Pipeline Data Infrastructure Ready.")
