"""
download_roboflow_ewaste.py
===========================
Downloads the Roboflow E-Waste & Cables dataset using the provided API key
and organizes it into the target 12-class scrap taxonomy.
"""

import os
import sys
import json
import shutil
from pathlib import Path

ROBOFLOW_API_KEY = "xS2ruiOM7CIP1wnChq6K"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ROBOFLOW_DIR = os.path.join(RAW_DIR, "roboflow_ewaste")

os.makedirs(ROBOFLOW_DIR, exist_ok=True)

def download_dataset():
    print("=" * 70)
    print("DOWNLOADING ROBOFLOW E-WASTE & CABLES DATASET")
    print(f"Target Directory: {ROBOFLOW_DIR}")
    print("=" * 70)
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    # Download E-Waste 11-class dataset
    print("\nFetching E-Waste dataset version 1 from Roboflow Universe...")
    project = rf.workspace("e-waste-detection").project("e-waste-kxzga")
    dataset = project.version(1).download("folder", location=ROBOFLOW_DIR)
    
    print(f"\n[SUCCESS] Roboflow E-Waste dataset downloaded to: {ROBOFLOW_DIR}")

if __name__ == "__main__":
    download_dataset()
