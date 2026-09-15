"""
01_dataset_research.py
======================
Kabadiwala AI/ML Module - Stage 1: Dataset Research, Manifest Generation & Download Setup

This script sets up the dataset research infrastructure, defines all legitimate public
data sources, specifies their licenses and original URLs for hackathon verification,
and initializes standard manifest schemas for both Vision (Model 1) and Price (Model 2).
"""

import os
import json
import pandas as pd

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANIFEST_DIR = os.path.join(DATA_DIR, "manifests")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. TAXONOMY & CLASS DEFINITIONS (MODEL 1)
# ---------------------------------------------------------
TARGET_TAXONOMY = {
    "e_waste": [
        "pcb",
        "cables_wires",
        "batteries",
        "mobile_laptops",
        "displays"
    ],
    "general_recyclables": [
        "copper",
        "aluminium",
        "iron_steel",
        "pet_plastic",
        "mixed_plastic",
        "cardboard_paper",
        "glass"
    ]
}

ALL_CLASSES = TARGET_TAXONOMY["e_waste"] + TARGET_TAXONOMY["general_recyclables"]

# ---------------------------------------------------------
# 2. VISION DATASETS SPECIFICATIONS (FOR HACKATHON DEFENSE)
# ---------------------------------------------------------
VISION_DATASETS_META = [
    {
        "dataset_name": "TrashNet",
        "citation": "Gary Thung, Mindy Yang. Stanford CS229 Project (2016).",
        "url": "https://github.com/garythung/trashnet",
        "license": "MIT License",
        "target_classes_mapped": ["cardboard_paper", "glass", "iron_steel", "pet_plastic"],
        "approx_images": 2527,
        "environment_type": "Studio / White Background",
        "hackathon_justification": "Standard baseline for clean material optical features."
    },
    {
        "dataset_name": "TACO (Trash Annotations in Context)",
        "citation": "Pedro F. Proença, Pedro Simões. TACO Dataset (2020).",
        "url": "https://github.com/pedropro/TACO",
        "license": "MIT License",
        "target_classes_mapped": ["pet_plastic", "mixed_plastic", "aluminium", "glass", "batteries"],
        "approx_images": 1500,
        "environment_type": "Unconstrained Outdoor / Litter",
        "hackathon_justification": "Provides complex real-world outdoor backgrounds."
    },
    {
        "dataset_name": "DeepPCB",
        "citation": "Tang et al. DeepPCB Dataset for Circuit Board Defect & Recognition (2019).",
        "url": "https://github.com/tangsanli5201/DeepPCB",
        "license": "MIT License / Research",
        "target_classes_mapped": ["pcb"],
        "approx_images": 1500,
        "environment_type": "Industrial Circuit Board Scan",
        "hackathon_justification": "Fine-grained PCB pattern recognition."
    },
    {
        "dataset_name": "Roboflow E-Waste Collection",
        "citation": "Aggregated Roboflow Public E-Waste Computer Vision Projects (2022-2024).",
        "url": "https://universe.roboflow.com/",
        "license": "CC BY 4.0",
        "target_classes_mapped": ["pcb", "cables_wires", "batteries", "mobile_laptops", "displays"],
        "approx_images": 3500,
        "environment_type": "Crowdsourced Electronics & Scrap",
        "hackathon_justification": "Specific e-waste category image density."
    },
    {
        "dataset_name": "Indian Field Kabadi Set (Self-Collected)",
        "citation": "Kabadiwala Project Self-Collected Field Dataset (2026).",
        "url": "Local Field Ingestion",
        "license": "Proprietary Hackathon Project",
        "target_classes_mapped": ALL_CLASSES,
        "approx_images": 800,
        "environment_type": "Real Indian Kabadi Shops / Scrap Piles",
        "hackathon_justification": "100% Unseen evaluation benchmark for Indian domain shift."
    }
]

# ---------------------------------------------------------
# 3. PAN-INDIA SCRAP PRICE DATASETS SPECIFICATIONS (MODEL 2)
# ---------------------------------------------------------
PRICE_DATASETS_META = [
    {
        "source_name": "MSTC Limited e-Auction Records",
        "authority": "Government of India PSU Enterprise",
        "url": "https://www.mstcecommerce.com/",
        "data_type": "Historical Scrap Auction Bids",
        "coverage": "Pan-India PSU & Railway Scrap Sales",
        "justification": "Official government benchmark for bulk scrap metal & equipment."
    },
    {
        "source_name": "Industrial Metals Market Exchange (SteelBaba / BigMint)",
        "authority": "Steel & Scrap Market Intelligence Platforms",
        "url": "https://www.steelbaba.com / https://www.bigmint.co",
        "data_type": "Daily Wholesale Spot Rates",
        "coverage": "Mandi Gobindgarh, Alang, Jalna, Mumbai, Raipur, Chennai",
        "justification": "Real-time industrial foundry scrap spot price indices."
    },
    {
        "source_name": "Regional Kabadi Rate Aggregators (The Kabadiwala / Kabadigo / ScrapBuddy)",
        "authority": "Doorstep Recycling Platforms",
        "url": "https://www.thekabadiwala.com",
        "data_type": "Retail Doorstep Pickup Rates",
        "coverage": "Delhi-NCR, Indore, Bhopal, Jaipur, Ahmedabad, Pune, Bangalore, Hyderabad",
        "justification": "Doorstep retail rates paid directly to Indian consumers."
    }
]


def save_metadata_dossier():
    """Saves dataset metadata files for hackathon defense documentation."""
    vision_meta_path = os.path.join(MANIFEST_DIR, "vision_datasets_metadata.json")
    price_meta_path = os.path.join(MANIFEST_DIR, "price_datasets_metadata.json")
    taxonomy_path = os.path.join(MANIFEST_DIR, "class_taxonomy.json")

    with open(vision_meta_path, "w", encoding="utf-8") as f:
        json.dump(VISION_DATASETS_META, f, indent=4)

    with open(price_meta_path, "w", encoding="utf-8") as f:
        json.dump(PRICE_DATASETS_META, f, indent=4)

    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(TARGET_TAXONOMY, f, indent=4)

    print(f" Saved Vision Dataset Metadata to: {vision_meta_path}")
    print(f" Saved Price Dataset Metadata to: {price_meta_path}")
    print(f" Saved Class Taxonomy to: {taxonomy_path}")


def generate_empty_manifest_template():
    """Generates empty master CSV manifests with exact column schema."""
    vision_manifest_columns = [
        "image_id",
        "filename",
        "dataset_source",
        "original_class",
        "mapped_class",
        "group_id",
        "environment_type",
        "sha256_hash",
        "phash",
        "width",
        "height",
        "channels",
        "is_corrupt",
        "is_duplicate"
    ]
    
    price_manifest_columns = [
        "record_id",
        "date",
        "material_class",
        "state",
        "city",
        "zone",
        "city_tier",
        "market_type",  # wholesale vs retail
        "price_per_kg",
        "currency",
        "source_name"
    ]

    df_vision = pd.DataFrame(columns=vision_manifest_columns)
    df_price = pd.DataFrame(columns=price_manifest_columns)

    vision_csv = os.path.join(MANIFEST_DIR, "master_vision_manifest_template.csv")
    price_csv = os.path.join(MANIFEST_DIR, "master_price_manifest_template.csv")

    df_vision.to_csv(vision_csv, index=False)
    df_price.to_csv(price_csv, index=False)

    print(f" Created Master Vision Manifest Template: {vision_csv}")
    print(f" Created Master Price Manifest Template: {price_csv}")


if __name__ == "__main__":
    print("=" * 70)
    print("KABADIWALA ML PIPELINE: 01_DATASET_RESEARCH & MANIFEST INITIALIZATION")
    print("=" * 70)
    save_metadata_dossier()
    generate_empty_manifest_template()
    print("\n[SUCCESS] Stage 1 Research Infrastructure Initialized Cleanly.")
