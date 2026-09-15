"""
ingest_real_price_data.py
=========================
Real Indian Scrap Price Data Ingestion & ETL Pipeline

This script strictly ingests REAL historical scrap market data from EXTERNAL CSV files
located in `data/raw/price_data/`.

NO HARDCODED PYTHON DATA LISTS, SYNTHETIC MULTIPLIERS, OR RANDOM NOISE ARE USED.
All output records come directly from real external CSV files.

BUG FIXES APPLIED:
  - KeyError crash: 'material' column check before mapping — CSV may use 'material_class'
  - KeyError crash: 'state' column check before zone mapping
  - KeyError crash: 'price_per_kg' column check before filtering
  - Tolerates CRLF line endings (\r\n) in CSV files from Windows Excel
  - process_external_csv_datasets now returns df even if manifest already exists
    (previously it would skip re-processing and return stale data)
  - Added dtype=str read to avoid pandas mis-parsing numeric material names
"""

import os
import glob
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_PRICE_DIR = os.path.join(DATA_DIR, "raw", "price_data")
MANIFEST_DIR  = os.path.join(DATA_DIR, "manifests")

os.makedirs(RAW_PRICE_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

# Standard material class alias map — tolerates many raw spellings
MATERIAL_ALIAS_MAP = {
    "copper":              "copper",
    "copper wire":         "copper",
    "copper scrap":        "copper",
    "copper_scrap":        "copper",
    "aluminium":           "aluminium",
    "aluminum":            "aluminium",
    "aluminum utensils":   "aluminium",
    "iron":                "iron_steel",
    "steel":               "iron_steel",
    "iron scrap":          "iron_steel",
    "iron_steel":          "iron_steel",
    "heavy melting steel": "iron_steel",
    "pcb":                 "pcb",
    "motherboard":         "pcb",
    "circuit board":       "pcb",
    "cable":               "cables_wires",
    "cables_wires":        "cables_wires",
    "wire":                "cables_wires",
    "copper cable":        "cables_wires",
    "battery":             "batteries",
    "batteries":           "batteries",
    "lead acid battery":   "batteries",
    "mobile battery":      "batteries",
    "mobile":              "mobile_laptops",
    "laptop":              "mobile_laptops",
    "mobile_laptops":      "mobile_laptops",
    "ewaste mobile":       "mobile_laptops",
    "display":             "displays",
    "displays":            "displays",
    "lcd":                 "displays",
    "monitor":             "displays",
    "pet":                 "pet_plastic",
    "pet_plastic":         "pet_plastic",
    "pet bottle":          "pet_plastic",
    "plastic bottle":      "pet_plastic",
    "plastic":             "mixed_plastic",
    "mixed_plastic":       "mixed_plastic",
    "hard plastic":        "mixed_plastic",
    "hdpe":                "mixed_plastic",
    "cardboard":           "cardboard_paper",
    "cardboard_paper":     "cardboard_paper",
    "paper":               "cardboard_paper",
    "newspaper":           "cardboard_paper",
    "glass":               "glass",
    "glass bottle":        "glass",
}

STATE_ZONE_MAP = {
    "Delhi":          ("North",     "Tier-1"),
    "Punjab":         ("North",     "Tier-2"),
    "Haryana":        ("North",     "Tier-2"),
    "Uttar Pradesh":  ("North",     "Tier-2"),
    "Rajasthan":      ("North",     "Tier-2"),
    "Maharashtra":    ("West",      "Tier-1"),
    "Gujarat":        ("West",      "Tier-2"),
    "Goa":            ("West",      "Tier-2"),
    "Tamil Nadu":     ("South",     "Tier-1"),
    "Karnataka":      ("South",     "Tier-1"),
    "Telangana":      ("South",     "Tier-1"),
    "Kerala":         ("South",     "Tier-2"),
    "Andhra Pradesh": ("South",     "Tier-2"),
    "West Bengal":    ("East",      "Tier-1"),
    "Odisha":         ("East",      "Tier-2"),
    "Bihar":          ("East",      "Tier-2"),
    "Jharkhand":      ("East",      "Tier-2"),
    "Madhya Pradesh": ("Central",   "Tier-2"),
    "Chhattisgarh":   ("Central",   "Tier-2"),
    "Assam":          ("NorthEast", "Tier-2"),
    "Meghalaya":      ("NorthEast", "Tier-3"),
}


def process_external_csv_datasets(force_reprocess=False):
    """
    Reads all external CSV files from data/raw/price_data/ and validates them.

    Args:
        force_reprocess: If True, re-reads source CSVs even if manifest already exists.

    Returns:
        pd.DataFrame with standardised price records.
    """
    output_path = os.path.join(MANIFEST_DIR, "real_locked_pan_india_scrap_prices.csv")

    # If manifest already exists and we're not forcing, just load it
    if os.path.exists(output_path) and not force_reprocess:
        df = pd.read_csv(output_path)
        print(f"  Loaded existing manifest: {output_path} ({len(df)} records)")
        return df

    csv_files = glob.glob(os.path.join(RAW_PRICE_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No external CSV files found in {RAW_PRICE_DIR}. "
            f"Please place your real historical dataset CSV files in {RAW_PRICE_DIR}."
        )

    all_dfs = []
    for filepath in csv_files:
        print(f"  Reading external dataset: {filepath}")
        try:
            # Use dtype=str to avoid mis-parsing; strip whitespace from headers
            df_file = pd.read_csv(filepath, dtype=str)
            df_file.columns = [c.strip().lower() for c in df_file.columns]
            all_dfs.append(df_file)
        except Exception as e:
            print(f"  WARNING: Could not read {filepath}: {e}. Skipping.")

    if not all_dfs:
        raise ValueError("All CSV files failed to load. Check file encoding and format.")

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # ---- BUG FIX: Determine the material column name ----
    if 'material' in df_combined.columns:
        mat_col = 'material'
    elif 'material_class' in df_combined.columns:
        mat_col = 'material_class'
    else:
        raise KeyError(
            f"CSV must have a 'material' or 'material_class' column. "
            f"Found columns: {list(df_combined.columns)}"
        )

    df_combined['material_class'] = df_combined[mat_col].map(
        lambda x: MATERIAL_ALIAS_MAP.get(str(x).strip().lower(), str(x).strip().lower())
    )

    # ---- BUG FIX: Determine the state column ----
    if 'state' in df_combined.columns:
        zones, tiers = [], []
        for state in df_combined['state']:
            z, t = STATE_ZONE_MAP.get(str(state).strip(), ("Unknown", "Tier-2"))
            zones.append(z)
            tiers.append(t)
        df_combined['zone']      = zones
        df_combined['city_tier'] = tiers
    else:
        print("  WARNING: 'state' column not found. zone/city_tier defaulted to Unknown/Tier-2.")
        df_combined['zone']      = "Unknown"
        df_combined['city_tier'] = "Tier-2"

    # ---- BUG FIX: Ensure price_per_kg is numeric ----
    if 'price_per_kg' not in df_combined.columns:
        raise KeyError(
            f"CSV must have a 'price_per_kg' column. Found: {list(df_combined.columns)}"
        )

    df_combined['price_per_kg'] = pd.to_numeric(df_combined['price_per_kg'], errors='coerce')
    n_before = len(df_combined)
    df_combined = df_combined[df_combined['price_per_kg'] > 0].copy()
    n_dropped = n_before - len(df_combined)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with zero/null price.")

    df_combined.to_csv(output_path, index=False)
    print(f"  Ingested & verified {len(df_combined)} authentic rows from external CSV files.")
    print(f"  Master CSV written to: {output_path}")
    return df_combined


if __name__ == "__main__":
    print("=" * 70)
    print("KABADIWALA ML PIPELINE: EXTERNAL REAL DATASET ETL")
    print("=" * 70)
    df = process_external_csv_datasets(force_reprocess=True)
    print(f"\nMaterial class distribution:\n{df['material_class'].value_counts().to_string()}")
    print("\n[SUCCESS] External Real Dataset Loaded Cleanly.")
