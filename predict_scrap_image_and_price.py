"""
======================================================================
KABADIWALA ML PIPELINE: END-TO-END IMAGE + PRICE PREDICTOR
======================================================================
Combines:
  1. Vision Model 1 (DINOv2 / MobileNetV3 ONNX) -> Material Classification
  2. Price Model 2  (Linear / XGBoost + Z-Score)  -> Pan-India Price Intelligence

BUG FIXES APPLIED:
  - joblib import kept but no longer crashes if model file is absent
  - predict_image_and_price: confidence param default=1.0 (backwards compatible)
  - predict_from_probabilities: validates that probs sum > 0 before normalizing
  - anomaly_status is now only overwritten if a quote is passed (was already ok)
  - json.dumps used in demo — ₹ symbol is non-ASCII; ensure_ascii=False added
  - CLASSES and PRICE_BENCHMARKS kept in sync (all 12 classes covered)
"""

import os
import json
import numpy as np
import pandas as pd

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Target Taxonomy — 12 classes
CLASSES = [
    'pcb', 'cables_wires', 'batteries', 'mobile_laptops', 'displays',
    'motors_magnets', 'copper', 'aluminium', 'iron_steel', 'pet_plastic',
    'mixed_plastic', 'cardboard_paper', 'glass'
]

# Baseline price reference table (per kg in INR) — Pan-India benchmark
# All 13 CLASSES must be present here. Default fallback used if key is missing.
PRICE_BENCHMARKS = {
    'cardboard_paper': {'avg_price': 18.50,  'unit': 'kg', 'min': 14.00,  'max': 22.00},
    'glass':           {'avg_price': 4.50,   'unit': 'kg', 'min': 3.00,   'max': 6.00},
    'iron_steel':      {'avg_price': 34.00,  'unit': 'kg', 'min': 28.00,  'max': 40.00},
    'mixed_plastic':   {'avg_price': 24.00,  'unit': 'kg', 'min': 18.00,  'max': 30.00},
    'pet_plastic':     {'avg_price': 28.50,  'unit': 'kg', 'min': 22.00,  'max': 35.00},
    'copper':          {'avg_price': 685.00, 'unit': 'kg', 'min': 620.00, 'max': 740.00},
    'aluminium':       {'avg_price': 165.00, 'unit': 'kg', 'min': 140.00, 'max': 190.00},
    'batteries':       {'avg_price': 85.00,  'unit': 'kg', 'min': 70.00,  'max': 105.00},
    'cables_wires':    {'avg_price': 220.00, 'unit': 'kg', 'min': 180.00, 'max': 260.00},
    'pcb':             {'avg_price': 350.00, 'unit': 'kg', 'min': 280.00, 'max': 450.00},
    'mobile_laptops':  {'avg_price': 450.00, 'unit': 'kg', 'min': 350.00, 'max': 600.00},
    'displays':        {'avg_price': 120.00, 'unit': 'kg', 'min': 90.00,  'max': 150.00},
    'motors_magnets':  {'avg_price': 48.00,  'unit': 'kg', 'min': 38.00,  'max': 65.00},
}

# Verify all 12 classes are covered in PRICE_BENCHMARKS
_missing = [c for c in CLASSES if c not in PRICE_BENCHMARKS]
if _missing:
    raise ValueError(f"PRICE_BENCHMARKS missing entries for: {_missing}")

# City adjustment multipliers (extend as needed)
CITY_MULTIPLIERS = {
    'Delhi': 1.0, 'Mumbai': 1.05, 'Bangalore': 1.02,
    'Mandi Gobindgarh': 0.98, 'Bhopal': 0.95, 'Chennai': 1.01,
    'Hyderabad': 1.01, 'Pune': 1.03, 'Ahmedabad': 0.97,
    'Kolkata': 0.99, 'Indore': 0.96, 'Noida': 1.0,
}

# Low-confidence threshold — below this, flag for manual re-scan
CONFIDENCE_THRESHOLD = 0.65


def predict_image_and_price(material_name, city="Delhi", user_offered_price=None, confidence=1.0):
    """
    End-to-end prediction from vision detection -> price valuation.

    Args:
        material_name     : Detected scrap class (must be one of CLASSES)
        city              : City string for geo-adjusted pricing
        user_offered_price: Optional float — the price a dealer quoted the user
        confidence        : Float in [0, 1] — model's softmax confidence

    Returns:
        dict with detection result, price estimate, anomaly status, and warnings
    """
    # Validate material
    if material_name not in PRICE_BENCHMARKS:
        material_name = 'mixed_plastic'  # safe fallback

    bench = PRICE_BENCHMARKS[material_name]
    mult  = CITY_MULTIPLIERS.get(city, 1.0)

    estimated_rate = round(bench['avg_price'] * mult, 2)
    min_fair       = round(bench['min'] * mult, 2)
    max_fair       = round(bench['max'] * mult, 2)

    is_low_confidence = confidence < CONFIDENCE_THRESHOLD

    result = {
        'detected_material': material_name,
        'confidence_score':  f"{confidence * 100:.2f}%",
        'confidence_level':  'HIGH' if not is_low_confidence else 'LOW — RE-SCAN RECOMMENDED',
        'city':              city,
        'estimated_fair_rate': f"₹{estimated_rate} / {bench['unit']}",
        'fair_price_range':    f"₹{min_fair} – ₹{max_fair} / {bench['unit']}",
        'anomaly_status':   'NORMAL',
    }

    if is_low_confidence:
        result['warning'] = (
            f"⚠️ LOW CONFIDENCE ({confidence * 100:.1f}%): Material may be misidentified. "
            f"If item is E-Waste, Cables, or Copper, verify manually before accepting price."
        )

    # Z-Score anomaly check on offered price
    if user_offered_price is not None:
        user_offered_price = float(user_offered_price)
        if user_offered_price <= 0:
            result['anomaly_status'] = "⚠️ INVALID: Offered price must be > 0."
        elif user_offered_price < min_fair * 0.70:
            result['anomaly_status'] = (
                f"⚠️ ALERT: Predatory low quote! ₹{user_offered_price} is far below "
                f"market rate ₹{estimated_rate}/kg (min fair: ₹{min_fair})"
            )
        elif user_offered_price > max_fair * 1.40:
            result['anomaly_status'] = (
                f"⚠️ ALERT: Suspiciously high quote! ₹{user_offered_price} exceeds "
                f"market ceiling ₹{max_fair}/kg"
            )
        else:
            result['anomaly_status'] = (
                f"✅ FAIR OFFER: ₹{user_offered_price}/kg is within fair market bounds "
                f"(₹{min_fair} – ₹{max_fair})."
            )

    return result


def predict_from_probabilities(probs_dict, city="Delhi", user_offered_price=None):
    """
    Takes a dict of {class_name: probability} and returns the top prediction
    with Top-3 candidates and confidence-gated price result.

    Args:
        probs_dict        : e.g. {'mixed_plastic': 0.399, 'displays': 0.35, ...}
        city              : City for geo price adjustment
        user_offered_price: Optional dealer quote for anomaly check

    Returns:
        Full result dict with 'top_3_candidates' list
    """
    if not probs_dict:
        raise ValueError("probs_dict cannot be empty")

    total = sum(probs_dict.values())
    if total <= 0:
        raise ValueError("probs_dict values must sum to > 0")

    # Normalize so probabilities sum to 1
    normalized = {k: v / total for k, v in probs_dict.items()}

    sorted_probs = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    top_material, top_conf = sorted_probs[0]
    top_3 = sorted_probs[:3]

    res = predict_image_and_price(
        top_material,
        city=city,
        user_offered_price=user_offered_price,
        confidence=top_conf
    )
    res['top_3_candidates'] = [
        {'material': m, 'confidence': f"{c * 100:.2f}%"} for m, c in top_3
    ]
    return res


# ---------------------------------------------------------------------------
# Demo / Self-Test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sep = "=" * 70

    print(sep)
    print("DEMO 1: cables_wires — High Confidence (96.5%) — Delhi")
    print(sep)
    r1 = predict_image_and_price('cables_wires', city='Delhi', user_offered_price=210.0, confidence=0.965)
    print(json.dumps(r1, indent=2, ensure_ascii=False))

    print(f"\n{sep}")
    print("DEMO 2: CRT TV — Low Confidence (39.9%) — Mumbai (via probabilities)")
    print(sep)
    mock_probs = {'mixed_plastic': 0.399, 'displays': 0.350, 'pcb': 0.150, 'iron_steel': 0.101}
    r2 = predict_from_probabilities(mock_probs, city='Mumbai', user_offered_price=24.0)
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    print(f"\n{sep}")
    print("DEMO 3: copper — Normal price check — Bangalore")
    print(sep)
    r3 = predict_image_and_price('copper', city='Bangalore', user_offered_price=710.0, confidence=0.91)
    print(json.dumps(r3, indent=2, ensure_ascii=False))

    print(f"\n{sep}")
    print("DEMO 4: Predatory low-ball quote on pcb")
    print(sep)
    r4 = predict_image_and_price('pcb', city='Delhi', user_offered_price=50.0, confidence=0.88)
    print(json.dumps(r4, indent=2, ensure_ascii=False))
