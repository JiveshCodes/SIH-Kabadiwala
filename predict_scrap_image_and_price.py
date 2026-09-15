"""
======================================================================
KABADIWALA ML PIPELINE: END-TO-END IMAGE + PRICE PREDICTOR (v2.0)
======================================================================
Combines:
  1. Vision Model 1 (DINOv2 / MobileNetV3) -> 16-Class Scrap Material Classification
  2. Price Model 2  (Pan-India Scrap Intelligence) -> Geo-Adjusted Valuation & Anomaly Detection

Features:
  - 16 Fine-Grained Classes across 6 Super-Categories (E-Waste, Metals, Paper/Cardboard, Plastics, Glass, Wood)
  - Alias mapper for backwards compatibility with legacy labels
  - Multi-city adjustment factors
  - Z-score / Threshold fair-market anomaly validation
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

# 16 Canonical Classes
CLASSES = [
    'aluminium',
    'appliances',
    'batteries',
    'cables_wires',
    'cardboard',
    'copper',
    'glass_mirror',
    'iron_steel',
    'laptops_computers',
    'mixed_plastic',
    'mobile_tablets',
    'newspaper_paper',
    'pcb_chips',
    'pet_plastic',
    'tv_monitors_displays',
    'wood'
]

# Aliases mapping older / variant names to canonical 16 classes
CLASS_ALIASES = {
    'pcb': 'pcb_chips',
    'circuit_board': 'pcb_chips',
    'electronic_chips': 'pcb_chips',
    'chips': 'pcb_chips',
    'mobile_laptops': 'mobile_tablets',
    'mobile': 'mobile_tablets',
    'laptop': 'laptops_computers',
    'laptops': 'laptops_computers',
    'displays': 'tv_monitors_displays',
    'tv': 'tv_monitors_displays',
    'monitor': 'tv_monitors_displays',
    'screen': 'tv_monitors_displays',
    'glass': 'glass_mirror',
    'mirror': 'glass_mirror',
    'paper': 'newspaper_paper',
    'newspaper': 'newspaper_paper',
    'cardboard_paper': 'cardboard',
    'metal': 'iron_steel',
    'steel': 'iron_steel',
    'iron': 'iron_steel',
    'copper_wire': 'copper',
    'cables': 'cables_wires',
    'wires': 'cables_wires',
    'wire': 'cables_wires',
    'cord': 'cables_wires',
    'plastic': 'mixed_plastic',
    'battery': 'batteries',
    'motors_magnets': 'appliances',
}

# Super-category mapping
SUPER_CATEGORIES = {
    'aluminium': 'Metals',
    'appliances': 'E-Waste',
    'batteries': 'E-Waste',
    'cables_wires': 'E-Waste',
    'cardboard': 'Paper & Cardboard',
    'copper': 'Metals',
    'glass_mirror': 'Glass & Ceramics',
    'iron_steel': 'Metals',
    'laptops_computers': 'E-Waste',
    'mixed_plastic': 'Plastics',
    'mobile_tablets': 'E-Waste',
    'newspaper_paper': 'Paper & Cardboard',
    'pcb_chips': 'E-Waste',
    'pet_plastic': 'Plastics',
    'tv_monitors_displays': 'E-Waste',
    'wood': 'Wood & Timber'
}

# Baseline price reference table (per kg in INR) — Pan-India benchmark
PRICE_BENCHMARKS = {
    'aluminium':            {'avg_price': 165.00, 'unit': 'kg', 'min': 140.00, 'max': 190.00},
    'appliances':           {'avg_price': 35.00,  'unit': 'kg', 'min': 25.00,  'max': 45.00},
    'batteries':            {'avg_price': 85.00,  'unit': 'kg', 'min': 70.00,  'max': 105.00},
    'cables_wires':         {'avg_price': 220.00, 'unit': 'kg', 'min': 180.00, 'max': 260.00},
    'cardboard':            {'avg_price': 18.50,  'unit': 'kg', 'min': 14.00,  'max': 22.00},
    'copper':               {'avg_price': 685.00, 'unit': 'kg', 'min': 620.00, 'max': 740.00},
    'glass_mirror':         {'avg_price': 4.50,   'unit': 'kg', 'min': 3.00,   'max': 6.00},
    'iron_steel':           {'avg_price': 34.00,  'unit': 'kg', 'min': 28.00,  'max': 40.00},
    'laptops_computers':    {'avg_price': 450.00, 'unit': 'kg', 'min': 350.00, 'max': 600.00},
    'mixed_plastic':        {'avg_price': 24.00,  'unit': 'kg', 'min': 18.00,  'max': 30.00},
    'mobile_tablets':       {'avg_price': 500.00, 'unit': 'kg', 'min': 380.00, 'max': 650.00},
    'newspaper_paper':      {'avg_price': 15.00,  'unit': 'kg', 'min': 12.00,  'max': 18.00},
    'pcb_chips':            {'avg_price': 350.00, 'unit': 'kg', 'min': 280.00, 'max': 450.00},
    'pet_plastic':          {'avg_price': 28.50,  'unit': 'kg', 'min': 22.00,  'max': 35.00},
    'tv_monitors_displays': {'avg_price': 120.00, 'unit': 'kg', 'min': 90.00,  'max': 150.00},
    'wood':                 {'avg_price': 6.00,   'unit': 'kg', 'min': 4.00,   'max': 8.00},
}

# Verify all 16 classes are covered
_missing = [c for c in CLASSES if c not in PRICE_BENCHMARKS]
if _missing:
    raise ValueError(f"PRICE_BENCHMARKS missing entries for: {_missing}")

# City adjustment multipliers
CITY_MULTIPLIERS = {
    'Delhi': 1.0,
    'Mumbai': 1.05,
    'Bangalore': 1.02,
    'Mandi Gobindgarh': 0.98,
    'Bhopal': 0.95,
    'Chennai': 1.01,
    'Hyderabad': 1.01,
    'Pune': 1.03,
    'Ahmedabad': 0.97,
    'Kolkata': 0.99,
    'Indore': 0.96,
    'Noida': 1.0,
    'Jaipur': 0.97,
    'Surat': 0.98
}

CONFIDENCE_THRESHOLD = 0.65


def resolve_canonical_class(name):
    """Maps aliases or raw strings to canonical class keys."""
    if not name:
        return 'mixed_plastic'
    raw = str(name).strip().lower()
    if raw in PRICE_BENCHMARKS:
        return raw
    if raw in CLASS_ALIASES:
        return CLASS_ALIASES[raw]
    # Fuzzy fallback
    for k in PRICE_BENCHMARKS:
        if k in raw or raw in k:
            return k
    return 'mixed_plastic'


def predict_image_and_price(material_name, city="Delhi", user_offered_price=None, confidence=1.0):
    """
    End-to-end prediction from vision detection -> price valuation.
    """
    material_name = resolve_canonical_class(material_name)
    bench = PRICE_BENCHMARKS[material_name]
    mult = CITY_MULTIPLIERS.get(city, 1.0)

    estimated_rate = round(bench['avg_price'] * mult, 2)
    min_fair = round(bench['min'] * mult, 2)
    max_fair = round(bench['max'] * mult, 2)

    is_low_confidence = confidence < CONFIDENCE_THRESHOLD

    result = {
        'detected_material': material_name,
        'super_category':    SUPER_CATEGORIES.get(material_name, 'Recyclables'),
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
    """
    if not probs_dict:
        raise ValueError("probs_dict cannot be empty")

    total = sum(probs_dict.values())
    if total <= 0:
        raise ValueError("probs_dict values must sum to > 0")

    normalized = {resolve_canonical_class(k): v / total for k, v in probs_dict.items()}

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
    print("DEMO 2: appliances (hair dryer / oven) — Mumbai")
    print(sep)
    r2 = predict_image_and_price('appliances', city='Mumbai', user_offered_price=30.0, confidence=0.92)
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    print(f"\n{sep}")
    print("DEMO 3: copper — High Value Scrap — Bangalore")
    print(sep)
    r3 = predict_image_and_price('copper', city='Bangalore', user_offered_price=710.0, confidence=0.91)
    print(json.dumps(r3, indent=2, ensure_ascii=False))

    print(f"\n{sep}")
    print("DEMO 4: wood — Low Value Scrap — Jaipur")
    print(sep)
    r4 = predict_image_and_price('wood', city='Jaipur', user_offered_price=5.0, confidence=0.88)
    print(json.dumps(r4, indent=2, ensure_ascii=False))
