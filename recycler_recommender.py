"""
======================================================================
KABADIWALA ML PIPELINE: AI AUTHORIZED RECYCLER MATCHING & RANKING ENGINE
SIH Problem Statement #26229 Compliance
======================================================================
Ranks authorized recyclers for informal scrap collectors based on:
  1. CPCB/SPCB Authorization Status (35% weight)
  2. Offered Buying Rate vs Market Benchmark (30% weight)
  3. Proximity / GPS Distance in km (20% weight)
  4. Pickup Availability & Service Rating (15% weight)
"""

import math

# Sample Verified Authorized Recycler Database (Pan-India CPCB/SPCB Registered)
AUTHORIZED_RECYCLERS = [
    {
        "recycler_id": "REC_DEL_001",
        "name": "EcoRecycle India Authorized E-Waste Plant",
        "city": "Delhi",
        "lat": 28.6139, "lon": 77.2090,
        "cpcb_reg_no": "CPCB/E-WASTE/REG/DL/2023/1042",
        "authorization_status": "VALID",
        "accepted_materials": ["pcb", "cables_wires", "batteries", "mobile_laptops", "displays", "motors_magnets"],
        "offered_rate_multiplier": 1.08,  # Pays 8% above benchmark
        "pickup_available": True,
        "min_pickup_weight_kg": 20,
        "rating": 4.8,
        "contact_phone": "+91-9876543210"
    },
    {
        "recycler_id": "REC_BOM_002",
        "name": "Attero Recycling Center (Maharashtra Hub)",
        "city": "Mumbai",
        "lat": 19.0760, "lon": 72.8777,
        "cpcb_reg_no": "SPCB/MH/EPR/REC/8891",
        "authorization_status": "VALID",
        "accepted_materials": ["pcb", "cables_wires", "batteries", "mobile_laptops", "displays", "copper", "aluminium"],
        "offered_rate_multiplier": 1.10,  # Pays 10% above benchmark
        "pickup_available": True,
        "min_pickup_weight_kg": 15,
        "rating": 4.9,
        "contact_phone": "+91-9812345678"
    },
    {
        "recycler_id": "REC_BLR_003",
        "name": "GreenTek Remanufacturers & Metals",
        "city": "Bangalore",
        "lat": 12.9716, "lon": 77.5946,
        "cpcb_reg_no": "KSPCB/E-WASTE/AUTHORISED/2024/044",
        "authorization_status": "VALID",
        "accepted_materials": ["pcb", "cables_wires", "copper", "aluminium", "iron_steel", "motors_magnets"],
        "offered_rate_multiplier": 1.05,
        "pickup_available": True,
        "min_pickup_weight_kg": 10,
        "rating": 4.7,
        "contact_phone": "+91-9765432109"
    },
    {
        "recycler_id": "REC_DEL_004",
        "name": "Mayapuri Authorized Metal & E-Scrap Hub",
        "city": "Delhi",
        "lat": 28.6280, "lon": 77.1120,
        "cpcb_reg_no": "DPCC/W/RECYCLER/2022/990",
        "authorization_status": "VALID",
        "accepted_materials": ["iron_steel", "copper", "aluminium", "motors_magnets", "cables_wires"],
        "offered_rate_multiplier": 1.03,
        "pickup_available": False,
        "min_pickup_weight_kg": 0,
        "rating": 4.4,
        "contact_phone": "+91-9988776655"
    },
    {
        "recycler_id": "REC_NOIDA_005",
        "name": "Noida Circular Plastics & Glass Recyclers",
        "city": "Noida",
        "lat": 28.5355, "lon": 77.3910,
        "cpcb_reg_no": "UPPCB/PLASTIC-EPR/REC/2023/112",
        "authorization_status": "VALID",
        "accepted_materials": ["pet_plastic", "mixed_plastic", "cardboard_paper", "glass"],
        "offered_rate_multiplier": 1.04,
        "pickup_available": True,
        "min_pickup_weight_kg": 25,
        "rating": 4.6,
        "contact_phone": "+91-9811223344"
    }
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in kilometers between two GPS coordinates."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def rank_authorized_recyclers(lot_items, collector_lat=28.6139, collector_lon=77.2090, collector_city="Delhi"):
    """
    Ranks authorized recyclers for a collector's lot based on AI multi-criteria scoring.

    Args:
        lot_items: list of dicts e.g. [{'material': 'pcb', 'weight_kg': 15.0, 'estimated_rate': 350.0}]
        collector_lat: latitude float
        collector_lon: longitude float
        collector_city: string

    Returns:
        Sorted list of ranked recyclers with score details and total payout
    """
    if not lot_items:
        return []

    lot_materials = set(item['material'] for item in lot_items)
    total_lot_weight = sum(item.get('weight_kg', 1.0) for item in lot_items)
    base_lot_value = sum(item.get('weight_kg', 1.0) * item.get('estimated_rate', 50.0) for item in lot_items)

    ranked_results = []

    for rec in AUTHORIZED_RECYCLERS:
        accepted_set = set(rec['accepted_materials'])
        matching_materials = lot_materials.intersection(accepted_set)

        if not matching_materials:
            continue  # Recycler does not accept any item in this lot

        # 1. Authorization Score (35%)
        # Full score if accepts ALL hazardous materials in lot
        mat_coverage_ratio = len(matching_materials) / len(lot_materials)
        auth_score = (1.0 if rec['authorization_status'] == 'VALID' else 0.0) * mat_coverage_ratio * 35.0

        # 2. Price Score (30%)
        # Based on rate multiplier offered over baseline
        price_score = min(30.0, (rec['offered_rate_multiplier'] / 1.10) * 30.0)

        # 3. Proximity Score (20%)
        dist_km = haversine_distance(collector_lat, collector_lon, rec['lat'], rec['lon'])
        # If city matches, cap distance penalty
        if rec['city'].lower() == collector_city.lower() and dist_km > 50:
            dist_km = 12.0
        prox_score = max(0.0, 20.0 * (1.0 - (dist_km / 100.0)))

        # 4. Service Score (15%)
        pickup_bonus = 7.5 if (rec['pickup_available'] and total_lot_weight >= rec['min_pickup_weight_kg']) else 2.0
        rating_pts = (rec['rating'] / 5.0) * 7.5
        service_score = pickup_bonus + rating_pts

        total_score = round(auth_score + price_score + prox_score + service_score, 1)
        estimated_payout = round(base_lot_value * rec['offered_rate_multiplier'], 2)

        ranked_results.append({
            "recycler_id": rec['recycler_id'],
            "name": rec['name'],
            "cpcb_reg_no": rec['cpcb_reg_no'],
            "authorization_status": rec['authorization_status'],
            "match_score": total_score,
            "distance_km": dist_km,
            "pickup_available": rec['pickup_available'],
            "estimated_payout_inr": f"₹{estimated_payout}",
            "offered_rate_boost": f"+{round((rec['offered_rate_multiplier'] - 1.0) * 100, 1)}%",
            "contact_phone": rec['contact_phone'],
            "accepted_lot_materials": list(matching_materials)
        })

    # Sort descending by match score
    ranked_results.sort(key=lambda x: x['match_score'], reverse=True)
    return ranked_results


if __name__ == "__main__":
    # Test Recycler Recommendation Engine
    sample_lot = [
        {"material": "pcb", "weight_kg": 10.0, "estimated_rate": 350.0},
        {"material": "cables_wires", "weight_kg": 25.0, "estimated_rate": 220.0},
        {"material": "displays", "weight_kg": 15.0, "estimated_rate": 120.0}
    ]

    recommendations = rank_authorized_recyclers(sample_lot, collector_lat=28.6139, collector_lon=77.2090, collector_city="Delhi")
    print(f"Top Recommended Recyclers for E-Waste Lot (Total Weight: 50 kg):")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec['name']} | Score: {rec['match_score']}/100 | Dist: {rec['distance_km']} km | Payout: {rec['estimated_payout_inr']}")
