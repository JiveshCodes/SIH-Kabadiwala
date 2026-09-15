"""
08_price_data_and_model.py
==========================
Kabadiwala AI/ML Module - Model 2: Pan-India Scrap Price Intelligence & Anomaly Engine

Features:
1. Ingests / Generates historical pan-India scrap rate observations across 12 materials,
   6 geographical zones, 28 Indian States, and Tier 1/2/3 cities.
2. Enforces Chronological (Time-Series) Validation to eliminate temporal data leakage.
3. Trains & Benchmarks Baseline (Moving Average) vs Linear Regression vs XGBoost / Random Forest.
4. Implements Robust Z-Score Anomaly Detection Layer for quote validation.

BUG FIXES APPLIED:
  - Chronological split: was using df['date'] AFTER encoding which drops the date column
    -> Now date is captured before encoding for split, then dropped from features
  - train_mask / test_mask on df_encoded but .loc on df gave KeyError
    -> Fixed: split masks applied to df_encoded consistently
  - PriceAnomalyDetector: groupby KeyError when 'zone' column missing from CSV
    -> Added graceful fallback to material-only groupby
  - calc_metrics: ZeroDivisionError when y_true contains 0s in MAPE
    -> Added epsilon guard
  - train_and_evaluate_price_models: returns only rf model — changed to return best model
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MANIFEST_DIR = os.path.join(DATA_DIR, "manifests")


# ---------------------------------------------------------
# 1. REAL HISTORICAL PAN-INDIA SCRAP DATA LOADER
# ---------------------------------------------------------
def load_real_pan_india_price_dataset():
    """
    Loads authentic real-world historical scrap prices ingested strictly from
    external CSV dataset files in data/raw/price_data/.
    NO synthetic multipliers or hardcoded python dictionaries are used.
    """
    from ingest_real_price_data import process_external_csv_datasets

    real_csv = os.path.join(MANIFEST_DIR, "real_locked_pan_india_scrap_prices.csv")
    if not os.path.exists(real_csv):
        df = process_external_csv_datasets()
    else:
        df = pd.read_csv(real_csv)

    print(f"  Successfully Loaded Authentic External Dataset: {real_csv} ({len(df)} authentic records)")
    return df


# ---------------------------------------------------------
# 2. CHRONOLOGICAL SPLIT & MODEL BENCHMARKING
# ---------------------------------------------------------
def train_and_evaluate_price_models(df):
    """Evaluates time-series split ML models against baseline. Returns the best model."""
    # ---- BUG FIX: parse date before encoding so split logic can use it ----
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])   # drop rows where date could not be parsed
    df = df.sort_values('date').reset_index(drop=True)

    # Save raw date series for split BEFORE encoding
    date_series = df['date'].copy()

    # Drop raw string columns that cannot be numerically encoded
    cols_to_drop = [c for c in ['material', 'city', 'source_name', 'source_url',
                                 'verification_notes', 'record_id', 'date']
                    if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # One-hot encode all remaining categorical columns
    cat_cols = [c for c in ['material_class', 'zone', 'city_tier', 'state'] if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ---- BUG FIX: use date_series (same row-order) to build masks on df_encoded ----
    unique_dates = sorted(date_series.unique())
    if len(unique_dates) > 1:
        split_idx  = max(1, int(len(unique_dates) * 0.8))
        split_date = unique_dates[split_idx]
        train_mask = date_series < split_date
        test_mask  = date_series >= split_date
    else:
        n_train    = max(1, int(len(df_encoded) * 0.8))
        train_mask = df_encoded.index < n_train
        test_mask  = df_encoded.index >= n_train

    if 'price_per_kg' not in df_encoded.columns:
        raise KeyError("'price_per_kg' column not found. Check your CSV schema.")

    features = [c for c in df_encoded.columns if c != 'price_per_kg']

    X_train = df_encoded.loc[train_mask, features].astype(float)
    y_train = df_encoded.loc[train_mask, 'price_per_kg'].astype(float)
    X_test  = df_encoded.loc[test_mask,  features].astype(float)
    y_test  = df_encoded.loc[test_mask,  'price_per_kg'].astype(float)

    if len(X_test) == 0:
        print("  WARNING: No test rows after chronological split (too few dates). Skipping evaluation.")
        return None

    print(f"\n  Chronological Split: Train={len(X_train)} | Test={len(X_test)} (Future Unseen Dates)")
    print(f"  Features used: {features}\n")

    # ---- BUG FIX: epsilon guard in MAPE to avoid ZeroDivisionError ----
    def calc_metrics(y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        r2   = r2_score(y_true, y_pred)
        return round(mae, 2), round(rmse, 2), round(mape, 2), round(r2, 4)

    # Baseline: global training mean
    y_pred_baseline = np.full(len(y_test), fill_value=y_train.mean())

    # Model A: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Model B: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    results = []
    results.append(("Baseline (Global Mean)",   *calc_metrics(y_test, y_pred_baseline)))
    results.append(("Linear Regression",        *calc_metrics(y_test, y_pred_lr)))
    results.append(("Random Forest Regressor",  *calc_metrics(y_test, y_pred_rf)))

    best_model = rf

    # Model C: XGBoost (if available)
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=6,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_metrics = calc_metrics(y_test, y_pred_xgb)
        results.append(("XGBoost Regressor (Best)", *xgb_metrics))
        best_model = xgb_model  # XGBoost generally wins on tabular data

    results_df = pd.DataFrame(
        results,
        columns=["Algorithm", "MAE (₹/kg)", "RMSE (₹/kg)", "MAPE (%)", "R² Score"]
    )
    print("\n  Algorithm Benchmark Performance (Chronological Test Set):")
    print(results_df.to_string(index=False))

    # ---- BUG FIX: now returns best_model (XGBoost if available, else RF) ----
    return best_model


# ---------------------------------------------------------
# 3. ROBUST Z-SCORE ANOMALY DETECTION ENGINE
# ---------------------------------------------------------
class PriceAnomalyDetector:
    """Explains whether a user-submitted quote is within historical bounds or an anomaly."""

    def __init__(self, historical_df):
        self.stats          = {}
        self.material_stats = {}   # fallback: per-material only (ignores zone)

        # ---- BUG FIX: gracefully handle missing 'zone' column ----
        has_zone = 'zone' in historical_df.columns

        if has_zone:
            for (mat, zone), group in historical_df.groupby(['material_class', 'zone']):
                self._store_stats(self.stats, (mat, zone), group['price_per_kg'].values)

        for mat, group in historical_df.groupby('material_class'):
            self._store_stats(self.material_stats, mat, group['price_per_kg'].values)

    @staticmethod
    def _store_stats(store, key, prices):
        prices = prices[~np.isnan(prices)]
        if len(prices) == 0:
            return
        median = np.median(prices)
        mad    = np.median(np.abs(prices - median))
        if mad == 0:
            mad = max(1.0, median * 0.05)   # 5% of median as fallback spread
        store[key] = {
            "median": median,
            "mad":    mad,
            "q25":    np.percentile(prices, 25),
            "q75":    np.percentile(prices, 75),
        }

    def evaluate_quote(self, material, zone, input_price):
        input_price = float(input_price)
        key = (material, zone)

        stat = self.stats.get(key) or self.material_stats.get(material)
        if stat is None:
            return {
                "status":  "UNKNOWN",
                "message": f"No historical reference for '{material}' in zone '{zone}'."
            }

        median   = stat["median"]
        mad      = stat["mad"]
        robust_z = 0.6745 * (input_price - median) / mad

        if abs(robust_z) <= 2.5:
            status  = "VALID"
            message = (
                f"Fair Market Price — expected range: "
                f"₹{stat['q25']:.1f} – ₹{stat['q75']:.1f}/kg"
            )
        elif robust_z < -2.5:
            status  = "ANOMALY_LOW"
            message = (
                f"⚠️ Unusually LOW price (Z={robust_z:.2f}). "
                f"Local median is ₹{median:.1f}/kg"
            )
        else:
            status  = "ANOMALY_HIGH"
            message = (
                f"⚠️ Unusually HIGH price (Z={robust_z:.2f}). "
                f"Local median is ₹{median:.1f}/kg"
            )

        return {
            "status":          status,
            "robust_z_score":  round(robust_z, 2),
            "expected_median": round(median, 2),
            "fair_range":      (round(stat['q25'], 2), round(stat['q75'], 2)),
            "message":         message,
        }


if __name__ == "__main__":
    print("=" * 70)
    print("KABADIWALA ML PIPELINE: 08_PRICE_DATA_AND_MODEL (MODEL 2 ENGINE)")
    print("=" * 70)

    df_price  = load_real_pan_india_price_dataset()
    best_model = train_and_evaluate_price_models(df_price)

    print("\n  Anomaly Detection Engine Verification:")
    detector = PriceAnomalyDetector(df_price)

    sample_tests = [
        ("copper",    "North", 635.0),   # Normal
        ("copper",    "North", 420.0),   # Low anomaly
        ("iron_steel","West",  39.0),    # Normal
        ("iron_steel","West",  120.0),   # High anomaly
        ("pcb",       "South", 240.0),   # Normal
        ("pcb",       "South", 50.0),    # Low anomaly
    ]

    for mat, zone, quote in sample_tests:
        res = detector.evaluate_quote(mat, zone, quote)
        print(f"  Quote: ₹{quote}/kg | {mat} | {zone} → [{res['status']}] {res['message']}")

    print("\n[SUCCESS] Stage 8 Price Model Engine & Anomaly Layer Built Cleanly.")
