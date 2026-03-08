import os
import dill  # noqa: F401 — needed to unpickle xgboost models saved with cloudpickle/dill
import joblib
import numpy as np
import pandas as pd

# ── Paths (relative to backend/ folder) ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS  = os.path.join(BASE_DIR, "artifacts")
DATA_FILE  = os.path.join(BASE_DIR, "data", "Clean For Model", "Air_Quality_linear_ready.csv")

CLASS_NAMES = ["Good", "Moderate", "Poor"]


# ── Lazy-load models once ─────────────────────────────────────────────────────
_models = None
_scaler = None
_feature_cols = None


def _load_models():
    global _models, _scaler, _feature_cols
    if _models is not None:
        return

    _scaler       = joblib.load(os.path.join(ARTIFACTS, "scaler_X_linear.pkl"))
    _feature_cols = joblib.load(os.path.join(ARTIFACTS, "feature_cols_linear.pkl"))

    _models = {
        "xgboost":  joblib.load(os.path.join(ARTIFACTS, "xgboost.pkl")),
        "lightgbm": joblib.load(os.path.join(ARTIFACTS, "lightgbm.pkl")),
        "rf":       joblib.load(os.path.join(ARTIFACTS, "random_forest.pkl")),
        "lr":       joblib.load(os.path.join(ARTIFACTS, "logistic_regression.pkl")),
    }


def predict() -> dict:
    """
    Loads the last row of the processed dataset, runs the VotingClassifier,
    and returns prediction results for the next 24 hours.
    """
    _load_models()

    # ── Read latest data row ─────────────────────────────────────────────────
    df = pd.read_csv(DATA_FILE)
    latest = df.tail(1)[_feature_cols].values

    # Handle any NaN in the latest row by forward-filling from recent rows
    if np.isnan(latest).any():
        df_tail = df.tail(50)[_feature_cols].ffill().bfill()
        latest  = df_tail.tail(1).values

    X_scaled = _scaler.transform(latest)

    # ── Soft Voting (manual) ─────────────────────────────────────────────────
    proba_xgb = _models["xgboost"].predict_proba(X_scaled)[0]
    proba_lgb = _models["lightgbm"].predict_proba(X_scaled)[0]
    proba_rf  = _models["rf"].predict_proba(X_scaled)[0]
    proba_lr  = _models["lr"].predict_proba(X_scaled)[0]

    avg_proba   = (proba_xgb + proba_lgb + proba_rf + proba_lr) / 4
    pred_idx    = int(np.argmax(avg_proba))
    category    = CLASS_NAMES[pred_idx]

    # ── Current PM2.5 estimate ────────────────────────────────────────────────
    pm25_value = float(df["value"].iloc[-1]) if "value" in df.columns else None

    # ── Simulated 24-hour forecast ───────────────────────────────────────────
    # We apply a realistic diurnal fluctuation around the current reading.
    base = pm25_value if pm25_value else 20.0
    rng  = np.random.default_rng(seed=int(base * 100) % 9999)

    forecast_24h = []
    current_hour = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").hour
    for i in range(24):
        hour = (current_hour + i) % 24
        # Rush-hour peaks (7–9 AM and 5–7 PM)
        diurnal = 1.0
        if 7 <= hour <= 9:
            diurnal = 1.25
        elif 17 <= hour <= 19:
            diurnal = 1.20
        elif 2 <= hour <= 5:
            diurnal = 0.75

        pm_pred = max(0, base * diurnal + rng.normal(0, base * 0.08))
        if pm_pred < 12:
            cat = "Good"
        elif pm_pred < 35:
            cat = "Moderate"
        else:
            cat = "Poor"

        forecast_24h.append({
            "hour": (current_hour + i) % 24,
            "label": f"+{i}h",
            "pm25": round(pm_pred, 1),
            "category": cat,
        })

    return {
        "category":    category,
        "probabilities": {
            "Good":     round(float(avg_proba[0]) * 100, 1),
            "Moderate": round(float(avg_proba[1]) * 100, 1),
            "Poor":     round(float(avg_proba[2]) * 100, 1),
        },
        "pm25_current": round(pm25_value, 1) if pm25_value else None,
        "forecast_24h": forecast_24h,
    }


def current_district_data() -> list:
    """
    Returns simulated real-time AQI readings for major HCMC districts
    based on the latest model data as a baseline.
    """
    df = pd.read_csv(DATA_FILE)
    base_pm = float(df["value"].iloc[-1]) if "value" in df.columns else 22.0

    # Major HCMC districts with approximate coordinates
    districts = [
        {"name": "Quận 1",          "lat": 10.7743, "lng": 106.7020},
        {"name": "Quận 3",          "lat": 10.7898, "lng": 106.6861},
        {"name": "Quận 5",          "lat": 10.7538, "lng": 106.6620},
        {"name": "Quận 7",          "lat": 10.7349, "lng": 106.7208},
        {"name": "Quận 10",         "lat": 10.7738, "lng": 106.6670},
        {"name": "Bình Thạnh",      "lat": 10.8123, "lng": 106.7106},
        {"name": "Tân Bình",        "lat": 10.8027, "lng": 106.6478},
        {"name": "Gò Vấp",          "lat": 10.8382, "lng": 106.6647},
        {"name": "Thủ Đức",         "lat": 10.8503, "lng": 106.7717},
        {"name": "Bình Dương border","lat": 10.9102, "lng": 106.7183},
        {"name": "Nhà Bè",          "lat": 10.6923, "lng": 106.7391},
        {"name": "Hóc Môn",         "lat": 10.8906, "lng": 106.5938},
    ]

    rng = np.random.default_rng(seed=42)
    result = []
    for d in districts:
        pm = max(0.5, base_pm + rng.normal(0, base_pm * 0.15))
        if pm < 12:
            cat = "Good"
        elif pm < 35:
            cat = "Moderate"
        else:
            cat = "Poor"
        result.append({**d, "pm25": round(pm, 1), "category": cat})

    return result
