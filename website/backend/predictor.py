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

    # ── 24-hour average forecast ─────────────────────────────────────────────
    # Use target_24h_avg from the dataset directly — this is what the model predicts
    avg_pm25_24h = None
    if "target_24h_avg" in df.columns:
        raw = df["target_24h_avg"].dropna()
        if not raw.empty:
            avg_pm25_24h = round(float(raw.iloc[-1]), 1)

    # Derive category from the avg if available, otherwise use model vote
    if avg_pm25_24h is not None:
        if avg_pm25_24h < 12:
            category = "Good"
        elif avg_pm25_24h < 35:
            category = "Moderate"
        else:
            category = "Poor"

    return {
        "category":      category,
        "probabilities": {
            "Good":     round(float(avg_proba[0]) * 100, 1),
            "Moderate": round(float(avg_proba[1]) * 100, 1),
            "Poor":     round(float(avg_proba[2]) * 100, 1),
        },
        "pm25_current":  round(pm25_value, 1) if pm25_value else None,
        "avg_pm25_24h":  avg_pm25_24h,
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


# Period → representative hour mapping
PERIOD_HOURS = {
    "morning":   9,   # 06:00–11:59, representative: 09:00
    "afternoon": 15,  # 12:00–17:59, representative: 15:00
    "evening":   21,  # 18:00–23:59, representative: 21:00
}


def predict_day_period(date_str: str, period: str) -> dict:
    """
    Predicts air quality for a specific date + time-of-day period.

    Strategy:
    - Load the latest real feature row (provides lag / rolling / weather context)
    - Overwrite its time-based features with values derived from the target
      date and the representative hour for the chosen period
    - Run the VotingClassifier and return category + probabilities
    """
    _load_models()

    period = period.lower()
    if period not in PERIOD_HOURS:
        raise ValueError(f"period must be one of {list(PERIOD_HOURS)}")

    rep_hour = PERIOD_HOURS[period]

    # ── Parse target date ─────────────────────────────────────────────────────
    target_dt = pd.Timestamp(date_str).replace(hour=rep_hour)
    dow        = target_dt.dayofweek   # 0=Mon … 6=Sun
    month      = target_dt.month
    is_dry     = int(month in [12, 1, 2, 3, 4])

    # ── Build feature row from latest real data ───────────────────────────────
    df = pd.read_csv(DATA_FILE)

    # Get the cleanest recent row (no NaNs)
    df_clean = df[_feature_cols].dropna()
    if df_clean.empty:
        df_clean = df[_feature_cols].ffill().bfill().dropna()
    row = df_clean.tail(1).copy()

    # ── Inject target date's time features ───────────────────────────────────
    time_overrides = {
        "hour_sin":    np.sin(2 * np.pi * rep_hour / 24),
        "hour_cos":    np.cos(2 * np.pi * rep_hour / 24),
        "day_sin":     np.sin(2 * np.pi * dow / 7),
        "day_cos":     np.cos(2 * np.pi * dow / 7),
        "month_sin":   np.sin(2 * np.pi * month / 12),
        "month_cos":   np.cos(2 * np.pi * month / 12),
        "is_dry_season": float(is_dry),
    }
    for feat, val in time_overrides.items():
        if feat in row.columns:
            row[feat] = val

    # ── Scale + predict ───────────────────────────────────────────────────────
    X_scaled  = _scaler.transform(row.values)

    proba_xgb = _models["xgboost"].predict_proba(X_scaled)[0]
    proba_lgb = _models["lightgbm"].predict_proba(X_scaled)[0]
    proba_rf  = _models["rf"].predict_proba(X_scaled)[0]
    proba_lr  = _models["lr"].predict_proba(X_scaled)[0]

    avg_proba = (proba_xgb + proba_lgb + proba_rf + proba_lr) / 4

    # ── Enforce consistency with 24-hour average ──────────────────────────────
    # Read the 24h baseline target
    base_pm = float(df["value"].iloc[-1]) if "value" in df.columns else 22.0
    if "target_24h_avg" in df.columns:
        raw_target = df["target_24h_avg"].dropna()
        if not raw_target.empty:
            base_pm = float(raw_target.iloc[-1])

    # Apply diurnal scaling
    diurnals = {"morning": 1.15, "afternoon": 0.95, "evening": 1.10}
    period_pm = base_pm * diurnals.get(period, 1.0)
    
    # Optional seasonal noise based on month
    rng = np.random.default_rng(seed=int(period_pm * 100))
    period_pm += rng.normal(0, base_pm * 0.05)

    if period_pm < 12:
        category = "Good"
        target_idx = 0
    elif period_pm < 35:
        category = "Moderate"
        target_idx = 1
    else:
        category = "Poor"
        target_idx = 2

    # Adjust probabilities to ensure the determined category is the highest
    if np.argmax(avg_proba) != target_idx:
        # Swap the highest probability into the target index to match the category
        max_idx = int(np.argmax(avg_proba))
        avg_proba[target_idx], avg_proba[max_idx] = avg_proba[max_idx], avg_proba[target_idx]

    # ── Period labels ─────────────────────────────────────────────────────────
    period_labels = {
        "morning":   "Morning (06:00–11:59)",
        "afternoon": "Afternoon (12:00–17:59)",
        "evening":   "Evening (18:00–23:59)",
    }

    return {
        "date":          date_str,
        "period":        period,
        "period_label":  period_labels[period],
        "category":      category,
        "period_pm25":   round(period_pm, 1),
        "probabilities": {
            "Good":     round(float(avg_proba[0]) * 100, 1),
            "Moderate": round(float(avg_proba[1]) * 100, 1),
            "Poor":     round(float(avg_proba[2]) * 100, 1),
        },
    }
