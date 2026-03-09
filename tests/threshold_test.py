"""
Threshold Tuning Test - Trick #1
Tests various probability thresholds for Class 2 (Poor) to reduce missed Poor air quality predictions.
Run from the project root: python tests/threshold_test.py
"""

import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _safe_metric(report_dict: dict, class_key: str, metric: str) -> float:
    return float(report_dict.get(class_key, {}).get(metric, 0.0))


def _init_wandb(thresholds: list):
    """Initialize wandb run only when available and enabled by env var."""
    if wandb is None:
        print("WandB not installed. Skipping WandB logging.")
        return None

    if os.getenv("USE_WANDB", "1") != "1":
        print("USE_WANDB!=1. Skipping WandB logging.")
        return None

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "dat301m-air-quality"),
        name=os.getenv("WANDB_RUN_NAME", "threshold-tuning-test"),
        job_type="threshold_tuning",
        config={
            "model_file": "saved_models/aqi_model.pkl",
            "data_file": "tests/demo_forecast_data.csv",
            "thresholds": thresholds,
            "good_threshold": 0.40,
            "poor_threshold_default": 0.50,
        },
    )
    return run

# ── Load model and test data ──────────────────────────────────────────────
print("Loading model and data...")
model = joblib.load("saved_models/aqi_model.pkl")
df = pd.read_csv("tests/demo_forecast_data.csv")

# Features used for training (24 features as described in the SMOTE output)
feature_cols = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation", "cloud_cover",
    "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "pm2_5_lag_1", "pm2_5_lag_2", "pm2_5_lag_3", "pm2_5_lag_6",
    "temperature_2m_lag_1", "temperature_2m_lag_2", "temperature_2m_lag_3", "temperature_2m_lag_6",
    "AQI_Class_lag_1", "AQI_Class_lag_2", "AQI_Class_lag_3", "AQI_Class_lag_6",
]

X_test = df[feature_cols]
y_true = df["AQI_Class_Actual"]

# ── Baseline: Default model predictions ───────────────────────────────────
print("\n" + "="*60)
print("BASELINE (Default threshold = 0.5)")
print("="*60)
y_pred_default = model.predict(X_test)
print(classification_report(y_true, y_pred_default, target_names=["Good", "Moderate", "Poor"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_default))
baseline_f1 = f1_score(y_true, y_pred_default, average="macro")
print(f"Macro F1: {baseline_f1:.4f}")

# ── Get prediction probabilities ──────────────────────────────────────────
probas = model.predict_proba(X_test)
# Columns: [Prob_Good(0), Prob_Moderate(1), Prob_Poor(2)]

# ── Test different thresholds for "Poor" ─────────────────────────────────
thresholds_to_test = [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45]
wandb_run = _init_wandb(thresholds_to_test)

default_report = classification_report(y_true, y_pred_default, output_dict=True)
default_accuracy = float(np.mean(np.array(y_pred_default) == y_true.values))
default_poor_recall = _safe_metric(default_report, "2", "recall")
default_poor_precision = _safe_metric(default_report, "2", "precision")

if wandb_run is not None:
    wandb.log(
        {
            "baseline/threshold": 0.50,
            "baseline/accuracy": default_accuracy,
            "baseline/macro_f1": float(baseline_f1),
            "baseline/poor_recall": default_poor_recall,
            "baseline/poor_precision": default_poor_precision,
        }
    )

results = []
for poor_threshold in thresholds_to_test:
    y_pred_custom = []
    for prob in probas:
        if prob[2] >= poor_threshold:  # If >= threshold confident it's Poor → warn
            y_pred_custom.append(2)
        elif prob[0] >= 0.40:          # If >= 40% confident it's Good
            y_pred_custom.append(0)
        else:
            y_pred_custom.append(1)

    macro_f1 = f1_score(y_true, y_pred_custom, average="macro")
    poor_recall = classification_report(y_true, y_pred_custom, output_dict=True)["2"]["recall"]
    poor_precision = classification_report(y_true, y_pred_custom, output_dict=True)["2"]["precision"]
    accuracy = np.mean(np.array(y_pred_custom) == y_true.values)

    results.append({
        "poor_threshold": poor_threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "poor_recall": poor_recall,
        "poor_precision": poor_precision,
    })

    if wandb_run is not None:
        wandb.log(
            {
                "sweep/threshold": float(poor_threshold),
                "sweep/accuracy": float(accuracy),
                "sweep/macro_f1": float(macro_f1),
                "sweep/poor_recall": float(poor_recall),
                "sweep/poor_precision": float(poor_precision),
            }
        )

# ── Print comparison table ────────────────────────────────────────────────
print("\n" + "="*60)
print("THRESHOLD COMPARISON (varying 'Poor' probability threshold)")
print("="*60)
print(f"{'Threshold':>10} {'Accuracy':>10} {'MacroF1':>10} {'PoorRecall':>12} {'PoorPrecision':>15}")
print("-"*60)
for r in results:
    print(f"{r['poor_threshold']:>10.2f} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['poor_recall']:>12.4f} {r['poor_precision']:>15.4f}")

# ── Best threshold by Macro F1 ────────────────────────────────────────────
best = max(results, key=lambda x: x["macro_f1"])
print(f"\n[BEST] Best threshold by Macro F1: {best['poor_threshold']}")
print(f"   Accuracy:      {best['accuracy']:.4f}")
print(f"   Macro F1:      {best['macro_f1']:.4f}")
print(f"   Poor Recall:   {best['poor_recall']:.4f}")
print(f"   Poor Precision:{best['poor_precision']:.4f}")

if wandb_run is not None:
    wandb.log(
        {
            "best/threshold": float(best["poor_threshold"]),
            "best/accuracy": float(best["accuracy"]),
            "best/macro_f1": float(best["macro_f1"]),
            "best/poor_recall": float(best["poor_recall"]),
            "best/poor_precision": float(best["poor_precision"]),
            "improvement/poor_recall_delta": float(best["poor_recall"] - default_poor_recall),
        }
    )

    threshold_table = wandb.Table(
        columns=["threshold", "accuracy", "macro_f1", "poor_recall", "poor_precision"]
    )
    for row in results:
        threshold_table.add_data(
            float(row["poor_threshold"]),
            float(row["accuracy"]),
            float(row["macro_f1"]),
            float(row["poor_recall"]),
            float(row["poor_precision"]),
        )
    wandb.log({"threshold_sweep": threshold_table})

# ── Full report for best threshold ───────────────────────────────────────
print(f"\n{'='*60}")
print(f"FULL REPORT at threshold = {best['poor_threshold']}")
print("="*60)
y_pred_best = []
for prob in probas:
    if prob[2] >= best["poor_threshold"]:
        y_pred_best.append(2)
    elif prob[0] >= 0.40:
        y_pred_best.append(0)
    else:
        y_pred_best.append(1)

print(classification_report(y_true, y_pred_best, target_names=["Good", "Moderate", "Poor"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_best))

if wandb_run is not None:
    wandb.finish()
