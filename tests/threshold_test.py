"""
Threshold Tuning Test - Trick #1
Tests various probability thresholds for Class 2 (Poor) to reduce missed Poor air quality predictions.
Run from the project root: python tests/threshold_test.py
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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
