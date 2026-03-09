import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

print("Loading model and data...")
model = joblib.load("saved_models/aqi_model.pkl")
df = pd.read_csv("tests/demo_forecast_data.csv")

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

print("\n" + "="*60)
print("BASELINE (Default threshold = 0.5)")
print("="*60)
y_pred_default = model.predict(X_test)
print(classification_report(y_true, y_pred_default, target_names=["Good", "Moderate", "Poor"]))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_default))
baseline_f1 = f1_score(y_true, y_pred_default, average="macro")
print(f"Macro F1: {baseline_f1:.4f}")

probas = model.predict_proba(X_test)
thresholds_to_test = [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45]

results = []
for poor_threshold in thresholds_to_test:
    y_pred_custom = []
    for prob in probas:
        if prob[2] >= poor_threshold:
            y_pred_custom.append(2)
        elif prob[0] >= 0.40:
            y_pred_custom.append(0)
        else:
            y_pred_custom.append(1)

    rep = classification_report(y_true, y_pred_custom, output_dict=True)
    results.append({
        "poor_threshold": poor_threshold,
        "accuracy": np.mean(np.array(y_pred_custom) == y_true.values),
        "macro_f1": f1_score(y_true, y_pred_custom, average="macro"),
        "poor_recall": rep["2"]["recall"],
        "poor_precision": rep["2"]["precision"],
        "y_pred": y_pred_custom,
    })

print("\n" + "="*60)
print("THRESHOLD COMPARISON")
print("="*60)
print(f"{'Threshold':>10} {'Accuracy':>10} {'MacroF1':>10} {'PoorRecall':>12} {'PoorPrecision':>15}")
print("-"*60)
for r in results:
    print(f"{r['poor_threshold']:>10.2f} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['poor_recall']:>12.4f} {r['poor_precision']:>15.4f}")

best = max(results, key=lambda x: x["macro_f1"])
print(f"\n[BEST] Threshold = {best['poor_threshold']}")
print(f"  Accuracy:       {best['accuracy']:.4f}")
print(f"  Macro F1:       {best['macro_f1']:.4f}")
print(f"  Poor Recall:    {best['poor_recall']:.4f}")
print(f"  Poor Precision: {best['poor_precision']:.4f}")

print(f"\n{'='*60}")
print(f"FULL REPORT at threshold = {best['poor_threshold']}")
print("="*60)
print(classification_report(y_true, best["y_pred"], target_names=["Good", "Moderate", "Poor"]))
print("Confusion Matrix:\n", confusion_matrix(y_true, best["y_pred"]))
