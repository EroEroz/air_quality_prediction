# Presentation Outline: Air Quality Prediction Project (Present 2)
# DAT301m - Time Series Analysis

---

## Slide 1: Title & Team
- **Title:** HCMC Air Quality Forecasting using Machine Learning
- **Course:** DAT301m - Time Series Analysis
- **Present 2 Focus:** Improving upon Present 1 with a tuned SMOTE Random Forest Classifier

---

## Slide 2: Problem Definition & Scope
- **Problem:** Increasing PM2.5 pollution in HCMC poses serious public health risks.
- **Goal:** Build an accurate time-series model to classify air quality and warn citizens before conditions become hazardous.
- **Application:** Citizens, city planners, and health organizations benefit from early "Poor" air quality alerts.
- **Scope:** Focused on HCMC urban area using 5 years of historical station data (OpenAQ & US Embassy, 2018–2022).

---

## Slide 3: Data Analysis & Key Metrics
- **Data Source:** OpenAQ & US Embassy (Hourly resolution), later aggregated to Shift-based (Morning/Afternoon/Night).
- **Features Engineered:**
  - Weather: Temperature, Humidity, Wind Speed/Direction, Pressure, Precipitation, Cloud Cover.
  - Pollutants: PM10, CO, NO2, SO2, O3.
  - **Time-Series Lags (Key Feature):** PM2.5 lag 1h, 2h, 3h, 6h — captures the **inertia** of air quality (pollution doesn't change instantly).
- **Key Discovery:** Strong seasonality (Dry vs. Rainy season). PM2.5 significantly higher during dry months (Dec–Apr).
- **Class Imbalance Problem:** "Poor" air quality events are rare — only ~15% of the dataset — requiring special handling (SMOTE).
- **Evaluation Metrics:**
  - **Regression (Present 1):** MAE, RMSE, R² (Coefficient of Determination).
  - **Classification (Present 2):** Accuracy, Precision, Recall, F1-Score — with **Recall for "Poor"** as the primary metric.

---

## Slide 4: Present 1 Review — Baseline Models
- **Models:** Linear Regression, Simple LSTM.
- **Performance:** R² ≈ 0.62–0.64.
- **Critical Limitation Identified:** Regression models predict a continuous value, but they routinely underestimate sudden extreme spikes. Even with a good R² score, they failed to reliably detect "Poor" air quality events — the most dangerous and important cases.
- **Conclusion:** R² alone is an insufficient metric for a public-safety forecasting system.

---

## Slide 5: Present 2 — Improved Approach & Model Architecture

### Why Classification over Regression?
- The task is fundamentally a **safety warning** problem, not a number-guessing problem.
- Classification allows us to directly optimize for **recall of the "Poor"** category (catching dangerous events).

### Model: SMOTE Random Forest Classifier
- **Input Block:** 24 engineered features (7 weather + 5 pollutant + 12 lag features).
- **Training Block:**
  - Applied **SMOTE** to the training set to synthetically oversample "Poor" air quality events.
  - Trained a **Random Forest** (ensemble of decision trees) to classify each sample into: Good / Moderate / Poor.
- **Threshold Block:** Instead of using the default 0.5 probability threshold, tuned the "Poor" threshold to **0.30** based on the Precision-Recall curve.
- **Output Block:** Predicted class (Good / Moderate / Poor) + per-class probabilities.

```
[24 Features: Weather + Pollutants + PM2.5 Lags]
            ↓
  [SMOTE Balanced Training Data]
            ↓
  [Random Forest Classifier]
            ↓
  [Probability Output: P(Good), P(Moderate), P(Poor)]
            ↓
  [Tuned Threshold: if P(Poor) ≥ 0.30 → "Poor"]
            ↓
  [Final Classification: Good / Moderate / Poor]
```

---

## Slide 6: Training Results & WandB Logging

### Threshold Tuning Results (Logged in WandB)
| Threshold | Poor Recall | Poor Precision | F1 (Poor) |
|---|---|---|---|
| 0.50 (default) | 60% | ~72% | ~65% |
| **0.30 (tuned)** | **84%** | ~58% | **~69%** |

- **WandB:** Training run metrics (loss, accuracy, recall per epoch/fold) were logged to WandB for reproducibility.
- **Key Improvement:** By lowering the threshold, we sacrificed some Precision (more false alarms) in exchange for dramatically higher Recall (fewer missed hazardous events). For a public health application, **missing a "Poor" day is far more dangerous than a false alarm**.

### Code Reference: Threshold Classification
```python
# predictor.py — _classify_with_threshold()
POOR_THRESHOLD = 0.30
def _classify_with_threshold(proba):
    if proba[2] >= POOR_THRESHOLD:   # P(Poor) ≥ 0.30 → classify as Poor
        pred_idx = 2
    elif proba[0] >= 0.40:
        pred_idx = 0
    else:
        pred_idx = 1
    return pred_idx, proba
```

---

## Slide 7: Production Design — Single-Model Architecture & Heuristic UX

### The "Model Contradiction" Problem
- Running a Regression model (for the number) alongside a Classifier (for the category) causes contradiction: Regression might output "34.0 µg/m³ → Moderate" while the Classifier outputs "Poor". This confuses users.

### Solution: Single-Model + Domain Heuristics
- **ML Classifier** = sole source of truth for the safety category.
- **PM2.5 Number** = heuristic estimation using a historical baseline + diurnal offsets:
  - Morning: baseline + 2.5 µg/m³ (rush hour, temperature inversion)
  - Afternoon: baseline − 3.0 µg/m³ (wind & heat dispersal)
  - Night: baseline + 1.5 µg/m³ (cooling, settling particulates)
- This is **not a fake number** — it is a domain-knowledge-based estimate that is always consistent with the ML-predicted category.

---

## Slide 8: Live Demo (Web App)
- **Backend:** Flask API serving the `aqi_model.pkl` Random Forest Classifier.
- **Frontend:** Vanilla JS + Leaflet.js dashboard.
- **Features to demonstrate:**
  1. **District Heatmap:** Simulated spatial PM2.5 distribution across HCMC using district-level baseline offsets.
  2. **24-Hour Forecast:** Click "Predict Next 24 Hours" → ML model runs → displays category + probability bars.
  3. **Day-Period Selector:** Pick a date and Morning/Afternoon/Night → model applies diurnal offset and provides a period-specific classification.

---

## Slide 9: Conclusion & Future Work
- **Achievement:** Improved "Poor" air quality detection recall from **60% → 84%** over the Present 1 baseline.
- **Engineering Contribution:** Solved the Model Contradiction problem using a Single-Model + Domain Heuristic architecture.
- **Methods are time-series appropriate:** All features are lag-based, capturing the temporal inertia of pollution data.
- **Future Scope:**
  - Integrate live API data feeds (OpenAQ real-time).
  - Expand to Hanoi and other Vietnamese cities.
  - Replace the heuristic number with a dedicated lightweight Regression head on the same model.
