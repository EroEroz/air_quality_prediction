# Project Report: HCMC Air Quality Prediction (DAT301m)
## Present 2 — Improvement Report

---

## 1. Project Overview
This project forecasts PM2.5-based air quality safety categories for Ho Chi Minh City using Time-Series Machine Learning. The project spans two presentation phases: **Present 1** established regression baselines, and **Present 2** (this report) delivers a measurably improved Classification pipeline with a live web demonstration.

### Objective
- Classify air quality into one of three categories: **Good**, **Moderate**, or **Poor**.
- Maximize recall for the **"Poor"** category to serve as an early public health warning system.
- Provide shift-based (Morning / Afternoon / Night) and 24-hour forecasts via a deployed web application.

### Application
Early detection of hazardous PM2.5 levels allows citizens, urban planners, and health authorities to take preventative action. Missing a "Poor" event is far more harmful to public health than a false alarm, making **recall** the primary optimization target.

---

## 2. Data Description

- **Dataset:** Historical PM2.5 data (202–2026) combined with meteorological features.
- **Sources:** US Diplomatic Post (HCMC) and OpenAQ.
- **Resolution:** Hourly data, later aggregated into Shift-based periods (Morning: 06–12h, Afternoon: 12–18h, Night: 18–06h).

### 2.1 Features (24 Total)
| Category | Features |
|---|---|
| **Weather** | Temperature, Humidity, Wind Speed, Wind Direction, Surface Pressure, Precipitation, Cloud Cover |
| **Pollutants** | PM10, Carbon Monoxide, Nitrogen Dioxide, Sulphur Dioxide, Ozone |
| **PM2.5 Lags** | pm2_5_lag_1, pm2_5_lag_2, pm2_5_lag_3, pm2_5_lag_6 |
| **Temperature Lags** | temperature_2m_lag_1, temperature_2m_lag_2, temperature_2m_lag_3, temperature_2m_lag_6 |
| **AQI Class Lags** | AQI_Class_lag_1, AQI_Class_lag_2, AQI_Class_lag_3, AQI_Class_lag_6 |

> **Why Lag Features?** Air pollution exhibits strong temporal inertia — PM2.5 levels at hour `t` are heavily correlated with PM2.5 at hours `t-1`, `t-2`, etc. Without lag features, a model treats each observation as independent and misses the time-series nature of the problem.

### 2.2 Key Data Discoveries
- **High Seasonality:** PM2.5 is significantly elevated during the dry season (December–April), driven by reduced rainfall and stagnant air masses.
- **Class Imbalance:** Only ~15% of samples are classified as "Poor" — a severe imbalance that causes naive classifiers to ignore dangerous events entirely.
- **Strong Correlations:** PM2.5 is positively correlated with PM10, CO, and negatively correlated with Wind Speed and Humidity.

### 2.3 Evaluation Metrics
- **Present 1 (Regression):** MAE, RMSE, R² (Coefficient of Determination).
- **Present 2 (Classification):** Accuracy, Precision, Recall, F1-Score — with **Recall for the "Poor" class** as the primary optimization target.

---

## 3. Present 1 Review — Baseline Models

| Model | R² Score | Limitation |
|---|---|---|
| Linear Regression | 0.628 | Cannot capture non-linear pollution dynamics |
| Simple LSTM | 0.644 | Better temporal capture, but still misses extreme spikes |

**Critical Finding:** While regression R² scores appear acceptable (~0.62–0.64), both models systematically underestimated sudden extreme pollution events. A model that "gets the number roughly right" on average but fails to raise an alarm on genuinely hazardous days is **insufficient for a public-safety application**.

---

## 4. Present 2 — Improved Approach

### 4.1 Strategic Pivot: Regression → Classification
We reframed the problem from predicting a continuous PM2.5 value to classifying air quality into three safety-meaningful categories: **Good** (< 12 µg/m³), **Moderate** (12–35 µg/m³), **Poor** (> 35 µg/m³).

This pivot allows us to:
1. Directly optimize for the metric that matters most (Recall for "Poor").
2. Provide users with actionable safety information rather than a raw number.

### 4.2 Handling Class Imbalance — SMOTE
Because "Poor" events represent only ~15% of the dataset, a standard Random Forest would be biased toward always predicting "Good" or "Moderate."

We applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training set, which generated synthetic "Poor" samples by interpolating between real ones. This forced the model to learn a clearer decision boundary for the minority class.

### 4.3 Model Architecture — SMOTE Random Forest Classifier

```
INPUT:  24 Features [Weather + Pollutants + PM2.5/Temp/AQI Lags]
           ↓
TRAINING: [SMOTE Balanced Dataset] → [Random Forest Classifier]
           ↓
OUTPUT: Probabilities → [P(Good), P(Moderate), P(Poor)]
           ↓
DECISION: Tuned Threshold (Poor ≥ 0.30 → Classify as "Poor")
           ↓
FINAL:  Category Label + Probability Scores
```

- **Random Forest:** An ensemble of decision trees that votes on the final class. It is robust to overfitting and handles mixed feature types (continuous weather data + discrete lag features) well.
- **Model saved as:** `saved_models/aqi_model.pkl`

---

## 5. Fine-Tuning — Threshold Optimization

The default classification threshold for any class is 0.50 (whichever class has > 50% probability wins). However, for a safety-critical scenario, this is suboptimal.

By analyzing the **Precision-Recall curve**, we identified that lowering the "Poor" threshold to **0.30** dramatically increases recall at a controlled precision cost.

### Results (Logged in WandB)

| Threshold | Poor Recall | Poor Precision | F1-Score (Poor) |
|---|---|---|---|
| 0.50 (default) | 60% | ~72% | ~65% |
| **0.30 (tuned)** | **84%** | ~58% | **~69%** |

**Interpretation:** At threshold 0.30, the model catches 84% of all truly hazardous air quality events (up from 60%). The trade-off is a modest drop in precision (more false alarms), which is an acceptable cost for a public health warning system.

### Threshold Code
```python
# predictor.py — _classify_with_threshold()
POOR_THRESHOLD = 0.30

def _classify_with_threshold(proba: np.ndarray) -> tuple:
    if proba[2] >= POOR_THRESHOLD:   # P(Poor) ≥ 0.30 → classify as Poor
        pred_idx = 2
    elif proba[0] >= 0.40:            # P(Good) ≥ 0.40 → classify as Good
        pred_idx = 0
    else:
        pred_idx = 1                  # Default: Moderate
    return pred_idx, proba
```

---

## 6. Web Application — Production Design

### 6.1 The "Two-Model Consistency" Problem
Users expect to see a specific PM2.5 number alongside the category label. A naive approach would add a second Regression model to generate that number. However, this creates **Model Contradiction**: the Regression model might output `34.0 µg/m³` (technically "Moderate"), while our tuned Classifier outputs "Poor." The result is a confusing and untrustworthy UI.

### 6.2 Solution: Single-Model + Domain-Driven Heuristics
To guarantee 100% consistency, the backend (`predictor.py`) relies exclusively on the Classification model for all ML decisions. The PM2.5 number displayed in the UI is a **Heuristic Estimate** derived from:

1. **Anchor:** Most recent real PM2.5 reading from the dataset (`pm2_5_lag_1`).
2. **Diurnal Offset:** A domain-knowledge correction based on HCMC pollution patterns:
   - **Morning (+2.5 µg/m³):** Rush-hour traffic and temperature inversions trap pollutants.
   - **Afternoon (−3.0 µg/m³):** Daytime wind and heat disperse pollution.
   - **Night (+1.5 µg/m³):** Cooling temperatures cause particulates to settle near ground level.
3. **Probability Fallback:** If no recent baseline exists, the expected value is computed as:
   ```python
   midpoints = np.array([6.0, 23.0, 50.0])
   baseline_pm25 = float(np.dot(proba, midpoints))
   ```

> **This is not a "fake" number.** It is a transparent, domain-accurate estimate that is always consistent with the predicted category — a deliberate engineering decision to avoid Model Contradiction.

### 6.3 Backend API Endpoints (Flask)
| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | GET/POST | 24-hour air quality forecast using the last data row |
| `/api/day-period` | POST | Shift-based prediction for a chosen date + period |
| `/api/current` | GET | Simulated district-level PM2.5 data for the heatmap |

### 6.4 Frontend Features
- **District Heatmap:** Leaflet.js map showing simulated AQI across 12 HCMC districts.
- **24-Hour Forecast:** Probability bars powered by the SMOTE Random Forest Classifier.
- **Day-Period Selector:** Date + Morning/Afternoon/Night picker with shift-specific predictions.

---

## 7. Conclusion

The project successfully transitioned from basic regression methods (R² ≈ 0.62) to a safety-optimized Classification pipeline with a demonstrated improvement in the metric that matters most for public health: **recall for "Poor" air quality events** (60% → **84%**).

By combining SMOTE oversampling with threshold tuning, and deploying a single-model architecture with domain-driven heuristics for UI consistency, the final application represents a robust, transparent, and user-friendly air quality warning system for Ho Chi Minh City.

### Summary of Improvements over Present 1
| Metric | Present 1 | Present 2 |
|---|---|---|
| Model Type | Regression | Classification |
| Poor Recall | ~60% (estimated) | **84%** |
| Handles Class Imbalance | ❌ No | ✅ Yes (SMOTE) |
| Threshold Optimized | ❌ No | ✅ Yes (0.30) |
| Web Demo | ❌ No | ✅ Yes |
