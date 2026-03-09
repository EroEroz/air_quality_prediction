# Project Report: HCMC Air Quality Prediction (DAT301m)

## 1. Project Overview
This project focuses on forecasting PM2.5 concentration levels in Ho Chi Minh City using Time-Series analysis. It demonstrates the transition from baseline statistical methods to advanced machine learning ensembles and deep learning architectures.

### Objective
- Predict 24-hour average PM2.5 levels.
- Provide shift-based (Morning, Afternoon, Evening) air quality classifications.
- Visualize spatial distribution via a district-level heatmap.

---

## 2. Data Description
- **Dataset:** Historical PM2.5 data (2018–2022) combined with meteorological features.
- **Source:** US Diplomatic Post (HCMC) and OpenAQ.
- **Resolution:** Hourly & Shift-based (Morning: 6-12h, Afternoon: 12-18h, Night/Evening: 18-6h).
- **Features:** 
  - **Weather:** Temp, Humidity, Wind Speed/Direction, Pressure, Precipitation, Cloud Cover.
  - **Pollutants:** PM10, CO, NO2, SO2, O3.
  - **Time-Series:** Lag features (1h, 24h, 168h), Cyclical time (Sin/Cos).

---

## 3. Methodology & Model Architecture

### Phase 1: Baseline (Present 1)
Implemented basic regression models to establish a performance floor:
- **Linear Regression:** R² = 0.628
- **Simple LSTM:** R² = 0.644

### Phase 2: Advanced Improvement (Present 2 - Current)
To improve performance as required by the course, we implemented a dual-path approach:

#### A. Voting Ensemble (Regression)
Combines multiple high-performance estimators:
- **XGBoost, LightGBM, Random Forest, Logistic Regression.**
- **Benefit:** Reduces variance and prevents overfitting on specific seasonal patterns.

#### B. Bi-Directional LSTM (Deep Learning)
- **Architecture:** 64-unit Bi-LSTM layer followed by 32-unit Dense layer.
- **R² Score:** **0.647** (Highest performing model).
- **Benefit:** Captures both past and "future" context within the sequence window.

#### C. Classification Model (Shift-based)
Used for the "Day-Period" feature:
- **Target:** 3-class classification (Good, Moderate, Poor).
- **Improvement:** Applied **Poor Threshold Tuning (0.30)** to increase recall for hazardous air quality events (Recall improved from 60% to 84%).

---

## 4. Feature Engineering
- **Lag Features:** Crucial for capture inertia in air quality (PM2.5 levels don't change instantly).
- **SMOTE (Synthetic Minority Over-sampling Technique):** Used during training to handle the imbalance in "Poor" air quality samples.
- **Threshold Optimization:** Shifted classification boundary to be more sensitive to "Poor" conditions, providing a safer warning system for users.

---

## 5. Web Demo (Demo Project)
Developed a Flask-based backend and Vanilla JS frontend to demonstrate the models in a real-world context:
- **Spatial Heatmap:** Simulates AQI across HCMC districts.
- **Probability Bars:** Uses boundary-distance math to represent model uncertainty visually.
- **Static Demo Mode:** Configured for school presentation using stable historical data (Historical baseline: 38.5 µg/m³).

---

## 6. Conclusion
The project successfully improved upon initial baseline models by 3-5% in accuracy metrics. The inclusion of ensemble methods and Bi-LSTM allows the system to handle the complex, multi-variable nature of urban air pollution effectively.
