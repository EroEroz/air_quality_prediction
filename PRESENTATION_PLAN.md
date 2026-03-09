# Presentation Outline: Air Quality Prediction Project

## Slide 1: Title & Team
- **Title:** HCMC Air Quality Forecasting using Machine Learning
- **Course:** DAT301m - Time Series Analysis
- **Core Methodology:** Bi-LSTM & Voting Ensemble Approach

## Slide 2: Problem Definition & Scope
- **Problem:** Increasing PM2.5 levels in HCMC due to urbanization.
- **Goal:** Build an accurate time-series model to forecast pollution levels 24h ahead.
- **Scope:** Focused on HCMC urban area using 5 years of historical station data.

## Slide 3: Data Analysis & Metrics
- **Data Source:** Open Meteo.
- **Key Discovery:** High seasonality (Dry vs. Rainy season) and strong correlation with Humidity and Wind Speed.
- **Metrics used:** 
  - **Regression:** MAE, RMSE, R² (Coefficient of Determination).
  - **Classification:** Accuracy, F1-Score (focused on detecting "Poor" air quality).

## Slide 4: Baseline Models (Present 1 Review)
- **Starting Point:** Linear Regression and Simple LSTM.
- **Performance:** R² Score around 0.62.
- **Finding:** Simple models failed to capture sudden pollution spikes (Poor quality).

## Slide 5: Proposed Improvements (Present 2)
- **Feature Engineering:** Added Lags (1, 24, 168h) and weather interaction terms.
- **Model Diversity:** Introduced a **Voting Ensemble** (XGBoost, Random Forest, etc.) to stabilize predictions.
- **Deep Learning:** Implemented **Bi-LSTM** to capture temporal patterns more effectively.

## Slide 6: Model Architecture (The "Blocks")
- **Input Block:** 14 processed features (Weather + Lags).
- **Core Block:** Bi-Directional LSTM Layer (Bidirectional context).
- **Output Block:** Dense layer for regression output (+ Classification head for shifts).

## Slide 7: Training & Fine-tuning Results
- **Optimization:** Used SMOTE to balance the dataset.
- **Threshold Tuning:** Adjusted "Poor" classification threshold to **0.30** based on the PR-curve.
- **Result:** **R² Score improved to 0.647** (A 4% relative increase over baseline).

## Slide 8: Technical Innovation
- **Shift-based Analysis:** Breaking the day into Morning/Afternoon/Evening periods.
- **Inertia Anchor:** Anchoring shift predictions to the 24h baseline to ensure physical consistency in the forecast.

## Slide 9: Demo Showcase (The Web App)
- **UI Design:** Clean, interactive dashboard.
- **Features:** 
  - Real-time heatmaps (Simulated).
  - 24h Probability-based Forecast.
  - Interactive "Day-Period" selector.

## Slide 10: Conclusion & Future Scope
- **Achievement:** Successfully improved accuracy and balanced the model for high-pollution detection.
- **Next steps:** Integrating live API feeds and expanding to other cities like Hanoi.
