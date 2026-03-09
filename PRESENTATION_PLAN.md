# Presentation Outline: Air Quality Prediction Project

## Slide 1: Title & Team
- **Title:** HCMC Air Quality Forecasting using Machine Learning
- **Course:** DAT301m - Time Series Analysis
- **Core Methodology:** SMOTE Random Forest Classification & Heuristic UX Integration

## Slide 2: Problem Definition & Scope
- **Problem:** Increasing PM2.5 levels in HCMC due to urbanization.
- **Goal:** Build an accurate time-series model to forecast pollution levels and warn citizens about hazardous air.
- **Scope:** Focused on HCMC urban area using historical station data (OpenAQ & US Embassy).

## Slide 3: Research Phase & Explored Models
- **Initial Baseline (Present 1):** Linear Regression, Simple LSTM (R² Score ~ 0.62).
- **Advanced Exploration (Present 2):** Voting Ensembles, Bi-Directional LSTM (Highest R² at 0.647).
- **Key Discovery:** While regression models provided good R² scores, they often struggled to accurately predict sudden, dangerous spikes in pollution contextually.

## Slide 4: The Production Shift - Focus on Classification
- **Strategic Pivot:** Transitioned from Regression (predicting exact numbers) to Classification (predicting safety categories: Good, Moderate, Poor).
- **Model Choice:** Developed a tuned **Random Forest Classifier** utilizing 24 features (Weather + PM2.5 Lags).
- **SMOTE Integration:** Applied Synthetic Minority Over-sampling Technique to handle the imbalance in "Poor" air quality samples.

## Slide 5: Optimizing for Public Safety
- **Threshold Tuning:** Adjusted the classification threshold for the "Poor" category down to **0.30**.
- **Impact:** By acting as an early warning system, we increased the recall rate for hazardous air quality events from 60% to 84%.
- **Philosophy:** Prioritizing public health warnings over marginal accuracy gains in safe conditions.

## Slide 6: The "Two-Model Consistency" Problem in Production
- **The UX Challenge:** Users expect to see a specific PM2.5 number, but using a separate Regression model alongside our tuned Classifier creates "Model Contradiction" (e.g., Regression predicts 34 µg/m³ [Moderate], while Classifier flags it as "Poor" [Hazardous]).
- **The Solution:** A unified Single-Model Architecture. The Classification model acts as the sole Machine Learning source of truth to guarantee 100% UI consistency.

## Slide 7: Domain-Driven Heuristics (The UI Integration)
- **Heuristic Estimation:** Instead of predicting a confusing regression number, we synthetically derive the UI number using a recent historical baseline.
- **Diurnal Offsets:** We apply domain-knowledge heuristics to the baseline based on HCMC pollution patterns:
  - **Morning (+2.5 µg/m³):** Rush hour traffic & temperature inversions.
  - **Afternoon (-3.0 µg/m³):** Daytime heat and wind dispersion.
  - **Night (+1.5 µg/m³):** Cooling temperatures cause pollutants to settle.

## Slide 8: Demo Showcase (The Web App)
- **UI Design:** Clean, interactive dashboard with AI-powered forecasting.
- **Features:** 
  - **24-Hour Forecast & Day-Period Selector:** Both powered consistently by the same underlying Random Forest model.
  - **Probability Bars:** Boundary-distance math visualizing the model's certainty.
  - **Simulated Spatial Heatmap:** Demonstrating visual AQI distribution.

## Slide 9: Conclusion
- **Engineering Choice:** Proved that combining powerful Machine Learning (for core logic) with Rule-Based Heuristics (for UI rendering) creates a robust, user-friendly application.
- **Achievement:** Successfully improved hazard detection (84% recall) while maintaining a seamless, non-contradictory User Experience.
