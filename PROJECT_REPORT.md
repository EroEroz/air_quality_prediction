# Project Report: HCMC Air Quality Prediction (DAT301m)

## 1. Project Overview
This project focuses on forecasting PM2.5 concentration levels and safety categories in Ho Chi Minh City using Time-Series analysis. It demonstrates the transition from baseline statistical methods to advanced machine learning and deep learning architectures, ultimately culminating in a robust, production-ready Classification model enhanced with domain-specific heuristics.

### Objective
- Predict general air quality for the next 24 hours.
- Provide shift-based (Morning, Afternoon, Night) air quality classifications.
- Build a consistent, user-friendly web application without model contradiction.

---

## 2. Data Description
- **Dataset:** Historical PM2.5 data (2018–2022) combined with meteorological features.
- **Source:** US Diplomatic Post (HCMC) and OpenAQ.
- **Resolution:** Hourly & Shift-based (Morning: 6-12h, Afternoon: 12-18h, Night: 18-6h).
- **Features (24 Core Features):** 
  - **Weather:** Temp, Humidity, Wind Speed/Direction, Pressure, Precipitation, Cloud Cover.
  - **Pollutants:** PM10, CO, NO2, SO2, O3.
  - **Time-Series / Lags:** Crucial lag features for PM2.5, Temperature, and AQI Class (1h, 2h, 3h, 6h).

---

## 3. Research & Development Phases

### Phase 1: Baseline Exploration (Present 1)
Implemented basic regression models to establish a performance floor:
- **Linear Regression:** R² = 0.628
- **Simple LSTM:** R² = 0.644

### Phase 2: Advanced Regression Models (Present 2)
Explored complex architectures to maximize regression metrics:
- **Voting Ensemble:** Combined XGBoost, LightGBM, Random Forest, and Logistic Regression.
- **Bi-Directional LSTM:** Achieved the highest regression R² score of **0.647**.
- **Limitation:** We discovered that while R² scores improved, Regression models often failed to reliably predict sudden, extreme pollution spikes, making them less ideal for safety-critical applications.

---

## 4. Production Implementation: The Single-Model Classification Approach
For the final web demonstration, we made a deliberate architectural pivot. Instead of deploying complex regression models that might contradict one another, we elected to deploy a single, highly-optimized **Random Forest Classifier**.

### A. SMOTE Random Forest Classifier
- **Target:** 3-class classification (Good, Moderate, Poor).
- **Handling Imbalance:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** during training to handle the natural scarcity of "Poor" air quality samples.

### B. Optimizing for Public Safety (Threshold Tuning)
- We aggressively tuned the probability threshold for the "Poor" category, lowering it to **0.30**.
- **Result:** This shifted the classification boundary to be highly sensitive to "Poor" conditions. While it may occasionally over-warn, it improved our recall for hazardous air quality events dramatically (from **60% to 84%**), acting as a much safer early-warning system.

---

## 5. Web Application & Heuristic UX Integration

### The "Two-Model Consistency" Problem
Deploying a Regression model (to show the exact PM2.5 number) alongside our tuned Classification model (to show the safety category) introduced UX contradictions. For example, Regression might predict 34.0 µg/m³ (Moderate), while the SMOTE Classifier flags it as "Poor". 

### The Solution: Domain-Driven Heuristics
To guarantee 100% User Interface consistency, our backend (`predictor.py`) relies **exclusively** on the Classification model for its ML decisions, and uses **Heuristic Estimation** to generate the UI numbers:
1. **Anchoring:** We extract the most recent real-world PM2.5 reading to use as our numerical baseline.
2. **Diurnal Offsets:** We apply statically defined domain knowledge to adjust this number based on the time of day:
   - **Morning (+2.5 µg/m³):** Pollution increases due to rush-hour traffic and temperature inversions.
   - **Afternoon (-3.0 µg/m³):** Daytime heat and wind disperse pollutants.
   - **Night (+1.5 µg/m³):** Cooling temperatures cause particulates to settle.
3. **Probability-based Fallbacks:** If the baseline is missing, the system synthesizes a number by interpolating the midpoints of our classification categories against the predicted probabilities.

**Conclusion:** This hybrid approach—using powerful Machine Learning for the core safety logic, and Rule-Based Heuristics for the numerical UI rendering—provides users with a seamless, domain-accurate, and contradiction-free experience.

---

## 6. Conclusion
The project successfully evolved from basic statistical exploration into a mature software engineering solution. By prioritizing recall (84%) through a tuned SMOTE Classifier, and solving the Two-Model Consistency problem using domain heuristics, the final application stands as a robust, safe, and highly reliable public health tool.
