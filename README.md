# HCMC Air Quality Classification (DAT301m)

Shift-based air quality classification project for Ho Chi Minh City using weather + pollutant features, lag engineering, SMOTE balancing, Random Forest training, threshold tuning, and a Flask web demo.

## What This Project Does

- Predicts AQI class at shift level: `Good`, `Moderate`, `Poor`
- Uses time-series aware features (lag-1, lag-2, lag-3, lag-6)
- Handles class imbalance with SMOTE on training data only
- Tunes the `Poor` decision threshold (default `0.50` -> tested down to `0.30`)
- Exposes predictions through a Flask backend + frontend dashboard
- Logs experiments and threshold sweeps to Weights and Biases (WandB)

## Project Structure

- `feature_engineering.py`: builds `processed_data.csv` from raw hourly data
- `model-training.ipynb`: model training, evaluation, export (`aqi_model.pkl` + demo data)
- `tests/threshold_test.py`: threshold sweep and reporting
- `tests/run_threshold.py`: threshold sweep + WandB logging
- `website/backend/app.py`: Flask API
- `website/backend/predictor.py`: model loading and prediction logic
- `saved_models/`: exported model(s)
- `tests/demo_forecast_data.csv`: test/demo data used by API and threshold scripts

## Data

Current source in this repository is Open Meteo-based weather data merged for AQI modeling.

Main raw input used by feature engineering:

- `data/hcmc_weather_and_aqi.csv`

Feature engineering output:

- `processed_data.csv`

### AQI Class Rules (from PM2.5)

- `0` -> Good: `pm2_5 <= 12.0`
- `1` -> Moderate: `12.0 < pm2_5 <= 35.4`
- `2` -> Poor: `pm2_5 > 35.4`

## Setup

Use Python 3.9+ (Conda or venv).

### 1) Install dependencies for model work

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost joblib matplotlib seaborn wandb
```

### 2) Install dependencies for web backend

```bash
pip install -r website/backend/requirements.txt
```

## End-to-End Workflow

### Step 1: Build processed dataset

```bash
python feature_engineering.py
```

Expected output:

- `processed_data.csv`

### Step 2: Train and export model

Open and run all cells in:

- `model-training.ipynb`

Expected outputs:

- `aqi_model.pkl` (best model)
- `demo_forecast_data.csv` (test set with actual/predicted labels)

### Step 3: Threshold tuning

```bash
python tests/threshold_test.py
```

or

```bash
python tests/run_threshold.py
```

This prints baseline metrics, threshold comparison table, best threshold by macro F1, and confusion matrix.

## WandB Logging

WandB logging is integrated in both:

- `model-training.ipynb`
- `tests/run_threshold.py` and `tests/threshold_test.py`

### One-time login

```bash
wandb login
```

### Run threshold tuning with WandB

PowerShell:

```powershell
$env:USE_WANDB="1"
$env:WANDB_PROJECT="dat301m-air-quality"
$env:WANDB_RUN_NAME="threshold-tuning"
python tests/run_threshold.py
```

If you want to disable logging:

```powershell
$env:USE_WANDB="0"
python tests/run_threshold.py
```

## Run the Web Demo

From project root:

```bash
python website/backend/app.py
```

Open:

- `http://localhost:5000`

Useful API endpoints:

- `GET /api/health`
- `GET /api/predict`
- `GET /api/current`
- `POST /api/day-period`

## Notes

- SMOTE is applied only to training data to avoid leakage.
- `predictor.py` uses a tuned `POOR_THRESHOLD = 0.30` for the shift-based classifier.
- The repository still contains older regression artifacts for reference, but current presentation/training focus is classification.
