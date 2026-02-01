# HCMC Air Quality PM2.5 Forecasting

> Predicting PM2.5 pollution levels using weather features and tree/sequence models. **Best Model: R² ≈ 0.66, MAE ≈ 5.45 µg/m³**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## Quick Results
| Model | MAE | R² | Status |
|-------|-----|-------|--------|
| **Ensemble (Optimized)** | ~5.45 | **~0.659** | Best Overall |
| **CatBoost** | 5.45 | **0.6596** | Best Single Model |
| XGBoost | 5.49 | 0.656 | Strong Baseline |
| Bi-LSTM | 5.53 | 0.648 | Reference |
| Linear Reg | 5.68 | 0.630 | Baseline |

**Improvement**: ~+4.5% from baseline through weather integration + tree models + ensemble methods

## Data & Features (27 total)
- **PM2.5**: Lags (1h, 2h, 3h, 24h, 168h) + rolling mean/std
- **Weather**: Temperature, humidity, wind speed, precipitation
- **Engineered**: Lag interactions, cyclic time encoding
- **Period**: 2018-2022 HCMC, hourly resolution, ~34k clean records

## Models Evaluated
1. Linear Regression (baseline)
2. LSTM (flattened features)
3. Bi-LSTM (temporal direction)
4. XGBoost
5. CatBoost
6. Optimized weighted ensemble
7. LSTM with sequence windows

## Key Findings
- Tree models outperform sequence models on this tabular setup
- CatBoost slightly beats XGBoost as a single model
- Optimized ensemble provides the best overall score
- Diminishing returns above R² ~0.66 without more data or richer features

## Files
- `model_comparison.ipynb` - All models, feature importance, visualizations
- `airq_analyze.ipynb` - Data exploration and visualization
- `process_data.py` - Feature engineering pipeline (27 features)
- `/data/hcmc_lstm_ready_weather.csv` - Final dataset

## Next Steps
- Add more data (other cities + `city` feature)
- Add external signals (traffic, holidays)
- Extend time period
