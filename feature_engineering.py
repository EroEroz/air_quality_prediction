"""
Feature engineering pipeline for AQI shift-level classification.

Input:
    data/hcmc_weather_and_aqi.csv
Output:
    processed_data.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


RAW_COLUMNS = [
    "datetime",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
]

POLLUTANT_FEATURES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


def assign_shift(hour: int) -> str:
    """Map hour to Morning / Afternoon / Night shift."""
    if 6 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 17:
        return "Afternoon"
    return "Night"


def create_aqi_class(pm25_value: float) -> int:
    """Create AQI class from pm2_5 thresholds."""
    if pm25_value <= 12.0:
        return 0
    if pm25_value <= 35.4:
        return 1
    return 2


def create_lag_features(df: pd.DataFrame, lag_vars: list = None, lag_periods: list = None) -> pd.DataFrame:
    """
    Create lag features for time-series data with daily seasonality awareness.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe, should be sorted chronologically.
    lag_vars : list, optional
        Variables to create lags for. Default: ['pm2_5', 'temperature_2m', 'AQI_Class']
    lag_periods : list, optional
        Lag periods to create. 
        Default: [1, 2, 3, 6] where:
            - lag_1, lag_2: Recent shifts
            - lag_3: Same shift from previous day (3 shifts/day)
            - lag_6: Same shift from 2 days ago
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with lag features added, NaN rows dropped.
    """
    if lag_vars is None:
        lag_vars = ['pm2_5', 'temperature_2m', 'AQI_Class']
    if lag_periods is None:
        lag_periods = [1, 2, 3, 6]
    
    df = df.copy()
    
    # Create lag features
    for var in lag_vars:
        for lag in lag_periods:
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)
    
    # Drop rows with NaN introduced by lag shifting
    df = df.dropna().reset_index(drop=True)
    
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parent
    raw_path = project_root / "data" / "hcmc_weather_and_aqi.csv"
    output_path = project_root / "processed_data.csv"

    df = pd.read_csv(raw_path)

    # Keep only expected columns in consistent order.
    df = df[RAW_COLUMNS].copy()

    # Parse datetime and sort to ensure proper temporal interpolation.
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Interpolate missing values, then fill any remaining boundary gaps.
    numeric_cols = WEATHER_FEATURES + POLLUTANT_FEATURES
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
    df[numeric_cols] = df[numeric_cols].bfill().ffill()

    # Drop rows where pm2_5 is still NaN after filling.
    df = df.dropna(subset=["pm2_5"]).copy()

    # Create target class from pm2_5.
    df["AQI_Class"] = df["pm2_5"].apply(create_aqi_class).astype(int)

    # Time features.
    df["Date"] = df["datetime"].dt.date
    df["Hour"] = df["datetime"].dt.hour
    df["Shift"] = df["Hour"].apply(assign_shift)

    # Aggregate by Date and Shift.
    agg_spec = {
        **{col: "mean" for col in WEATHER_FEATURES},
        **{col: "max" for col in POLLUTANT_FEATURES},
        "AQI_Class": "max",
    }

    processed_df = (
        df.groupby(["Date", "Shift"], as_index=False)
        .agg(agg_spec)
        .sort_values(["Date", "Shift"])
        .reset_index(drop=True)
    )

    # Create lag features for time-series modeling (with daily seasonality awareness)
    # lag_3 = same shift from previous day, lag_6 = same shift from 2 days ago
    processed_df = create_lag_features(
        processed_df,
        lag_vars=['pm2_5', 'temperature_2m', 'AQI_Class'],
        lag_periods=[1, 2, 3, 6]
    )

    # Save processed dataset for training notebook.
    processed_df.to_csv(output_path, index=False)

    print(f"Saved processed data: {output_path}")
    print(f"Shape: {processed_df.shape}")
    print("Class distribution:")
    print(processed_df["AQI_Class"].value_counts().sort_index())


if __name__ == "__main__":
    main()
