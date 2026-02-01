import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "data/hcmc_pm25_weather_2018_2022.csv"  # NEW: Weather-enriched dataset
OUTPUT_FILE = "data/hcmc_lstm_ready_weather.csv"     # NEW: Output with weather features

# Physics Constraints
MIN_PM25 = 0.0
MAX_PM25 = 500.0

def run_pipeline():
    print("==============================================")
    print("   HCMC AIR QUALITY PIPELINE      ")
    print("==============================================")

    # ---------------------------------------------------------
    # STEP 1: LOAD
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Convert datetime
    df['dt'] = pd.to_datetime(df['datetime'])
    df = df.set_index('dt').sort_index()
    
    # Select columns (pm25 + weather features)
    # Keep: pm25, temperature, humidity, wind_speed, wind_direction, precipitation
    value_cols = ['pm25', 'temperature', 'humidity', 'wind_speed', 'wind_direction', 'precipitation']
    available_cols = [col for col in value_cols if col in df.columns]
    df = df[available_cols]
    
    # Rename pm25 to value for consistency
    df = df.rename(columns={'pm25': 'value'})
    
    print(f"   > Loaded {len(df)} rows.")
    print(f"   > Available features: {list(df.columns)}")

    # ---------------------------------------------------------
    # STEP 2: CLEANING
    # ---------------------------------------------------------
    print("\n[Step 2] Cleaning...")
    
    # Remove Duplicates & Physics Check
    df = df[~df.index.duplicated(keep='first')]
    df.loc[df['value'] < MIN_PM25, 'value'] = np.nan
    df.loc[df['value'] > MAX_PM25, 'value'] = np.nan
    
    # Resample to Hourly Grid
    df = df.resample('h').mean(numeric_only=True)
    
    # Reindex (Create perfect grid)
    full_idx = pd.date_range(
        start=df.index.min(), 
        end=df.index.max(), 
        freq='h', 
        tz=df.index.tz
    )
    df = df.reindex(full_idx)
    df.index.name = 'datetime'
    
    # Interpolate
    df['value'] = df['value'].interpolate(method='linear', limit=4)
    
    # Drop rows where 'value' is still NaN
    df = df.dropna(subset=['value'])
    
    print(f"   > Valid Hours after cleaning: {len(df)}")
    
    # DEBUG CHECK: Are there any NaNs left?
    if df.isna().sum().sum() > 0:
        print("   WARNING: NaNs found in dataset before Feature Engineering!")
        print(df.isna().sum())

    # ---------------------------------------------------------
    # STEP 3: FEATURES
    # ---------------------------------------------------------
    print("\n[Step 3] Generating Features...")
    
    # === PM2.5 LAG FEATURES ===
    df['lag_1h'] = df['value'].shift(1)
    df['lag_2h'] = df['value'].shift(2)
    df['lag_3h'] = df['value'].shift(3)
    df['lag_24h'] = df['value'].shift(24)
    df['lag_168h'] = df['value'].shift(168)
    
    # === PM2.5 ROLLING FEATURES ===
    df['roll_mean_24h'] = df['value'].rolling(window=24).mean()
    df['roll_std_24h']  = df['value'].rolling(window=24).std()
    
    # === WEATHER LAG FEATURES ===
    if 'temperature' in df.columns:
        df['temp_lag_1h'] = df['temperature'].shift(1)
        df['temp_lag_24h'] = df['temperature'].shift(24)
    
    if 'humidity' in df.columns:
        df['humid_lag_1h'] = df['humidity'].shift(1)
        df['humid_lag_24h'] = df['humidity'].shift(24)
    
    if 'wind_speed' in df.columns:
        df['wind_lag_1h'] = df['wind_speed'].shift(1)
        df['wind_lag_24h'] = df['wind_speed'].shift(24)
    
    # === INTERACTION FEATURES ===
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humid_interaction'] = df['temperature'] * df['humidity'] / 100
    
    if 'wind_speed' in df.columns and 'precipitation' in df.columns:
        df['wind_precip_interaction'] = df['wind_speed'] * (df['precipitation'] + 0.1)
    
    # === TIME FEATURES ===
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Count rows before final drop
    rows_before = len(df)
    
    # Final Drop (Removes the first 168 hours of undefined lags)
    df_final = df.dropna()
    
    rows_after = len(df_final)
    print(f"   > Rows before drop: {rows_before}")
    print(f"   > Rows after drop:  {rows_after}")
    
    if rows_after == 0:
        print("\n ERROR: All rows were dropped!")
        print("   Checking which column has NaNs:")
        print(df.isna().sum()) 
    else:
        # Save - dynamically include all generated features
        base_cols = [
            'value', 
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
            'roll_mean_24h', 'roll_std_24h',
            'hour_sin', 'hour_cos', 
            'day_sin', 'day_cos', 
            'month_sin', 'month_cos'
        ]
        
        # Add weather features that exist
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'precipitation',
                       'temp_lag_1h', 'temp_lag_24h', 'humid_lag_1h', 'humid_lag_24h',
                       'wind_lag_1h', 'wind_lag_24h', 'temp_humid_interaction', 'wind_precip_interaction']
        
        # Only include columns that actually exist in the dataframe
        final_cols = base_cols + [col for col in weather_cols if col in df_final.columns]
        
        df_final = df_final[final_cols].reset_index()
        df_final['datetime'] = df_final['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\n SUCCESS! Saved to {OUTPUT_FILE}")
        print(f"   > Total features: {len(final_cols)}")
        print(f"   > Features: {final_cols[:5]}... (+{len(final_cols)-5} more)")

if __name__ == "__main__":
    run_pipeline()