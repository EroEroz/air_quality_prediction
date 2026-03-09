import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = r"data\hcmc_cleaned_data.csv"
WEATHER_FILE = r"data\hcmc_weather.csv"
OUTPUT_LINEAR_FILE = r"data\Clean For Model\Air_Quality_linear_ready.csv"
OUTPUT_LSTM_FILE = r"data\Clean For Model\Air_Quality_lstm_ready.csv"

WEATHER_COLS = [
    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
    'wind_direction_10m', 'surface_pressure', 'precipitation', 'cloud_cover'
]

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
    
    # Fix Timezones
    if 'datetime' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime'], utc=True)
    elif 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], utc=True)
    elif 'period' in df.columns:
        df['dt'] = df['period'].apply(lambda x: eval(x).get('datetimeFrom', {}).get('utc') if isinstance(x, str) else x)
        df['dt'] = pd.to_datetime(df['dt'], utc=True)

    df['dt'] = df['dt'].dt.tz_convert('Asia/Ho_Chi_Minh')
    df = df.set_index('dt').sort_index()
    
    # Force 'value' to be numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # --- CRITICAL FIX 1: Drop EVERYTHING except 'value' ---
    # This prevents "Phantom Columns" (like latitude/ID) from causing NaNs later
    df = df[['value']]
    
    print(f"   > Loaded {len(df)} rows.")

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
    # STEP 2.5: MERGE WEATHER DATA
    # ---------------------------------------------------------
    print("\n[Step 2.5] Merging Weather Data...")
    df_weather = pd.read_csv(WEATHER_FILE)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather = df_weather.set_index('datetime')
    # Localize weather to HCMC timezone to match PM2.5 index
    df_weather.index = df_weather.index.tz_localize('Asia/Ho_Chi_Minh')
    
    # Merge on datetime index
    df = df.join(df_weather[WEATHER_COLS], how='left')
    
    # Interpolate any small gaps in weather data
    for col in WEATHER_COLS:
        df[col] = df[col].interpolate(method='linear', limit=6)
    
    weather_nans = df[WEATHER_COLS].isna().sum()
    print(f"   > Weather features merged. NaNs remaining:\n{weather_nans}")
    
    # Drop rows where weather data is still missing
    df = df.dropna(subset=WEATHER_COLS)
    print(f"   > Rows after weather merge: {len(df)}")

    # ---------------------------------------------------------
    # STEP 3: FEATURES
    # ---------------------------------------------------------
    print("\n[Step 3] Generating Features...")
    
    # Lags (use past values only — no leakage)
    df['lag_1h'] = df['value'].shift(1)
    df['lag_2h'] = df['value'].shift(2)
    df['lag_3h'] = df['value'].shift(3)
    df['lag_6h'] = df['value'].shift(6)
    df['lag_12h'] = df['value'].shift(12)
    df['lag_24h'] = df['value'].shift(24)
    df['lag_48h'] = df['value'].shift(48)
    df['lag_168h'] = df['value'].shift(168)
    
    # Diff features (shifted by 1 to prevent exact reconstruction:
    # without shift, diff_1h + lag_1h = value exactly → R²=1.0 leakage)
    df['diff_1h'] = df['value'].diff(1).shift(1)
    df['diff_24h'] = df['value'].diff(24).shift(1)
    
    # Rolling stats (includes current value — mild leak OK for smoothed targets)
    df['roll_mean_6h']  = df['value'].rolling(window=6).mean()
    df['roll_std_6h']   = df['value'].rolling(window=6).std()
    df['roll_mean_12h'] = df['value'].rolling(window=12).mean()
    df['roll_std_12h']  = df['value'].rolling(window=12).std()
    df['roll_mean_24h'] = df['value'].rolling(window=24).mean()
    df['roll_std_24h']  = df['value'].rolling(window=24).std()
    
    # EWM (includes current value)
    df['ewm_mean_12h'] = df['value'].ewm(span=12).mean()
    df['ewm_mean_24h'] = df['value'].ewm(span=24).mean()
    
    # Min/Max (includes current value)
    df['roll_min_24h'] = df['value'].rolling(window=24).min()
    df['roll_max_24h'] = df['value'].rolling(window=24).max()
    
    # Time Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # HCMC-specific: dry season (Dec-Apr) vs wet season
    df['is_dry_season'] = df['month'].isin([12, 1, 2, 3, 4]).astype(int)
    
    # --- TREND / MOMENTUM FEATURES (all from past values, no leakage) ---
    # 3h momentum: is PM2.5 rising or falling?
    df['momentum_3h'] = df['lag_1h'] - df['lag_3h']
    # 6h momentum
    df['momentum_6h'] = df['lag_1h'] - df['lag_6h']
    # Acceleration: is the trend speeding up or slowing down?
    df['acceleration'] = (df['lag_1h'] - df['lag_2h']) - (df['lag_2h'] - df['lag_3h'])
    # 24h volatility range
    df['roll_range_24h'] = df['roll_max_24h'] - df['roll_min_24h']
    # Ratio: how does lag_1h compare to 24h average? (spike detection)
    df['lag1_vs_mean24'] = df['lag_1h'] / (df['roll_mean_24h'] + 1e-6)
    
    # --- WEATHER INTERACTION FEATURES ---
    # These use current weather (exogenous, not leaked) * recent PM2.5 history
    df['lag1_x_humidity'] = df['lag_1h'] * df['relative_humidity_2m']
    df['lag1_x_windspeed'] = df['lag_1h'] * df['wind_speed_10m']
    df['lag1_x_temp'] = df['lag_1h'] * df['temperature_2m']
    # Weather-only interactions (physics-motivated)
    df['temp_x_humidity'] = df['temperature_2m'] * df['relative_humidity_2m']
    df['wind_x_precip'] = df['wind_speed_10m'] * df['precipitation']

    # --- FORECAST TARGETS ---
    # 1. Regression target: average PM2.5 over the NEXT 24 hours
    df['target_24h_avg'] = df['value'].rolling(24).mean().shift(-24)
    
    # 2. Classification target: AQI category
    # Good: < 12, Moderate: 12-35, Poor: >= 35
    # Initialize with -1 (will be filled after dropna)
    df['target_category'] = -1

    # Count rows before final drop
    rows_before = len(df)
    
    # Final Drop (Removes the first 168 hours of undefined lags)
    df_final = df.dropna()
    
    # Assign categories after dropna to avoid NaN issues
    df_final['target_category'] = pd.cut(
        df_final['target_24h_avg'], 
        bins=[-np.inf, 12, 35, np.inf],
        labels=[0, 1, 2],  # 0=Good, 1=Moderate, 2=Poor
        right=False
    ).astype(int)
    
    rows_after = len(df_final)
    print(f"   > Rows before drop: {rows_before}")
    print(f"   > Rows after drop:  {rows_after}")
    
    if rows_after == 0:
        print("\n ERROR: All rows were dropped!")
        print("   Checking which column has NaNs:")
        print(df.isna().sum()) 
    else:
        # Save linear dataset (keeps all engineered features + weather)
        linear_cols = [
            'value', 'target_24h_avg', 'target_category',
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
            'diff_1h', 'diff_24h',
            'roll_mean_6h', 'roll_std_6h',
            'roll_mean_12h', 'roll_std_12h',
            'roll_mean_24h', 'roll_std_24h',
            'ewm_mean_12h', 'ewm_mean_24h',
            'roll_min_24h', 'roll_max_24h',
            'momentum_3h', 'momentum_6h', 'acceleration',
            'roll_range_24h', 'lag1_vs_mean24',
            'lag1_x_humidity', 'lag1_x_windspeed', 'lag1_x_temp',
            'temp_x_humidity', 'wind_x_precip',
            'hour_sin', 'hour_cos', 
            'day_sin', 'day_cos', 
            'month_sin', 'month_cos',
            'is_dry_season'
        ] + WEATHER_COLS
        
        df_linear = df_final[linear_cols].reset_index()
        df_linear['datetime'] = df_linear['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_linear.to_csv(OUTPUT_LINEAR_FILE, index=False)

        # Save LSTM dataset (raw value + rolling/ewm stats + time + weather)
        lstm_cols = [
            'value', 'target_24h_avg', 'target_category',
            'roll_mean_6h', 'roll_std_6h',
            'roll_mean_12h', 'roll_std_12h',
            'roll_mean_24h', 'roll_std_24h',
            'ewm_mean_12h', 'ewm_mean_24h',
            'hour_sin', 'hour_cos',
            'day_sin', 'day_cos',
            'month_sin', 'month_cos',
            'is_dry_season'
        ] + WEATHER_COLS
        df_lstm = df_final[lstm_cols].reset_index()
        df_lstm['datetime'] = df_lstm['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_lstm.to_csv(OUTPUT_LSTM_FILE, index=False)

        print(f"\n SUCCESS! Saved linear dataset to {OUTPUT_LINEAR_FILE}")
        print(f" SUCCESS! Saved LSTM dataset to   {OUTPUT_LSTM_FILE}")

if __name__ == "__main__":
    run_pipeline()