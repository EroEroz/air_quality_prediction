import pandas as pd
import numpy as np

# --- CONFIGURATION ---
<<<<<<< Updated upstream
INPUT_FILE = "data\NYC Data\Air_Quality.csv" 
OUTPUT_FILE = "data\NYC Data\Air_Quality_lstm_ready.csv"
=======
INPUT_FILE = r"data\new\hcmc_weather_and_aqi.csv"
WEATHER_FILE = r"data\hcmc_weather.csv"
OUTPUT_LINEAR_FILE = r"data\Clean For Model\Air_Quality_linear_ready2.csv"
OUTPUT_LSTM_FILE = r"data\Clean For Model\Air_Quality_lstm_ready2.csv"

WEATHER_COLS = [
    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
    'wind_direction_10m', 'surface_pressure', 'precipitation', 'cloud_cover'
]
>>>>>>> Stashed changes

# Additional Air Pollutants (besides PM2.5 which is the target)
AIR_POLLUTANT_COLS = [
    'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'
]

# Combined environmental features
ENVIRONMENTAL_COLS = WEATHER_COLS + AIR_POLLUTANT_COLS

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
    
    # --- AUTO-DETECT PM2.5 COLUMN ---
    # Support both 'value' and 'pm2_5' column names
    if 'pm2_5' in df.columns and 'value' not in df.columns:
        print("   > Detected 'pm2_5' column, renaming to 'value'")
        df['value'] = pd.to_numeric(df['pm2_5'], errors='coerce')
    elif 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    else:
        raise ValueError("Neither 'value' nor 'pm2_5' column found in INPUT_FILE")
    
    # --- FILTER OUT ROWS WITH MISSING PM2.5 ---
    rows_before_filter = len(df)
    df = df.dropna(subset=['value'])
    rows_after_filter = len(df)
    if rows_before_filter > rows_after_filter:
        print(f"   > Removed {rows_before_filter - rows_after_filter} rows with missing PM2.5")
        print(f"   > Date range with PM2.5: {df.index.min()} to {df.index.max()}")
    
    # --- CHECK IF ENVIRONMENTAL COLUMNS ALREADY PRESENT ---
    env_cols_present = [col for col in ENVIRONMENTAL_COLS if col in df.columns]
    
    if len(env_cols_present) >= len(WEATHER_COLS):
        # Environmental data already in file - keep them!
        print(f"   > Environmental data detected in INPUT_FILE ({len(env_cols_present)}/{len(ENVIRONMENTAL_COLS)} columns)")
        keep_cols = ['value'] + [col for col in ENVIRONMENTAL_COLS if col in df.columns]
        df = df[keep_cols]
        print(f"   > Keeping columns: {df.columns.tolist()}")
    else:
        # No environmental data - will merge later from WEATHER_FILE
        print(f"   > No environmental data in INPUT_FILE ({len(env_cols_present)}/{len(ENVIRONMENTAL_COLS)} columns)")
        df = df[['value']]
    
    print(f"   > Loaded {len(df)} rows with columns: {df.columns.tolist()}")

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
<<<<<<< Updated upstream
=======
    # STEP 2.5: MERGE ENVIRONMENTAL DATA (AUTO-DETECT)
    # ---------------------------------------------------------
    print("\n[Step 2.5] Checking Environmental Data...")
    
    # Check if environmental columns are already in the current dataframe
    env_cols_present = [col for col in ENVIRONMENTAL_COLS if col in df.columns]
    
    if len(env_cols_present) >= len(WEATHER_COLS):
        # Environmental data already present (loaded from INPUT_FILE in STEP 1)
        print(f"   ✓ Environmental data already present ({len(env_cols_present)}/{len(ENVIRONMENTAL_COLS)} columns)")
        print(f"   > Columns available: {env_cols_present}")
        print(f"   > Skipping WEATHER_FILE merge...")
        
        # Interpolate any small gaps in environmental data
        for col in env_cols_present:
            df[col] = df[col].interpolate(method='linear', limit=6)
        
        env_nans = df[env_cols_present].isna().sum()
        print(f"   > Environmental features ready. NaNs remaining:\n{env_nans}")
        
    else:
        # Environmental data NOT in dataframe, load from WEATHER_FILE (weather only)
        print(f"   ✗ Environmental data not found in INPUT_FILE ({len(env_cols_present)}/{len(ENVIRONMENTAL_COLS)} columns)")
        print(f"   > Loading weather from WEATHER_FILE: {WEATHER_FILE}")
        
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
        
        env_cols_present = WEATHER_COLS
    
    # Drop rows where essential environmental data is still missing
    df = df.dropna(subset=env_cols_present)
    print(f"   > Rows after environmental data processing: {len(df)}")

    # ---------------------------------------------------------
>>>>>>> Stashed changes
    # STEP 3: FEATURES
    # ---------------------------------------------------------
    print("\n[Step 3] Generating Features...")
    
    # Lags
    df['lag_1h'] = df['value'].shift(1)
    df['lag_2h'] = df['value'].shift(2)
    df['lag_3h'] = df['value'].shift(3)
    df['lag_24h'] = df['value'].shift(24)
    df['lag_168h'] = df['value'].shift(168)
    
    # Rolling
    df['roll_mean_24h'] = df['value'].rolling(window=24).mean()
    df['roll_std_24h']  = df['value'].rolling(window=24).std()
    
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
<<<<<<< Updated upstream
        # Save
        final_cols = [
            'value', 
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
=======
        # Detect which environmental columns are actually present
        env_cols_available = [col for col in ENVIRONMENTAL_COLS if col in df_final.columns]
        print(f"   > Environmental columns to save: {env_cols_available}")
        
        # Save linear dataset (keeps all engineered features + environmental data)
        linear_cols = [
            'value', 'target_24h_avg', 'target_category',
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
            'diff_1h', 'diff_24h',
            'roll_mean_6h', 'roll_std_6h',
            'roll_mean_12h', 'roll_std_12h',
>>>>>>> Stashed changes
            'roll_mean_24h', 'roll_std_24h',
            'hour_sin', 'hour_cos', 
            'day_sin', 'day_cos', 
<<<<<<< Updated upstream
            'month_sin', 'month_cos'
        ]
        
        df_final = df_final[final_cols].reset_index()
        df_final['datetime'] = df_final['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\n SUCCESS! Saved to {OUTPUT_FILE}")
=======
            'month_sin', 'month_cos',
            'is_dry_season'
        ] + env_cols_available
        
        df_linear = df_final[linear_cols].reset_index()
        df_linear['datetime'] = df_linear['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_linear.to_csv(OUTPUT_LINEAR_FILE, index=False)

        # Save LSTM dataset (raw value + rolling/ewm stats + time + environmental data)
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
        ] + env_cols_available
        df_lstm = df_final[lstm_cols].reset_index()
        df_lstm['datetime'] = df_lstm['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_lstm.to_csv(OUTPUT_LSTM_FILE, index=False)

        print(f"\n SUCCESS! Saved linear dataset to {OUTPUT_LINEAR_FILE}")
        print(f"          Columns: {len(linear_cols)} features + datetime")
        print(f" SUCCESS! Saved LSTM dataset to   {OUTPUT_LSTM_FILE}")
        print(f"          Columns: {len(lstm_cols)} features + datetime")
>>>>>>> Stashed changes

if __name__ == "__main__":
    run_pipeline()