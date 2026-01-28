import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "data\hcmc_full_2018_2022.csv" 
OUTPUT_FILE = "data\hcmc_lstm_ready.csv"

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
        # Save
        final_cols = [
            'value', 
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
            'roll_mean_24h', 'roll_std_24h',
            'hour_sin', 'hour_cos', 
            'day_sin', 'day_cos', 
            'month_sin', 'month_cos'
        ]
        
        df_final = df_final[final_cols].reset_index()
        df_final['datetime'] = df_final['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\n SUCCESS! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()