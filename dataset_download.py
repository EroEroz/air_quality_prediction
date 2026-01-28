import requests
import pandas as pd
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
API_KEY = ""  # <--- PASTE YOUR OPENAQ API KEY HERE
if not API_KEY:
    raise ValueError("API_KEY is empty. Please add your OpenAQ API key before running.")
HEADERS = {"X-API-Key": API_KEY}
BASE_URL = "https://api.openaq.org/v3"

# We check BOTH known US Consulate sensors to fill the 2020-2023 gap
TARGET_SENSORS = [4681, 21631] 

def download_range(sensor_id, start_year, end_year):
    print(f"\nScanning Sensor {sensor_id} for data ({start_year}-{end_year})...")
    all_data = []
    
    # We loop by YEAR to avoid timeouts and manage large requests
    for year in range(start_year, end_year + 1):
        print(f"  > Checking Year {year}...")
        
        start_date = f"{year}-01-01T00:00:00Z"
        end_date = f"{year}-12-31T23:59:59Z"
        page = 1
        year_data_count = 0
        
        while True:
            url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
            params = {
                "datetime_from": start_date,
                "datetime_to": end_date,
                "limit": 1000,
                "page": page,
                "sort": "asc"
            }
            
            try:
                r = requests.get(url, headers=HEADERS, params=params)
                if r.status_code != 200:
                    break 
                
                data = r.json().get('results', [])
                if not data:
                    break
                
                all_data.extend(data)
                year_data_count += len(data)
                page += 1
                time.sleep(0.2) # Rate limit safety
                
            except Exception as e:
                print(f"    ! Error: {e}")
                break
        
        if year_data_count > 0:
            print(f"    + Found {year_data_count} records for {year}.")
        else:
            print(f"    - No data for {year}.")
            
    return pd.DataFrame(all_data)

# --- EXECUTION ---
print("--- STARTING UNIFIED DOWNLOAD (2020 - Present) ---")
dfs = []

# 1. Download from all candidate sensors
for sensor in TARGET_SENSORS:
    df_sensor = download_range(sensor, 2018, 2022)
    if not df_sensor.empty:
        # Ensure we have a datetime column
        if 'datetime' not in df_sensor.columns and 'period' in df_sensor.columns:
            df_sensor['datetime'] = df_sensor['period'].apply(
                lambda x: x.get('datetimeFrom', {}).get('utc') if isinstance(x, dict) else None
            )
        
        # Validate that we have valid datetime values
        if 'datetime' in df_sensor.columns:
            df_sensor = df_sensor.dropna(subset=['datetime'])
        
        # Add a column to track which sensor gave us this data (useful for debugging)
        df_sensor['source_sensor'] = sensor
        dfs.append(df_sensor)

# 2. Merge and Clean
if dfs:
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Convert to datetime objects
    full_df['datetime'] = pd.to_datetime(full_df['datetime'], errors='coerce')
    
    # Remove rows with invalid datetimes
    before_count = len(full_df)
    full_df = full_df.dropna(subset=['datetime'])
    invalid_count = before_count - len(full_df)
    if invalid_count > 0:
        print(f"Removed {invalid_count} rows with invalid datetimes.")
    
    # SORT by time
    full_df = full_df.sort_values('datetime')
    
    # REMOVE DUPLICATES
    # If both sensors covered the same hour, keep the one that came first
    before_dedup = len(full_df)
    full_df = full_df.drop_duplicates(subset=['datetime'], keep='first')
    after_dedup = len(full_df)
    
    print(f"\n--- MERGE COMPLETE ---")
    print(f"Total Raw Records: {before_count}")
    print(f"After datetime validation: {len(full_df) + invalid_count}")
    print(f"After deduplication: {after_dedup}")
    print(f"Time Range: {full_df['datetime'].min()} to {full_df['datetime'].max()}")
    
    # Save
    os.makedirs("data", exist_ok=True)
    filename = "data/hcmc_full_2018_2022.csv"
    full_df.to_csv(filename, index=False)
    print(f"Saved to: {filename}")
    
else:
    print("No data found from 2018-2022 on these sensors.")