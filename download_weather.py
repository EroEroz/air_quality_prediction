"""
Download historical hourly weather data for HCMC from Open-Meteo API.
Free, no API key needed.
"""
import pandas as pd
import requests
import time
import os

# HCMC coordinates
LATITUDE = 10.8231
LONGITUDE = 106.6297

# Date range - extended to 2026 for full coverage
START_DATE = "2018-01-01"
END_DATE = "2026-01-18"

OUTPUT_FILE = r"data\new\hcmc_weather.csv"

# Hourly variables to download
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
]

def download_weather():
    print("Downloading HCMC weather data from Open-Meteo...")
    print(f"  Location: {LATITUDE}°N, {LONGITUDE}°E")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Variables: {HOURLY_VARS}")

    # Open-Meteo has a limit per request, so we split by year
    all_dfs = []
    for year in range(2018, 2027):
        sd = f"{year}-01-01"
        ed = min(f"{year}-12-31", END_DATE)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": sd,
            "end_date": ed,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "Asia/Ho_Chi_Minh",
        }
        
        print(f"  Fetching {year}...")
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        all_dfs.append(df)
        time.sleep(1)  # Be polite to the API

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.rename(columns={"time": "datetime"}, inplace=True)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    df_all = df_all.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_all.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSaved {len(df_all)} rows to {OUTPUT_FILE}")
    print(f"Columns: {df_all.columns.tolist()}")
    print(f"Date range: {df_all['datetime'].min()} to {df_all['datetime'].max()}")
    print(f"Missing values:\n{df_all.isna().sum()}")

if __name__ == "__main__":
    download_weather()
