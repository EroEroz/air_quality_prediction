import requests
import pandas as pd
from datetime import datetime, timedelta

# Open-Meteo API (FREE, no key needed)
# Get historical weather data for Ho Chi Minh City

def download_weather_data(start_date, end_date):
    """
    Download weather data for HCMC from Open-Meteo API
    Lat: 10.8231, Lon: 106.6297 (HCMC coordinates)
    """
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": 10.8231,
        "longitude": 106.6297,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "timezone": "Asia/Ho_Chi_Minh"
    }
    
    print(f"Downloading weather data from {start_date} to {end_date}...")
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Parse the hourly data
        hourly = data.get('hourly', {})
        
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'temperature': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'wind_speed': hourly['wind_speed_10m'],
            'wind_direction': hourly['wind_direction_10m'],
            'precipitation': hourly['precipitation']
        })
        
        print(f"Downloaded {len(df)} hourly records")
        return df
        
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Download for 2018-2022 to match your PM2.5 data
    weather_df = download_weather_data("2018-01-01", "2022-12-31")
    
    if not weather_df.empty:
        print("\nWeather data sample:")
        print(weather_df.head())
        print(f"\nShape: {weather_df.shape}")
        print(f"\nMissing values:")
        print(weather_df.isnull().sum())
        
        # Save
        output_file = "data/hcmc_weather_2018_2022.csv"
        weather_df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
        # Now merge with PM2.5 data
        pm25_file = "data/hcmc_full_2018_2022_multiparameter.csv"
        pm25_df = pd.read_csv(pm25_file)
        pm25_df['datetime'] = pd.to_datetime(pm25_df['datetime'], utc=True)
        
        # Remove timezone from both to merge
        pm25_df['datetime'] = pm25_df['datetime'].dt.tz_localize(None)
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_localize(None)
        
        # Merge on datetime
        merged_df = pd.merge(pm25_df, weather_df, on='datetime', how='left')
        
        print(f"\n=== MERGED DATA ===")
        print(f"Shape: {merged_df.shape}")
        print(f"\nColumns: {list(merged_df.columns)}")
        print(f"\nSample:")
        print(merged_df.head())
        
        # Save merged data
        merged_file = "data/hcmc_pm25_weather_2018_2022.csv"
        merged_df.to_csv(merged_file, index=False)
        print(f"\nSaved merged data to: {merged_file}")
    else:
        print("Failed to download weather data")
