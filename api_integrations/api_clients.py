import requests
from datetime import datetime
import pandas as pd

def get_weather_temp_on_date(date, city="Ahmedabad"):
    """
    Returns average temperature for a specific date and city using Open-Meteo.
    """
    # Default is Ahmedabad. If needed, add logic here to map city to lat/lon.
    latitude = 23.0225
    longitude = 72.5714
    date_str = date.strftime('%Y-%m-%d')
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&start_date={date_str}&end_date={date_str}&"
        "daily=temperature_2m_max,temperature_2m_min&timezone=Asia/Kolkata"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        daily = data['daily']
        avg_temp = (daily['temperature_2m_max'][0] + daily['temperature_2m_min'][0]) / 2
        return avg_temp
    except Exception as e:
        print(f"Open-Meteo API error ({date_str}): {e}")
        return 30.0 # Fall back to mean if API fails

# Example usage
if __name__ == '__main__':
    date_to_check = pd.Timestamp("2023-08-14")
    temp = get_weather_temp_on_date(date_to_check)
    print(f"Average temperature in Ahmedabad on {date_to_check.date()}: {temp}°C")