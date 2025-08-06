# api_integrations/api_clients.py

import requests
from config.config import WEATHER_API_URL, WEATHER_API_KEY

def get_weather_temp_on_date(date, city="Ahmedabad"):
    """
    Returns average temperature for a specific date and city using WeatherAPI.
    """
    date_str = date.strftime('%Y-%m-%d')
    params = {
        'key': WEATHER_API_KEY,
        'q': city,
        'dt': date_str
    }
    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data['forecast']['forecastday'][0]['day']['avgtemp_c']
    except Exception as e:
        print(f"Weather API error ({date_str}): {e}")
        return 30.0  # Fall back to mean if API fails

# Example usage
if __name__ == "__main__":
    import pandas as pd
    date = pd.Timestamp("2023-08-06")
    temp = get_weather_temp_on_date(date)
    print(f"Avg temperature in Ahmedabad on {date.date()}: {temp}°C")
