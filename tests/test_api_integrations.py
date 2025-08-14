# tests/test_api_integrations.py
import pandas as pd
import os
import sys

# --- The Fix: Add project root to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from api_integrations.api_clients import get_weather_temp_on_date

def test_get_weather_temp_on_date():
    """Tests the Open-Meteo API client for a valid response."""
    # Use a date in the past to ensure data availability
    date = pd.Timestamp("2023-08-14")
    temp = get_weather_temp_on_date(date)
    
    # Check that the temperature is a float and within a reasonable range
    assert isinstance(temp, float)
    assert 0 < temp < 50 # A reasonable temperature range for Ahmedabad