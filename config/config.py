import os

# API keys (set your environment variables or replace with string literal for testing)
FUEL_API_KEY = os.getenv('FUEL_API_KEY', 'YOUR_FUEL_API_KEY_HERE')

# API endpoints
FUEL_API_URL = 'https://api.example.com/fuelprice'  # Replace with real API if available

# Data paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'meal_demand_waste_data.csv'))
SYNTHETIC_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_meal_demand_waste.csv'))

# Model parameters
LOOKBACK_PERIOD = 10
BATCH_SIZE = 32
EPOCHS = 50