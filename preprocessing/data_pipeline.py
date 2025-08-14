import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# --- The Fix: Add project root to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from api_integrations.api_clients import get_weather_temp_on_date
from config.config import LOOKBACK_PERIOD

def add_holiday_flag(df):
    holidays = set(['2022-10-24', '2022-12-25', '2023-01-26'])  # Expand for more
    df['is_holiday'] = df['week'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)
    return df

def add_weather_feature(df):
    print("Fetching weather data for each week... (This may take a moment)")
    df['temperature'] = df['week'].apply(lambda x: get_weather_temp_on_date(x))
    return df

def add_placeholder_features(df):
    np.random.seed(42)  # Reproducibility
    df['fuel_price'] = np.random.uniform(95, 110, size=len(df))
    df['inflation_index'] = np.random.uniform(5, 8, size=len(df))
    return df

def preprocess_data(df, scaler_X=None, scaler_y=None):
    # Feature engineering
    df = add_holiday_flag(df)
    df = add_weather_feature(df)
    df = add_placeholder_features(df)
    
    # Features and targets
    # Corrected: The features for scaling should match the input for the LSTM
    features_to_scale = ['num_orders', 'waste_kg', 'is_holiday', 'temperature', 'fuel_price', 'inflation_index']
    targets_to_scale = ['num_orders', 'waste_kg']

    X_data = df[features_to_scale]
    y_data = df[targets_to_scale]
    
    if not scaler_X:
        scaler_X = MinMaxScaler().fit(X_data)
    if not scaler_y:
        scaler_y = MinMaxScaler().fit(y_data)
        
    X_scaled = scaler_X.transform(X_data)
    # Corrected: y_scaled should also use the y_data
    y_scaled = scaler_y.transform(y_data)
    return X_scaled, y_scaled, scaler_X, scaler_y

def create_sequences(X, y, lookback=LOOKBACK_PERIOD):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Example usage
if __name__ == "__main__":
    from config.config import SYNTHETIC_DATA_PATH
    df = pd.read_csv(SYNTHETIC_DATA_PATH, parse_dates=['week'])
    X_scaled, y_scaled, sx, sy = preprocess_data(df)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled)
    print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")