# ======================================================================================
# Combined Project Prototype: Unified Demand & Waste Forecasting with Exogenous Variables
#
# NOTE: This script is a single-file prototype. The final project is organized
# into a modular structure with separate directories for data, models, training, etc.
# This file is for demonstration and conceptual understanding of the end-to-end workflow.
# ======================================================================================

# STEP 1: Imports
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from datetime import datetime

# --- Corrected & Centralized Functions ---

def get_weather_temp_on_date(date, city="Ahmedabad"):
    """
    Corrected: Returns average temperature for a specific date using the Open-Meteo API.
    This version does not require an API key.
    """
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
        print(f"Open-Meteo API error on {date_str}: {e}")
        return 30.0  # Fallback to a reasonable default if API fails

def is_holiday(date):
    """Checks if a given date is a holiday."""
    # In a real project, this would use a library like 'holidays'
    holidays = ['2022-10-24', '2022-12-25', '2023-01-26']
    return int(date.strftime('%Y-%m-%d') in holidays)


# --- Main Workflow ---

# STEP 2: Load Dataset
print("Loading data...")
# Assuming 'meal_demand_waste_data.csv' is in the same directory.
# In the full project, this path comes from config.py.
df = pd.read_csv('data/meal_demand_waste_data.csv')
df['week'] = pd.to_datetime(df['week'])

# STEP 3: Add External Features
print("Adding external features...")
df['is_holiday'] = df['week'].apply(is_holiday)
df['temperature'] = df['week'].apply(lambda date: get_weather_temp_on_date(date))

# Placeholder features for fuel price & inflation (as in the original)
np.random.seed(42) # for reproducibility
df['fuel_price'] = np.random.uniform(95, 110, size=len(df))
df['inflation_index'] = np.random.uniform(5, 8, size=len(df))

print("Data with features:")
print(df.head())

# STEP 4: Feature Engineering & Scaling
print("Scaling features and targets...")
features = df[['num_orders', 'waste_kg', 'is_holiday', 'temperature', 'fuel_price', 'inflation_index']]
targets = df[['num_orders', 'waste_kg']]

# Initialize scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform the data
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(targets)

# STEP 5: Create Sequences for LSTM
print("Creating time-series sequences...")
lookback = 10
X_seq, y_seq = [], []
for i in range(lookback, len(X_scaled)):
    X_seq.append(X_scaled[i-lookback:i])
    y_seq.append(y_scaled[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)
print(f"Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

# STEP 6: Build Multi-output LSTM Model
print("Building LSTM model...")
input_layer = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
lstm_layer = LSTM(64, return_sequences=False)(input_layer)
dropout_layer = Dropout(0.3)(lstm_layer)

# Two separate output layers (regression heads)
demand_output = Dense(1, name='demand')(dropout_layer)
waste_output = Dense(1, name='waste')(dropout_layer)

model = Model(inputs=input_layer, outputs=[demand_output, waste_output])
model.compile(
    optimizer='adam',
    loss={'demand': 'mse', 'waste': 'mse'},
    loss_weights={'demand': 1.0, 'waste': 0.5} # Example: prioritize demand accuracy
)
model.summary()

# STEP 7: Train Model
print("Training model...")
# For a multi-output model, provide targets as a list of arrays
y_train_list = [y_seq[:, 0], y_seq[:, 1]]
model.fit(X_seq, y_train_list, epochs=50, batch_size=32, verbose=1)

# STEP 8: Predict on the Last Sequence
print("\nMaking a sample prediction on the last available data...")
# Get the last sequence from the dataset to use as prediction input
last_sequence = np.expand_dims(X_seq[-1], axis=0)

# Get scaled predictions
scaled_pred = model.predict(last_sequence)

# Reshape for inverse transform: from [array(demand), array(waste)] to [[demand, waste]]
pred_reshaped = np.array(scaled_pred).T.reshape(1, -1)

# Inverse transform to get the final values
final_prediction = scaler_y.inverse_transform(pred_reshaped)

print("-" * 30)
print(f"Predicted Demand (Orders): {final_prediction[0][0]:.2f}")
print(f"Predicted Waste (kg): {final_prediction[0][1]:.2f}")
print("-" * 30)