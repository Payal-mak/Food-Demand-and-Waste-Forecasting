# evaluate_model.py
import pandas as pd
import numpy as np
import os
import sys

# --- Add project root to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from prediction.prediction_system import PredictionSystem
from utils.visualization import plot_pred_vs_actual
from config.config import SYNTHETIC_DATA_PATH, LOOKBACK_PERIOD

print("Initializing Prediction System...")
prediction_system = PredictionSystem()

print("Loading and preparing data for evaluation...")
df = pd.read_csv(SYNTHETIC_DATA_PATH, parse_dates=['week'])

# The features our model expects
feature_columns = ['num_orders', 'waste_kg', 'is_holiday', 'temperature', 'fuel_price', 'inflation_index']
# Ensure all required features are present, adding placeholders if necessary
# This part is to make sure the evaluation script runs even if the synthetic data changes
if 'is_holiday' not in df.columns: df['is_holiday'] = 0
if 'temperature' not in df.columns: df['temperature'] = 30.0
if 'fuel_price' not in df.columns: df['fuel_price'] = np.random.uniform(95, 110, size=len(df))
if 'inflation_index' not in df.columns: df['inflation_index'] = np.random.uniform(5, 8, size=len(df))

features_df = df[feature_columns]

# Make predictions for the whole dataset
all_predictions = []
# Start from LOOKBACK_PERIOD to have enough data for the first prediction
for i in range(LOOKBACK_PERIOD, len(features_df)):
    input_sequence = features_df.iloc[i-LOOKBACK_PERIOD:i]
    demand, waste = prediction_system.predict(input_sequence)
    all_predictions.append([demand, waste])

predictions_array = np.array(all_predictions)

# Get the actual values that correspond to the predictions
actuals_df = df.iloc[LOOKBACK_PERIOD:]

# Plot the results
print("Generating plots...")
plot_pred_vs_actual(actuals_df['num_orders'].values, predictions_array[:, 0], ylabel='Predicted vs Actual Demand')
plot_pred_vs_actual(actuals_df['waste_kg'].values, predictions_array[:, 1], ylabel='Predicted vs Actual Waste')