from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import sys

# --- Add project root to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from prediction.prediction_system import PredictionSystem
# Corrected: Only import the functions you will call
from database.database_manager import save_prediction, create_table
from datetime import datetime

app = Flask(__name__)

# --- Run the setup function once to ensure the DB and table exist ---
create_table()
# --------------------------------------------------------------------


print("Initializing Prediction System. This may take a moment...")
prediction_system = PredictionSystem()
print("Prediction System ready.")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make a prediction.
    """
    try:
        json_data = request.get_json()
        if not json_data or 'features' not in json_data:
            return jsonify({'error': "Missing 'features' in request body"}), 400
            
        features = json_data['features']
        
        columns = ['num_orders', 'waste_kg', 'is_holiday', 'temperature', 'fuel_price', 'inflation_index']
        input_df = pd.DataFrame(features, columns=columns)

        demand, waste = prediction_system.predict(input_df)

        current_week = datetime.now().strftime('%Y-%m-%d')
        
        # --- CORRECTED DATABASE CALL ---
        # Pass the database path directly. The function will handle the connection.
        save_prediction('predictions.db', current_week, float(demand), float(waste))

        return jsonify({
            'predicted_demand': float(demand),
            'predicted_waste': float(waste)
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred. Please check the server logs.'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)