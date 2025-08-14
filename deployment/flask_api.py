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

app = Flask(__name__)

# Initialize the prediction system when the app starts
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

        # --- THE FIX IS HERE ---
        # Convert the numpy.float32 types to standard Python floats
        # before creating the JSON response.
        return jsonify({
            'predicted_demand': float(demand),
            'predicted_waste': float(waste)
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred. Please check the server logs.'}), 500

# --- This makes the script directly runnable ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)