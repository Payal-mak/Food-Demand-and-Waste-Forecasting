import numpy as np
import pickle
from tensorflow.keras.models import load_model
from config.config import LOOKBACK_PERIOD

class PredictionSystem:
    # Change the default path in the function definition
    def __init__(self, model_path='saved_models/lstm_model.keras', scaler_x_path='saved_models/scaler_X.pkl', scaler_y_path='saved_models/scaler_y.pkl'):
        """
        Initializes the prediction system by loading the trained model and scalers.
        """
        print("Loading prediction system...")
        self.model = load_model(model_path) # This will now load the .keras file
        with open(scaler_x_path, 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            self.scaler_y = pickle.load(f)
        print("Prediction system loaded successfully.")

    def predict(self, input_data_df):
        """
        Makes a prediction for demand and waste.
        Args:
            input_data_df (pd.DataFrame): A DataFrame containing the last 10 weeks of data
                                         with the same features used in training.
        Returns:
            A tuple containing the predicted demand and waste.
        """
        if len(input_data_df) < LOOKBACK_PERIOD:
            raise ValueError(f"Input data must contain at least {LOOKBACK_PERIOD} rows for lookback.")

        # Use the last 'lookback' period of rows
        input_sequence_df = input_data_df.tail(LOOKBACK_PERIOD)
        
        # Scale the input features
        scaled_input = self.scaler_X.transform(input_sequence_df)
        
        # Reshape for the model (1 sample, lookback_period steps, num_features)
        reshaped_input = np.reshape(scaled_input, (1, scaled_input.shape[0], scaled_input.shape[1]))
        
        # Get the scaled prediction from the model
        scaled_pred = self.model.predict(reshaped_input)
        
        # The output is a list of two arrays [demand_pred, waste_pred]
        # We need to reshape it to (1, 2) for the inverse scaler
        pred_reshaped = np.array(scaled_pred).T.reshape(1, -1)

        # Inverse transform the prediction to get the actual values
        final_prediction = self.scaler_y.inverse_transform(pred_reshaped)
        
        predicted_demand = final_prediction[0][0]
        predicted_waste = final_prediction[0][1]
        
        return predicted_demand, predicted_waste