# training/model_trainer.py
import pandas as pd
import pickle
import os
import sys

# --- Add project root to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from preprocessing.data_pipeline import preprocess_data, create_sequences
from models.model_architectures import build_lstm_multivariate
from config.config import SYNTHETIC_DATA_PATH, LOOKBACK_PERIOD, BATCH_SIZE
from tensorflow.keras.optimizers import Adam

# --- Use the BEST parameters from your tuning study ---
OPTIMAL_LSTM_UNITS = 38
OPTIMAL_DROPOUT_RATE = 0.366
OPTIMAL_LEARNING_RATE = 0.00011
# We can increase epochs now that we have good parameters
OPTIMAL_EPOCHS = 100 
# ---------------------------------------------------------


def train_model():
    """
    Main function to orchestrate the training of the FINAL, OPTIMIZED model.
    """
    print("Starting final model training with optimal hyperparameters...")

    # 1. Load Data
    print("Loading data from {}".format(SYNTHETIC_DATA_PATH))
    df = pd.read_csv(SYNTHETIC_DATA_PATH, parse_dates=['week'])

    # 2. Preprocess Data and Create Sequences
    print("Preprocessing data and creating sequences...")
    X_scaled, y_scaled, scaler_X, scaler_y = preprocess_data(df)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback=LOOKBACK_PERIOD)
    y_train_list = [y_seq[:, 0], y_seq[:, 1]]

    # 3. Build Model with Optimal Architecture
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    model = build_lstm_multivariate(
        input_shape,
        lstm_units=OPTIMAL_LSTM_UNITS,
        dropout_rate=OPTIMAL_DROPOUT_RATE
    )

    # 4. Compile Model with Optimal Optimizer
    optimizer = Adam(learning_rate=OPTIMAL_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss={'demand': 'mse', 'waste': 'mse'})
    print("Model Summary with Optimal Parameters:")
    model.summary()

    # 5. Train the Final Model
    print("Training final LSTM model...")
    model.fit(X_seq, y_train_list, epochs=OPTIMAL_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # 6. Save Final Model and Scalers
    print("Saving final model and scalers...")
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    model.save('saved_models/lstm_model.keras') # Use the .keras format
    with open('saved_models/scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('saved_models/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    print("Final model training complete and artifacts saved.")

if __name__ == "__main__":
    train_model()