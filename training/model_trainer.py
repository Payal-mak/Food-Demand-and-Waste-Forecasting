import pandas as pd
import pickle
import os
from tensorflow.keras.models import Model

# Import your project-specific modules
from preprocessing.data_pipeline import preprocess_data, create_sequences
from models.model_architectures import build_lstm_multivariate
from config.config import SYNTHETIC_DATA_PATH, LOOKBACK_PERIOD, EPOCHS, BATCH_SIZE

def train_model():
    """
    Main function to orchestrate the model training pipeline.
    """
    print("Starting model training...")

    # 1. Load Data
    print(f"Loading data from {SYNTHETIC_DATA_PATH}")
    df = pd.read_csv(SYNTHETIC_DATA_PATH, parse_dates=['week'])

    # 2. Preprocess Data and Create Sequences
    print("Preprocessing data and creating sequences...")
    X_scaled, y_scaled, scaler_X, scaler_y = preprocess_data(df)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback=LOOKBACK_PERIOD)

    # 3. Build Model Architecture
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    model = build_lstm_multivariate(input_shape)
    model.compile(optimizer='adam', loss='mse')
    print("Model Summary:")
    model.summary()

    # 4. Train Model
    print("Training LSTM model...")
    # Keras expects a list of arrays for multi-output models
    y_train_list = [y_seq[:, 0], y_seq[:, 1]]
    model.fit(X_seq, y_train_list, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # 5. Save Model and Scalers
    print("Saving model and scalers...")
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # Change this line
    model.save('saved_models/lstm_model.keras') # Use the new .keras format
    with open('saved_models/scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('saved_models/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    print("Training complete and artifacts saved.")

if __name__ == "__main__":
    train_model()