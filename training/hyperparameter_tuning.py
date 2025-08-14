# training/hyperparameter_tuning.py
import optuna
import pandas as pd
import numpy as np
import os
import sys

# --- Add project root to the Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from preprocessing.data_pipeline import preprocess_data, create_sequences
from models.model_architectures import build_lstm_multivariate
from config.config import SYNTHETIC_DATA_PATH, LOOKBACK_PERIOD
from tensorflow.keras.optimizers import Adam

# 1. Load and prepare data once
print("Loading and preparing data for tuning...")
df = pd.read_csv(SYNTHETIC_DATA_PATH, parse_dates=['week'])
X_scaled, y_scaled, _, _ = preprocess_data(df)
X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback=LOOKBACK_PERIOD)
y_train_list = [y_seq[:, 0], y_seq[:, 1]]

def objective(trial):
    """
    This is the main function for Optuna.
    It defines a single trial of training and evaluating a model.
    """
    # 2. Suggest Hyperparameters
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    # 3. Build and Compile the Model
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    model = build_lstm_multivariate(input_shape, lstm_units=lstm_units, dropout_rate=dropout_rate)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss={'demand': 'mse', 'waste': 'mse'})

    # 4. Train the Model
    history = model.fit(
        X_seq,
        y_train_list,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # 5. Evaluate and Return the Final Validation Loss
    val_loss = history.history['val_loss'][-1]
    # --- CORRECTED PRINT STATEMENT ---
    print("Trial {} finished with validation loss: {:.4f}".format(trial.number, val_loss))
    return val_loss

# 6. Create and Run the Study
print("\nStarting hyperparameter tuning study...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 7. Print the Results
print("\nTuning study complete!")
print("Best trial:")
trial = study.best_trial
# --- CORRECTED PRINT STATEMENT ---
print("  Validation Loss: {:.4f}".format(trial.value))
print("  Best Parameters:")
for key, value in trial.params.items():
    # --- CORRECTED PRINT STATEMENT ---
    print("    {}: {}".format(key, value))