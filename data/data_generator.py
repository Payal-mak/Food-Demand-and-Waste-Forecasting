import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# --- The Fix: Add project root to the Python path ---
# This ensures that imports like 'from config.config...' work correctly
# regardless of where you run the script from.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# ----------------------------------------------------

from config.config import SYNTHETIC_DATA_PATH

def generate_synthetic_data(n_weeks=104, start_date="2022-01-02"):
    """
    Generates two years of weekly synthetic demand and waste data.
    """
    print(f"Attempting to save synthetic data to: {SYNTHETIC_DATA_PATH}")
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W")
    demand = np.random.poisson(lam=100, size=n_weeks)
    waste = demand * np.random.uniform(0.05, 0.15, size=n_weeks)
    data = {
        "week": dates,
        "num_orders": demand,
        "waste_kg": waste.round(2)
    }
    df = pd.DataFrame(data)

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(SYNTHETIC_DATA_PATH), exist_ok=True)
    
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Synthetic data successfully generated in: {SYNTHETIC_DATA_PATH}")
    print(df.head())