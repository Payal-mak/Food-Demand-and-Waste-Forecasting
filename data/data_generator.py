# data/data_generator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.config import SYNTHETIC_DATA_PATH

def generate_synthetic_data(n_weeks=104, start_date="2022-01-02"):
    """
    Generates two years of weekly synthetic demand and waste data.
    """
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W")
    demand = np.random.poisson(lam=100, size=n_weeks)
    waste = demand * np.random.uniform(0.05, 0.15, size=n_weeks)
    data = {
        "week": dates,
        "num_orders": demand,
        "waste_kg": waste.round(2)
    }
    df = pd.DataFrame(data)
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Synthetic data generated in: {SYNTHETIC_DATA_PATH}")
    print(df.head())
