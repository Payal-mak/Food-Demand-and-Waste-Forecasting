# data/data_validator.py

import pandas as pd
import numpy as np

def validate_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['week'])
    errors = []

    # Check for missing values
    if df.isnull().any().any():
        errors.append("Data contains missing values.")

    # Check for negative values in orders or waste
    if (df['num_orders'] < 0).any():
        errors.append("Negative values found in num_orders.")
    if (df['waste_kg'] < 0).any():
        errors.append("Negative values found in waste_kg.")

    # Check for duplicates
    if df.duplicated(subset=["week"]).any():
        errors.append("Duplicate weeks found.")

    if errors:
        print("Validation Errors:")
        for err in errors:
            print("-", err)
        return False
    else:
        print("Data validation passed.")
        return True

if __name__ == "__main__":
    from config.config import SYNTHETIC_DATA_PATH
    validate_data(SYNTHETIC_DATA_PATH)
