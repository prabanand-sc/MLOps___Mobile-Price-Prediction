import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Example feature engineering"""
    df["battery_to_weight"] = df["battery_power"] / (df["ram"] + 1)
    df["px_area"] = df["px_height"] * df["px_width"]
    return df
