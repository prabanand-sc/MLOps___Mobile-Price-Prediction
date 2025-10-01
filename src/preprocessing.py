import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data(train_path: str, test_path: str):
    """Load training and test datasets"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df, artifacts_dir="artifacts"):
    """Handle missing values, scaling, and return split datasets"""
    os.makedirs(artifacts_dir, exist_ok=True)

    # Separate features and target
    X = train_df.drop(columns=["price_range"])
    y = train_df["price_range"]

    # Save test IDs and drop
    test_ids = test_df["id"]
    X_test = test_df.drop(columns=["id"])

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X_test_imputed = imputer.transform(X_test)
    joblib.dump(imputer, os.path.join(artifacts_dir, "num_imputer.pkl"))

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_val, y_train, y_val, X_test_scaled, test_ids
