from src.preprocessing import load_data, preprocess_data
from src.train_multiple import train_multiple
import joblib
import os

def main():
    # 1) Load data
    train_df, test_df = load_data("data/raw/train.csv", "data/raw/test.csv")

    # 2) Preprocess
    X_train, X_val, y_train, y_val, X_test, test_ids = preprocess_data(train_df, test_df)

    # 3) Train multiple models
    results = train_multiple(X_train, y_train, X_val, y_val)

    # 4) Select best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]

    # 5) Save best model locally
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/best_model.pkl")

    print(f"✅ Best Model: {best_name} (Acc: {best_acc:.4f}) saved to artifacts/best_model.pkl")

if __name__ == "__main__":
    main()
