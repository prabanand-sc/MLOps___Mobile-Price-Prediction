# src/train.py
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow.sklearn as mlflow_sklearn
def train_model(X_train, y_train, X_val, y_val, artifacts_dir="artifacts", experiment_name="MobilePricePrediction"):
    os.makedirs(artifacts_dir, exist_ok=True)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="LogisticRegression"):
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        mlflow.log_metric("accuracy", float(acc))

        mlflow_sklearn.log_model(model, "LogisticRegression")
        print(f"Saved LogisticRegression to {artifacts_dir}/best_model.pkl (acc {acc:.4f})")

    return model, acc
