import mlflow
from mlflow import sklearn as mlflow_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True),
}

def train_multiple(X_train, y_train, X_val, y_val, experiment_name="MobilePricePrediction"):
    """
    Trains multiple models, logs accuracy to MLflow (locally),
    and returns dict with model + accuracy.
    """
    mlflow.set_experiment(experiment_name)
    results = {}

    # Detect if running inside GitHub Actions (CI/CD)
    IS_CI = os.getenv("GITHUB_ACTIONS") == "true"

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)

            mlflow.log_metric("accuracy", float(acc))

            # ✅ Only log MLflow model locally (skip in CI to avoid artifact path errors)
            if not IS_CI:
                mlflow_sklearn.log_model(model, name=f"{name}_model")

            results[name] = {"model": model, "accuracy": acc}

    return results
