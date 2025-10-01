# train_and_select.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True)
}

def train_and_select(X_train, y_train, X_val, y_val, artifacts_dir="artifacts", experiment_name="MobilePricePrediction"):
    os.makedirs(artifacts_dir, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    best_acc = 0
    best_model_name = None
    best_model = None

    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        mlflow.autolog()  # autolog for each run

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            print(f"{name} Validation Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                best_model = model

    # Save best model
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest model: {best_model_name} with accuracy {best_acc:.4f}")
    print(f"Saved to {model_path}")
    return best_model, best_model_name 

