# app.py
import gradio as gr
import joblib
import numpy as np
import os
from pathlib import Path

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = Path(ARTIFACTS_DIR) / "best_model.pkl"
SCALER_PATH = Path(ARTIFACTS_DIR) / "scaler.pkl"
IMPUTER_PATH = Path(ARTIFACTS_DIR) / "num_imputer.pkl"

# Load model + preprocessors if available
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python run_experiment.py` first.")

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
imputer = joblib.load(IMPUTER_PATH) if IMPUTER_PATH.exists() else None

# If classification, map classes to human labels:
CLASS_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

def predict_price(*inputs):
    # inputs order must match training order used in preprocessing
    features = np.array(inputs).reshape(1, -1)

    # apply imputer/scaler if available
    if imputer is not None:
        features = imputer.transform(features)
    if scaler is not None:
        features = scaler.transform(features)

    pred = model.predict(features)[0]

    # handle classifier vs regressor detection
    try:
        # If classifier has predict_proba, we can show probabilities
        if hasattr(model, "predict_proba") and model.predict_proba is not None:
            # classification
            label = CLASS_MAP.get(int(pred), str(pred))
            probs = model.predict_proba(features)[0]
            prob_str = ", ".join([f"{CLASS_MAP.get(i, i)}: {p:.2f}" for i, p in enumerate(probs)])
            return f"Predicted category: {label}\nProbabilities → {prob_str}"
        else:
            # if numeric
            return f"Predicted value: {round(float(pred), 2)}"
    except Exception:
        # numeric fallback
        return f"Predicted: {round(float(pred), 2)}"

# --- Build inputs dynamically if you want, else hard-code the 20 features matching your data ---
inputs = [
    gr.Slider(500, 2000, step=1, label="battery_power"),
    gr.Radio([0, 1], label="blue"),
    gr.Slider(0.5, 3.2, step=0.1, label="clock_speed"),
    gr.Radio([0, 1], label="dual_sim"),
    gr.Slider(0, 20, step=1, label="fc"),
    gr.Radio([0, 1], label="four_g"),
    gr.Slider(2, 64, step=2, label="int_memory"),
    gr.Slider(0.1, 1.0, step=0.1, label="m_dep"),
    gr.Slider(80, 200, step=1, label="mobile_wt"),
    gr.Slider(1, 8, step=1, label="n_cores"),
    gr.Slider(0, 21, step=1, label="pc"),
    gr.Slider(0, 1960, step=10, label="px_height"),
    gr.Slider(0, 1980, step=10, label="px_width"),
    gr.Slider(256, 4096, step=128, label="ram"),
    gr.Slider(5, 19, step=1, label="sc_h"),
    gr.Slider(0, 18, step=1, label="sc_w"),
    gr.Slider(2, 20, step=1, label="talk_time"),
    gr.Radio([0, 1], label="three_g"),
    gr.Radio([0, 1], label="touch_screen"),
    gr.Radio([0, 1], label="wifi"),
]

outputs = gr.Textbox(label="Prediction")

gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=outputs,
    title="📱 Mobile Price Predictor",
    description="Provide phone specs. Model & preprocessing are loaded from artifacts/",
).launch()
