# 📱 Mobile Price Prediction — MLOps Project  

[![Hugging Face Spaces](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/GokulV/Mobile_price_prediction)

This project predicts mobile phone **price ranges** (`Low`, `Medium`, `High`, `Very High`) based on device specifications like battery power, RAM, internal memory, screen size, and camera resolution.  

It is built with **scikit-learn**, tracked using **MLflow**, deployed with **Gradio**, and continuously delivered to **Hugging Face Spaces** via **GitHub Actions (CI/CD pipeline)**.  

---

## 🚀 Features
- Preprocessing: Missing value handling, scaling, imputing  
- Training: Logistic Regression, RandomForest, SVM (best model selection)  
- Experiment Tracking: MLflow  
- Deployment: Interactive Gradio app hosted on Hugging Face Spaces  
- CI/CD: GitHub Actions pipeline (auto-trains + auto-deploys)  

---

## 📂 Project Structure
```
Mobile-price-prediction---MLOPS/
│
├── data/
│   └── raw/                # train.csv, test.csv
│
├── artifacts/              # Trained model & preprocessing artifacts
│   └── best_model.pkl
│
├── src/                    # Reusable ML code
│   ├── preprocessing.py
│   ├── train_multiple.py
│   ├── train.py
│   └── evaluation.py
│
├── app.py                  # Gradio frontend
├── run_experiment.py       # Full training + model selection
├── requirements.txt        # Dependencies
├── Dockerfile              # Containerization (optional)
│
├── .github/
│   └── workflows/
│       ├── ci.yml          # Lint + Test
│       └── train_and_deploy.yml  # Train + Deploy to HF Space
│
└── README.md               # Project documentation
```

---

## ⚡ Quickstart (Local)

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/Mobile-price-prediction---MLOPS.git
   cd Mobile-price-prediction---MLOPS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training pipeline:
   ```bash
   python run_experiment.py
   ```
   → This saves the best model in `artifacts/best_model.pkl`.

4. Launch app locally:
   ```bash
   python app.py
   ```
   → Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.  

---

## 🔎 MLflow Tracking

Start MLflow UI:
```bash
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore experiments.  

---

## 🖥️ Deployment

The app is deployed on Hugging Face Spaces:  
👉 [Live Demo](https://huggingface.co/spaces/GokulV/Mobile_price_prediction)

---

## ⚙️ CI/CD Pipeline

This project uses **GitHub Actions**:

- **CI (`ci.yml`)**  
  Runs on every push/PR:
  - Linting (`flake8`)  
  - Tests (`pytest`)  
  - Docker build (smoke test)  

- **CD (`train_and_deploy.yml`)**  
  Runs on main branch push:
  - Train models (`run_experiment.py`)  
  - Save best model → `artifacts/best_model.pkl`  
  - Deploy app + model to Hugging Face Space  

---

## 🔑 GitHub Secrets Required

Set these under **Repo → Settings → Secrets and Variables → Actions**:

- `HF_TOKEN` → Hugging Face Access Token (with `write` permissions).  
- *(Optional)* `GHCR_PAT` → GitHub Personal Access Token for Docker image push (only if using GHCR).  

---

## 📦 Requirements
- Python 3.9+  
- pandas, numpy, scikit-learn  
- mlflow, joblib, gradio  
- pytest, flake8  

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📊 Model Performance
- Logistic Regression → ~96% accuracy (best model)  
- RandomForest, SVM also tested via MLflow  

---

## ✨ Future Improvements
- Add hyperparameter tuning (GridSearchCV/Optuna).  
- Add more mobile features (weight, resolution, etc.).  
- Enable continuous training on new datasets.  
- Deploy as REST API using `mlflow models serve`.  

---

👨‍💻 **Author**: [@GokulV](https://huggingface.co/GokulV)  
