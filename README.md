# ML Serving for Fraud Detection

An end-to-end MLOps project for real-time credit card fraud detection. Covers the full pipeline from data preprocessing to model deployment on Kubernetes with automated CI/CD.

---

## Overview

This project trains a machine learning model to detect fraudulent credit card transactions and serves it as a production-ready REST API. The entire pipeline is automated using DVC, containerized with Docker, deployed on Kubernetes, and integrated with a CI/CD pipeline via GitHub Actions.

---

## Architecture

```
Raw Data (Kaggle)
      |
      v
DVC Pipeline
  1_preprocess.py  ->  2_train.py  ->  3_evaluate.py
      |
      v
Trained Models (XGBoost + RandomForest)
      |
      v
FastAPI REST API  (/predict, /health)
      |
      v
Docker Image  ->  Docker Hub
      |
      v
Kubernetes Cluster (Minikube)
  - Deployment (2 replicas)
  - Service (NodePort)
  - HPA (auto-scales 2 to 10 pods)
      |
      v
GitHub Actions CI/CD
  - Run tests
  - Build Docker image
  - Push to Docker Hub
```

---

## Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.10 |
| ML Models | XGBoost, Random Forest |
| Pipeline | DVC |
| API | FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes (Minikube) |
| CI/CD | GitHub Actions |
| Data Versioning | DVC |
| Class Imbalance | SMOTE |

---

## Project Structure

```
ML-Serving-for-fraude-detetction/
|
|-- data/
|   |-- FraudTrain.csv          (not tracked by git - see below)
|   |-- FraudTest.csv           (not tracked by git - see below)
|
|-- dvc-pipeline/
|   |-- src/
|   |   |-- 1_preprocess.py     (load, clean, feature engineering, encode)
|   |   |-- 2_train.py          (SMOTE, scale, train RF + XGBoost)
|   |   |-- 3_evaluate.py       (predict, metrics, confusion matrix)
|   |-- dvc.yaml                (pipeline definition)
|   |-- params.yaml             (all configurable parameters)
|
|-- k8s/
|   |-- deployment.yaml         (2 replicas, liveness + readiness probes)
|   |-- service.yaml            (NodePort service)
|   |-- hpa.yaml                (auto-scales 2-10 pods based on CPU)
|
|-- .github/
|   |-- workflows/
|       |-- ci.yml              (test, build, push on every git push)
|
|-- notebook/
|   |-- main.ipynb              (original exploration notebook)
|
|-- models/                     (created by DVC pipeline)
|   |-- xgboost.pkl
|   |-- random_forest.pkl
|   |-- scaler.pkl
|
|-- metrics/                    (created by DVC pipeline)
|   |-- scores.json
|
|-- main.py                     (FastAPI application)
|-- Dockerfile
|-- requirements.txt
|-- .gitignore
```

---

## Dataset

Download from Kaggle:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Place both files in the `data/` folder:
```
data/FraudTrain.csv
data/FraudTest.csv
```

The dataset contains real credit card transactions labeled as fraud or legitimate. It has a heavy class imbalance (fraud is less than 1% of transactions) which is handled using SMOTE during training.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ShreyanshuGhosh/ML-Serving-for-fraude-detetction.git
cd ML-Serving-for-fraude-detetction
```

### 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
pip install dvc
```

### 3. Download the dataset

Download from the Kaggle link above and place in `data/`.

### 4. Run the DVC pipeline

```bash
dvc init
cd dvc-pipeline
python -m dvc repro
```

This runs all 3 stages in order:
- Preprocessing: cleans data, extracts features, encodes categoricals
- Training: applies SMOTE, scales features, trains both models
- Evaluation: generates metrics and saves to metrics/scores.json

### 5. View metrics

```bash
python -m dvc metrics show
```

### 6. Run the API locally

```bash
cd ..
python -m uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Welcome message |
| /health | GET | Health check |
| /predict | POST | Predict fraud for a transaction |

### Sample Request

```json
POST /predict

{
  "merchant": 319,
  "category": 10,
  "amt": 2.86,
  "gender": 1,
  "city": 157,
  "state": 39,
  "zip": 29209,
  "lat": 33.9659,
  "long": -80.9355,
  "city_pop": 333497,
  "job": 275,
  "unix_time": 1371816865,
  "merch_lat": 33.986391,
  "merch_long": -81.200714,
  "hour": 12,
  "age": 58,
  "distance": 0.266004
}
```

### Sample Response

```json
{
  "prediction": 0,
  "label": "LEGIT",
  "fraud_probability": 0.0231,
  "legit_probability": 0.9769
}
```

---

## Docker

```bash
# Build
docker build -t shreyanshughosh/fraud-detector:latest .

# Run
docker run -p 8000:8000 shreyanshughosh/fraud-detector:latest

# Pull from Docker Hub
docker pull shreyanshughosh/fraud-detector:latest
```

---

## Kubernetes Deployment

Requirements: Minikube and kubectl installed.

```bash
# Start cluster
minikube start

# Deploy
kubectl apply -f k8s/

# Check pods
kubectl get pods

# Get API URL
minikube service fraud-api-service --url

# Check autoscaler
kubectl get hpa
```

The HPA automatically scales pods between 2 and 10 based on CPU usage. If CPU exceeds 70%, new pods are added automatically.

---

## CI/CD Pipeline

Every push to the `main` branch automatically:

1. Sets up Python environment
2. Installs dependencies
3. Runs tests
4. Builds Docker image
5. Pushes to Docker Hub with `latest` tag and commit SHA tag

Pipeline defined in `.github/workflows/ci.yml`.

Required GitHub Secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## Model Performance

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Precision (Fraud) | 0.59 | 0.06 |
| Recall (Fraud) | 0.69 | 0.93 |
| F1 Score (Fraud) | 0.64 | 0.10 |
| ROC AUC | 0.84 | 0.98 |

XGBoost achieves higher ROC AUC (0.98) and recall (0.93) which is more important for fraud detection — catching more fraud matters more than false alarms. A threshold of 0.3 is used instead of default 0.5.

---
