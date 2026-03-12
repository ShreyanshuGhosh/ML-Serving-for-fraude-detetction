"""
src/3_evaluate.py
─────────────────────────────────────────────────────────────
DVC Stage 3: MODEL EVALUATION

Reads from:   ../models/random_forest.pkl
              ../models/xgboost.pkl
              ../data/X_test_scaled.csv
              ../data/y_test.csv
Writes to:    ../metrics/scores.json

RUN: python src/3_evaluate.py
  OR: dvc repro evaluate  (from inside dvc-pipeline/)

View metrics:
  dvc metrics show
  dvc metrics diff
"""

import pandas as pd
import numpy as np
import yaml
import os
import json
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ── Load params ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

X_TEST_SCALED   = params["data"]["X_test_scaled"]
Y_TEST          = params["data"]["y_test"]
MODELS_DIR      = params["data"]["models_dir"]      # ../models
METRICS_DIR     = params["data"]["metrics_dir"]     # ../metrics
XGB_THRESHOLD   = params["evaluate"]["xgb_threshold"]


def load_models():
    model1 = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    model2 = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    print("[INFO] Both models loaded.")
    return model1, model2


def load_test_data():
    X_test = pd.read_csv(X_TEST_SCALED).values
    y_test = pd.read_csv(Y_TEST).values.ravel()
    print(f"[INFO] Test data: {X_test.shape}")
    return X_test, y_test


def evaluate_random_forest(model1, X_test, y_test):
    print("\n" + "="*50)
    print("  MODEL 1: RANDOM FOREST")
    print("="*50)
    y_pred = model1.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    roc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC: {roc:.4f}")

    r = classification_report(y_test, y_pred, output_dict=True)
    return {
        "rf_precision_fraud": round(r["1"]["precision"], 4),
        "rf_recall_fraud":    round(r["1"]["recall"], 4),
        "rf_f1_fraud":        round(r["1"]["f1-score"], 4),
        "rf_roc_auc":         round(roc, 4),
    }


def evaluate_xgboost(model2, X_test, y_test):
    print("\n" + "="*50)
    print(f"  MODEL 2: XGBOOST  (threshold={XGB_THRESHOLD})")
    print("="*50)
    y_prob = model2.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > XGB_THRESHOLD).astype(int)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {roc:.4f}")

    r = classification_report(y_test, y_pred, output_dict=True)
    return {
        "xgb_precision_fraud": round(r["1"]["precision"], 4),
        "xgb_recall_fraud":    round(r["1"]["recall"], 4),
        "xgb_f1_fraud":        round(r["1"]["f1-score"], 4),
        "xgb_roc_auc":         round(roc, 4),
        "xgb_threshold":       XGB_THRESHOLD,
    }


def save_metrics(rf_metrics, xgb_metrics):
    os.makedirs(METRICS_DIR, exist_ok=True)
    all_metrics = {**rf_metrics, **xgb_metrics}
    path = os.path.join(METRICS_DIR, "scores.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved: {path}")
    print(json.dumps(all_metrics, indent=2))


def main():
    print("\n" + "="*50)
    print("  STAGE 3: EVALUATION")
    print("="*50)
    model1, model2   = load_models()
    X_test, y_test   = load_test_data()
    rf_metrics       = evaluate_random_forest(model1, X_test, y_test)
    xgb_metrics      = evaluate_xgboost(model2, X_test, y_test)
    save_metrics(rf_metrics, xgb_metrics)
    print("\n[DONE] Evaluation complete.")
    print("\nCompare runs:  dvc metrics show")
    print("Diff runs:     dvc metrics diff")


if __name__ == "__main__":
    main()
