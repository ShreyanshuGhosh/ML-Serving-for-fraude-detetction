"""
src/2_train.py
─────────────────────────────────────────────────────────────
DVC Stage 2: MODEL TRAINING

Reads from:   ../data/processed_train.csv
              ../data/processed_test.csv
Writes to:    ../models/random_forest.pkl
              ../models/xgboost.pkl
              ../models/scaler.pkl
              ../data/X_test_scaled.csv
              ../data/y_test.csv

RUN: python src/2_train.py
  OR: dvc repro train  (from inside dvc-pipeline/)
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ── Load params ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROCESSED_TRAIN = params["data"]["processed_train"]
PROCESSED_TEST  = params["data"]["processed_test"]
X_TEST_SCALED   = params["data"]["X_test_scaled"]
Y_TEST          = params["data"]["y_test"]
MODELS_DIR      = params["data"]["models_dir"]      # ../models

RANDOM_STATE    = params["train"]["random_state"]
USE_SMOTE       = params["train"]["smote"]
RF_PARAMS       = params["train"]["random_forest"]
XGB_PARAMS      = params["train"]["xgboost"]


def load_processed_data():
    train_df = pd.read_csv(PROCESSED_TRAIN)
    test_df  = pd.read_csv(PROCESSED_TEST)
    print(f"[INFO] Loaded processed train: {train_df.shape}")
    print(f"[INFO] Loaded processed test:  {test_df.shape}")
    return train_df, test_df


def split_features_target(train_df, test_df):
    X_train = train_df.drop(columns=["is_fraud"])
    y_train = train_df["is_fraud"]
    X_test  = test_df.drop(columns=["is_fraud"])
    y_test  = test_df["is_fraud"]
    print(f"[INFO] X_train: {X_train.shape} | fraud rate: {y_train.mean()*100:.2f}%")
    print(f"[INFO] X_test:  {X_test.shape}  | fraud rate: {y_test.mean()*100:.2f}%")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    print(f"[INFO] Before SMOTE — fraud: {y_train.sum()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE  — fraud: {y_res.sum()} (50/50 balance)")
    return X_res, y_res


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler to ../models/
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Scaler saved: {scaler_path}")
    return X_train_scaled, X_test_scaled


def train_models(X_train, y_train):
    print("\n[INFO] Training Random Forest...")
    model1 = RandomForestClassifier(**RF_PARAMS)
    model1.fit(X_train, y_train)
    print("[INFO] Random Forest done.")

    print("\n[INFO] Training XGBoost...")
    model2 = xgb.XGBClassifier(**XGB_PARAMS)
    model2.fit(X_train, y_train)
    print("[INFO] XGBoost done.")
    return model1, model2


def save_models(model1, model2):
    os.makedirs(MODELS_DIR, exist_ok=True)
    rf_path  = os.path.join(MODELS_DIR, "random_forest.pkl")
    xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    joblib.dump(model1, rf_path)
    joblib.dump(model2, xgb_path)
    print(f"\n[INFO] Saved: {rf_path}")
    print(f"[INFO] Saved: {xgb_path}")


def save_test_data(X_test_scaled, y_test, feature_names):
    X_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    X_df.to_csv(X_TEST_SCALED, index=False)
    y_test.reset_index(drop=True).to_csv(Y_TEST, index=False)
    print(f"[INFO] Saved test data: {X_TEST_SCALED}")
    print(f"[INFO] Saved test labels: {Y_TEST}")


def main():
    print("\n" + "="*50)
    print("  STAGE 2: TRAINING")
    print("="*50)
    train_df, test_df = load_processed_data()
    X_train, X_test, y_train, y_test = split_features_target(train_df, test_df)

    if USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    model1, model2 = train_models(X_train_scaled, y_train)
    save_models(model1, model2)
    save_test_data(X_test_scaled, y_test, X_test.columns.tolist())
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
