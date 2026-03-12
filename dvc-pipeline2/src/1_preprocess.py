"""
src/1_preprocess.py
─────────────────────────────────────────────────────────────
DVC Stage 1: DATA PREPROCESSING

Reads from:   ../data/FraudTrain.csv  (main folder)
              ../data/FraudTest.csv   (main folder)
Writes to:    ../data/processed_train.csv
              ../data/processed_test.csv

RUN: python src/1_preprocess.py
  OR: dvc repro preprocess  (from inside dvc-pipeline/)
"""

import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import LabelEncoder

# ── Load params ───────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TRAIN_PATH       = params["data"]["train_path"]
TEST_PATH        = params["data"]["test_path"]
PROCESSED_TRAIN  = params["data"]["processed_train"]
PROCESSED_TEST   = params["data"]["processed_test"]
COLS_TO_DROP     = params["preprocess"]["cols_to_drop"]
CAT_COLS         = params["preprocess"]["cat_cols"]
CURRENT_YEAR     = params["preprocess"]["current_year"]


def load_data():
    print(f"[INFO] Loading: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)

    print(f"[INFO] Loading: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)

    print(f"[INFO] Train shape: {train_df.shape}")
    print(f"[INFO] Test shape:  {test_df.shape}")
    return train_df, test_df


def drop_columns(train_df, test_df):
    train_df = train_df.drop(columns=COLS_TO_DROP)
    test_df  = test_df.drop(columns=COLS_TO_DROP)
    print(f"[INFO] Dropped columns: {COLS_TO_DROP}")
    return train_df, test_df


def engineer_features(train_df, test_df):
    for df in [train_df, test_df]:
        # Extract hour from transaction datetime
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df.drop(columns=["trans_date_trans_time"], inplace=True)

        # Extract age from date of birth
        df["dob"] = pd.to_datetime(df["dob"])
        df["age"] = CURRENT_YEAR - df["dob"].dt.year
        df.drop(columns=["dob"], inplace=True)

        # Distance between customer and merchant
        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 +
            (df["long"] - df["merch_long"]) ** 2
        )

    print("[INFO] Feature engineering done: hour, age, distance")
    return train_df, test_df


def encode_categoricals(train_df, test_df):
    encoder = LabelEncoder()
    for col in CAT_COLS:
        train_df[col] = encoder.fit_transform(train_df[col])
        test_df[col]  = encoder.fit_transform(test_df[col])
    print(f"[INFO] Label encoded: {CAT_COLS}")
    return train_df, test_df


def save_processed(train_df, test_df):
    # ../data/ folder already exists (it has your CSV files)
    # No need to create it — just save directly
    train_df.to_csv(PROCESSED_TRAIN, index=False)
    test_df.to_csv(PROCESSED_TEST, index=False)
    print(f"[INFO] Saved: {PROCESSED_TRAIN}  shape={train_df.shape}")
    print(f"[INFO] Saved: {PROCESSED_TEST}   shape={test_df.shape}")


def main():
    print("\n" + "="*50)
    print("  STAGE 1: PREPROCESSING")
    print("="*50)
    train_df, test_df = load_data()
    train_df, test_df = drop_columns(train_df, test_df)
    train_df, test_df = engineer_features(train_df, test_df)
    train_df, test_df = encode_categoricals(train_df, test_df)
    save_processed(train_df, test_df)
    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()
