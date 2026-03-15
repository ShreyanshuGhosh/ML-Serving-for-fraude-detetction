# """
# main.py
# FastAPI app for Fraud Detection
# Run: uvicorn main:app --reload --port 8000
# Docs: http://localhost:8000/docs


import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ── Load model and scaler once at startup ─────────────────────
# These are saved by your DVC pipeline in models/
model   = joblib.load("models/xgboost.pkl")
scaler  = joblib.load("models/scaler.pkl")

app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a credit card transaction is fraud or legit",
    version="1.0.0"
)

# ── Input schema — matches your dataset columns ───────────────
class Transaction(BaseModel):
    merchant  : int
    category  : int
    amt       : float
    gender    : int
    city      : int
    state     : int
    zip       : int
    lat       : float
    long      : float
    city_pop  : int
    job       : int
    unix_time : int
    merch_lat : float
    merch_long: float
    hour      : int
    age       : int
    distance  : float

    class ConfigDict:
        json_schema_extra = {
            "example": {
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
        }

# ── Feature column order must match training ──────────────────
FEATURE_COLUMNS = [
    "merchant", "category", "amt", "gender", "city",
    "state", "zip", "lat", "long", "city_pop", "job",
    "unix_time", "merch_lat", "merch_long", "hour", "age", "distance"
]

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgboost"}


@app.post("/predict")
def predict(transaction: Transaction):
    # Step 1 — convert input to dataframe
    data = pd.DataFrame([transaction.dict()], columns=FEATURE_COLUMNS)

    # Step 2 — scale using the same scaler from training
    data_scaled = scaler.transform(data)

    # Step 3 — predict
    fraud_prob  = model.predict_proba(data_scaled)[0][1]
    prediction  = int(fraud_prob > 0.3)   # same threshold as your notebook

    return {
        "prediction"       : prediction,
        "label"            : "FRAUD" if prediction == 1 else "LEGIT",
        "fraud_probability": round(float(fraud_prob), 4),
        "legit_probability": round(1 - float(fraud_prob), 4),
    }