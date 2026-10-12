"""Prediction & feature-drift REST API over a locally trained synthetic model."""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURES
from .monitor import detect_drift

app = FastAPI(title="ModelWatch MLOps", version="1.0")
ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", "artifacts"))

class Reading(BaseModel):
    temperature_c: float = Field(ge=-100, le=250)
    vibration_mm_s: float = Field(ge=0, le=100)
    hours_since_service: float = Field(ge=0, le=100000)
    load_pct: float = Field(ge=0, le=100)

def as_frame(readings: list[Reading]):
    return pd.DataFrame([r.model_dump() for r in readings], columns=FEATURES)

@lru_cache(maxsize=1)
def artifacts():
    try:
        model = joblib.load(ARTIFACT_DIR / "model.joblib")
        reference = json.loads((ARTIFACT_DIR / "reference.json").read_text(encoding="utf-8"))
        metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
        return model, reference, metrics
    except FileNotFoundError:
        raise HTTPException(503, "Model artifacts missing: run python -m modelwatch.train")

@app.get("/health")
def health():
    return {"status": "ok", "artifacts_ready": (ARTIFACT_DIR / "model.joblib").exists()}

@app.post("/predict")
def predict(reading: Reading):
    model, _, _ = artifacts()
    probability = float(model.predict_proba(as_frame([reading]))[0, 1])
    return {"failure_probability": round(probability, 5),
            "model": "logistic_regression", "data_context": "synthetic_demo"}

@app.post("/drift")
def drift(readings: Annotated[list[Reading], Field(min_length=10, max_length=10000)]):
    _, reference, _ = artifacts()
    return detect_drift(as_frame(readings), reference)

@app.get("/metrics")
def metrics():
    return artifacts()[2]
