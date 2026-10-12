import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from modelwatch.data import FEATURES, make_data
from modelwatch.monitor import build_reference, population_stability_index, detect_drift
from modelwatch.train import train
from modelwatch import api

def test_reproducible_synthetic_data():
    a, b = make_data(200, seed=7), make_data(200, seed=7)
    assert a.equals(b)
    assert list(FEATURES) == list(a.columns[:-1])

def test_training_writes_artifacts(tmp_path):
    metrics = train(str(tmp_path), n=300)
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["train_samples"] + metrics["test_samples"] == 300
    assert (tmp_path / "model.joblib").exists()
    assert json.loads((tmp_path / "reference.json").read_text())

def test_psi_is_small_for_reference_distribution():
    frame = make_data(400)
    baseline = build_reference(frame, FEATURES)
    result = detect_drift(frame, baseline)
    assert all(value < 0.001 for value in result["feature_psi"].values())

def test_psi_detects_substantial_shift():
    x = np.linspace(0, 10, 500)
    baseline = build_reference(pd.DataFrame({"feature": x}), ["feature"])
    assert population_stability_index(x + 100, baseline["feature"]) > 0.2

def test_api_prediction_and_drift(tmp_path, monkeypatch):
    train(str(tmp_path), n=300)
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    api.artifacts.cache_clear()
    client = TestClient(api.app)
    item = {
        "temperature_c": 72.0, "vibration_mm_s": 2.1,
        "hours_since_service": 1000.0, "load_pct": 50.0,
    }
    prediction = client.post("/predict", json=item)
    assert prediction.status_code == 200
    assert 0 <= prediction.json()["failure_probability"] <= 1
    drift = client.post("/drift", json=[item] * 20)
    assert drift.status_code == 200
    assert "feature_psi" in drift.json()
    api.artifacts.cache_clear()
