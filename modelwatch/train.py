"""Train a reproducible classifier and serialize evaluation metrics + drift baseline."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, make_data
from .monitor import build_reference

def train(output: str = "artifacts", n: int = 1500) -> dict:
    frame = make_data(n=n)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[list(FEATURES)], frame["failure"], test_size=0.25,
        stratify=frame["failure"], random_state=42)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    labels = model.predict(x_test)
    metrics = {
        "dataset": "synthetic equipment telemetry",
        "samples": len(frame),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "classification_report": classification_report(y_test, labels, output_dict=True),
    }
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / "model.joblib")
    (target / "reference.json").write_text(
        json.dumps(build_reference(x_train, FEATURES), indent=2), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--rows", type=int, default=1500)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.rows), indent=2))
