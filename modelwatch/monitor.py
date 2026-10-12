"""PSI is a demonstration of feature distribution drift, not prediction accuracy."""
import numpy as np

def build_reference(frame, features, bins: int = 10) -> dict:
    reference = {}
    for feature in features:
        values = np.asarray(frame[feature], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("non-finite reference values")
        cuts = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.concatenate(([-np.inf], cuts, [np.inf]))
        counts, _ = np.histogram(values, bins=edges)
        reference[feature] = {
            "cuts": cuts.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
        }
    return reference

def population_stability_index(values, reference: dict, epsilon: float = 1e-6) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("values must be finite and non-empty")
    edges = np.concatenate(([-np.inf], reference["cuts"], [np.inf]))
    observed, _ = np.histogram(arr, bins=edges)
    actual = observed / observed.sum()
    expected = np.asarray(reference["proportions"], dtype=float)
    # Small epsilon stabilizes log where observed frequency is zero.
    a = np.maximum(actual, epsilon)
    e = np.maximum(expected, epsilon)
    return float(np.sum((a - e) * np.log(a / e)))

def detect_drift(frame, reference: dict, threshold: float = 0.2) -> dict:
    scores = {
        feature: round(population_stability_index(frame[feature], metadata), 5)
        for feature, metadata in reference.items()
    }
    return {"threshold": threshold, "feature_psi": scores,
            "drifting_features": [name for name, score in scores.items() if score > threshold]}
