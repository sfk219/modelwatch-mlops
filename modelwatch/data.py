"""Synthetic equipment telemetry. No real user, health or business data."""
import numpy as np
import pandas as pd

FEATURES = ("temperature_c", "vibration_mm_s", "hours_since_service", "load_pct")

def make_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    if n < 100:
        raise ValueError("n must be at least 100")
    rng = np.random.default_rng(seed)
    temp = rng.normal(67, 11, n)
    vibration = rng.gamma(2.5, 1.15, n)
    hours = rng.uniform(0, 10000, n)
    load = rng.uniform(10, 100, n)
    logit = (-3.2 + 0.055 * (temp - 67) + 0.5 * (vibration - 3)
             + 0.00018 * (hours - 5000) + 0.024 * (load - 55))
    probability = 1 / (1 + np.exp(-logit))
    labels = rng.binomial(1, probability)
    return pd.DataFrame({
        "temperature_c": temp,
        "vibration_mm_s": vibration,
        "hours_since_service": hours,
        "load_pct": load,
        "failure": labels,
    })
