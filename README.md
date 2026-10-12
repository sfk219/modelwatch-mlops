# ModelWatch — Model Serving and Drift Monitoring (MLOps)

An end-to-end **educational ML engineering** demonstration: generate synthetic industrial telemetry, train a logistic-regression model, serve REST predictions, establish a training-data reference distribution, and monitor incoming feature distributions with PSI.

## Run locally
```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements-dev.txt
python -m modelwatch.train
uvicorn modelwatch.api:app --reload
```
Open `http://localhost:8000/docs`.

### Predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature_c":72,"vibration_mm_s":2.1,"hours_since_service":1000,"load_pct":50}'
```
Send an array of at least 10 such telemetry objects to `POST /drift` for PSI scores per feature.

### Test and package
```bash
pytest -q
docker build -t modelwatch .
docker run -p 8000:8000 modelwatch
```

### How it works
1. `data.py` produces synthetic temperature, vibration, service-hour and load data.
2. `train.py` creates train/test splits, preprocessing and model artifacts.
3. `api.py` loads locally generated artifacts and serves predictions with FastAPI.
4. `monitor.py` compares input feature-bin proportions with training-data reference histograms.

**Limitations:** data are synthetic; the PSI threshold (0.2) is illustrative; serving has no auth, model registry, online retraining or production observability. This portfolio project is not a medical or industrial decision system.
