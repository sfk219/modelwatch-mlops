# ModelWatch — Model Serving and Drift Monitoring

An educational Python/FastAPI model serving application that trains a classifier on **synthetic telemetry**, serves predictions, and calculates population stability index (PSI) for incoming feature batches.

## Run
```bash
pip install -r requirements.txt
python -m modelwatch.train
uvicorn modelwatch.api:app --reload
pytest -q
```
API documentation: `http://localhost:8000/docs`. Models and reference data are created locally under `artifacts/`.

## Limitations
Synthetic data only. Drift values illustrate monitoring concepts and do not establish operational ML reliability. No patient data or customer data is included.
