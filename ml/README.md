# ML Service

FastAPI inference microservice for anomaly detection, fault classification, and RUL estimation.

## Running locally

```bash
pip install -r requirements.txt
uvicorn src.serve:app --port 8001 --reload
```

API docs: http://localhost:8001/docs

## Offline Training

```bash
# Export training data from TimescaleDB first
docker compose exec timescaledb psql -U digital_twin -d engine_db \
  -c "\COPY (SELECT * FROM telemetry) TO '/tmp/telemetry.csv' CSV HEADER"

# Or generate synthetic labelled telemetry from simulator physics
python src/generate_dataset.py --samples 1000

# Then train each model
python -m src.anomaly_detection
python -m src.fault_classifier            # XGBoost primary (or --fallback for Random Forest)
python -m src.rul_model
```

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## Files

| File | Purpose |
|------|---------|
| `src/serve.py` | FastAPI + MQTT subscriber → inference → alert publisher |
| `src/anomaly_detection.py` | Isolation Forest wrapper |
| `src/fault_classifier.py` | XGBoost 7-class classifier with Random Forest fallback |
| `src/generate_dataset.py` | Synthetic telemetry generator from engine physics |
| `src/rul_model.py` | GRU regression for RUL |
| `src/explainability.py` | SHAP top-K feature extractor |
| `tests/test_fault_classifier.py` | Unit and integration tests |

**Owner**: ML1, ML2
