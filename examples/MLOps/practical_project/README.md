# Practical MLOps Project: Churn Prediction

End-to-end MLOps pipeline for customer churn prediction.

Adapted from MLOps Lesson 12. Cloud references (S3, K8s, TorchServe) replaced
with local alternatives (filesystem, Docker Compose, FastAPI).

## Architecture

```
Data → Feast Feature Store → MLflow Training → FastAPI Serving → Evidently Monitoring
```

## Prerequisites

- Python 3.9+
- Docker & Docker Compose
- ~2 GB disk space

## Quick Start

```bash
# 1. Start infrastructure (MLflow, PostgreSQL, Redis)
docker compose up -d

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Generate synthetic churn data
python data/generate_churn_data.py

# 4. Initialize Feast feature store
cd features && feast apply && cd ..

# 5. Ingest and validate data
python src/ingest.py

# 6. Train model (logged to MLflow)
python src/train.py --config configs/training_config.yaml

# 7. Serve model
uvicorn src.serve:app --host 0.0.0.0 --port 8000

# 8. Monitor for drift
python src/monitor.py
```

## Services

| Service | Port | URL |
|---------|------|-----|
| MLflow UI | 5000 | http://localhost:5000 |
| FastAPI serving | 8000 | http://localhost:8000 |
| Redis (Feast online) | 6379 | localhost:6379 |
| PostgreSQL (MLflow backend) | 5432 | localhost:5432 |

## Directory Structure

```
practical_project/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── data/
│   └── generate_churn_data.py    # Synthetic churn dataset (~5000 rows)
├── features/
│   ├── feature_store.yaml        # Feast config (SQLite offline, Redis online)
│   ├── entities.py               # User entity definition
│   └── feature_views.py          # FeatureView definitions
├── src/
│   ├── ingest.py                 # Data ingestion + validation
│   ├── train.py                  # MLflow training + model registry
│   ├── serve.py                  # FastAPI model serving
│   └── monitor.py                # Evidently drift detection
├── configs/
│   └── training_config.yaml      # Training hyperparameters + quality gates
└── tests/
    ├── test_data.py              # Data pipeline tests
    └── test_model.py             # Model quality gate tests
```

## Customization

- **Different model**: Change `model.type` in `configs/training_config.yaml`
- **More features**: Add FeatureViews in `features/feature_views.py`
- **Quality gates**: Adjust thresholds in `configs/training_config.yaml`

## Related Lessons

- **MLOps L11**: Feature Stores (Feast concepts)
- **MLOps L12**: This project (E2E pipeline)
- **DE L14**: Data pipeline project (upstream data source)
