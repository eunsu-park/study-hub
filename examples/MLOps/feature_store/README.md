# Feature Store Examples

Standalone Feast Feature Store examples bridging theory (L11) and practice (L12).

Adapted from MLOps Lesson 11 (Feature Stores) with references to Lesson 12
(Practical MLOps Project).

## Prerequisites

- Python 3.9+
- Redis running (or `docker run -d -p 6379:6379 redis:7-alpine`)
- `pip install feast[redis] pandas pyarrow`

## Quick Start

```bash
# 1. Generate sample data (reuses L12 data schema)
cd ../practical_project
python data/generate_churn_data.py --output ../feature_store/data/churn.parquet
cd ../feature_store

# 2. Register features with Feast
feast apply

# 3. Materialize offline → online
python materialize.py

# 4. Serve features (both historical and online)
python serve_features.py
```

## Files

| File | Description | Lesson Reference |
|------|-------------|------------------|
| `feature_store.yaml` | Feast project config | L11 §2.3 |
| `entities.py` | User + Product entity definitions | L11 §3.1 |
| `feature_views.py` | Multiple FeatureViews + on-demand | L11 §3.1–3.2 |
| `materialize.py` | Offline → online materialization | L11 §4.3 |
| `serve_features.py` | Historical + online retrieval demo | L11 §4.2–4.3 |

## Connection to L12 Practical Project

The `practical_project/` directory uses the same entity and feature schema.
This standalone example focuses on **Feature Store concepts** in isolation,
while `practical_project/` integrates them into the full MLOps pipeline.
