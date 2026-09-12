"""FastAPI endpoints for assignment activities 3.1-3.3.

The API exposes application metadata, dataset facts, preprocessing summaries,
model performance and generated artifacts for the Bank Customer Churn project.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flows.bank_churn_dataops import (  # noqa: E402
    DEPLOYMENT_NAME,
    FLOW_NAME,
    SCHEDULE_INTERVAL_SECONDS,
    TASK_SEQUENCE,
)
from tasks.config import (  # noqa: E402
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    IDENTIFIER_COLUMNS,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET,
)
from tasks.data_ingestion import load_bank_customer_churn_data  # noqa: E402
from tasks.eda_feature_importance import prepare_features, train_and_rank  # noqa: E402

PROJECT_NAME = "Bank Customer Churn - Cloud Data Science Pipeline"
PROJECT_DESCRIPTION = (
    "API access for the Prefect-based Bank Customer Churn pipeline built for "
    "AIMLCZG549 Assignment I."
)

app = FastAPI(
    title="Bank Customer Churn DataOps API",
    version="1.0.0",
    description=PROJECT_DESCRIPTION,
)


@lru_cache(maxsize=1)
def _load_dataset_summary() -> dict:
    df = load_bank_customer_churn_data()
    return {
        "dataset_name": DATASET_PATH.name,
        "dataset_path": str(DATASET_PATH),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": TARGET,
        "churn_rate_percent": round(float(df[TARGET].mean() * 100), 2),
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "categorical_feature_count": len(CATEGORICAL_FEATURES),
        "binary_feature_count": len(BINARY_FEATURES),
    }


@lru_cache(maxsize=1)
def _load_preprocessing_summary() -> dict:
    df = load_bank_customer_churn_data()
    missing_by_column = {
        column: int(value)
        for column, value in df.isnull().sum().items()
        if value > 0
    }
    normalized_columns = [f"{column}_normalized" for column in NUMERIC_FEATURES]
    return {
        "identifier_columns_removed": IDENTIFIER_COLUMNS,
        "missing_total": int(df.isnull().sum().sum()),
        "missing_by_column": missing_by_column,
        "normalized_columns": normalized_columns,
        "normalized_column_count": len(normalized_columns),
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


@lru_cache(maxsize=1)
def _load_model_summary() -> dict:
    df = load_bank_customer_churn_data()
    X, y = prepare_features(df)
    rankings = train_and_rank(X, y)

    models = {}
    for name, result in rankings.items():
        importances = result["importances"]
        models[name] = {
            "accuracy": round(float(result["accuracy"]), 4),
            "top_feature": str(importances.idxmax()),
            "top_3_features": [
                {"feature": str(feature), "importance": round(float(score), 4)}
                for feature, score in importances.head(3).items()
            ],
        }

    return {
        "trained_models": list(models.keys()),
        "models": models,
    }


def _list_output_artifacts() -> list[dict]:
    artifacts = []
    for path in sorted(OUTPUT_DIR.glob("*")):
        if not path.is_file():
            continue
        modified_at = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        artifacts.append(
            {
                "file_name": path.name,
                "file_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "modified_at_utc": modified_at,
            }
        )
    return artifacts


def _flow_summary() -> dict:
    return {
        "flow_name": FLOW_NAME,
        "deployment_name": DEPLOYMENT_NAME,
        "schedule_interval_seconds": SCHEDULE_INTERVAL_SECONDS,
        "schedule_interval_minutes": SCHEDULE_INTERVAL_SECONDS / 60,
        "python_entrypoint": "flows.bank_churn_dataops:bank_customer_churn_flow",
        "task_count": len(TASK_SEQUENCE),
        "task_sequence": TASK_SEQUENCE,
        "prefect_api_url": os.getenv("PREFECT_API_URL"),
    }


@app.get("/")
def root() -> dict:
    """Landing route with quick links to the assignment APIs."""
    return {
        "project": PROJECT_NAME,
        "message": "Use the documented endpoints to retrieve pipeline details and screenshots for activities 3.1-3.3.",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "application_details": "/api/v1/application/details",
            "flow_details": "/api/v1/flow/details",
            "dataset_details": "/api/v1/dataset/details",
            "preprocessing_details": "/api/v1/preprocessing/details",
            "model_details": "/api/v1/model/details",
            "output_artifacts": "/api/v1/output/artifacts",
        },
    }


@app.get("/health")
def health() -> dict:
    """Basic service health for API testing."""
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_available": DATASET_PATH.is_file(),
        "output_directory_available": OUTPUT_DIR.is_dir(),
    }


@app.get("/api/v1/application/details")
def application_details() -> dict:
    """Return four-plus key application details in one response."""
    try:
        dataset = _load_dataset_summary()
        flow = _flow_summary()
        artifacts = _list_output_artifacts()
        models = _load_model_summary()
    except Exception as exc:  # pragma: no cover - surfaced to API clients
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build application details: {exc}",
        ) from exc

    return {
        "project_name": PROJECT_NAME,
        "key_application_details": [
            {"label": "Flow name", "value": flow["flow_name"]},
            {"label": "Deployment name", "value": flow["deployment_name"]},
            {
                "label": "Schedule",
                "value": f"Every {int(flow['schedule_interval_minutes'])} minutes",
            },
            {"label": "Dataset rows", "value": dataset["rows"]},
            {"label": "Dataset columns", "value": dataset["columns"]},
            {"label": "Churn rate (%)", "value": dataset["churn_rate_percent"]},
            {"label": "Task count", "value": flow["task_count"]},
            {"label": "Generated artifacts", "value": len(artifacts)},
        ],
        "dataset": dataset,
        "flow": flow,
        "model_accuracy": {
            name: details["accuracy"] for name, details in models["models"].items()
        },
    }


@app.get("/api/v1/flow/details")
def flow_details() -> dict:
    """Expose workflow and deployment information for activity 3.1."""
    return _flow_summary()


@app.get("/api/v1/dataset/details")
def dataset_details() -> dict:
    """Expose dataset-level facts used by the project."""
    try:
        return _load_dataset_summary()
    except Exception as exc:  # pragma: no cover - surfaced to API clients
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset details: {exc}",
        ) from exc


@app.get("/api/v1/preprocessing/details")
def preprocessing_details() -> dict:
    """Expose preprocessing-related facts required by the assignment."""
    try:
        return _load_preprocessing_summary()
    except Exception as exc:  # pragma: no cover - surfaced to API clients
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load preprocessing details: {exc}",
        ) from exc


@app.get("/api/v1/model/details")
def model_details() -> dict:
    """Expose model performance and top feature importances."""
    try:
        return _load_model_summary()
    except Exception as exc:  # pragma: no cover - surfaced to API clients
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build model details: {exc}",
        ) from exc


@app.get("/api/v1/output/artifacts")
def output_artifacts() -> dict:
    """List generated CSV and chart artifacts for the pipeline."""
    artifacts = _list_output_artifacts()
    return {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
