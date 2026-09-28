"""Prefect workflow for the Bank Customer Churn DataOps pipeline.

Run once from API_Driven:
    python -m flows.bank_churn_dataops

Start a local two-minute schedule:
    python flows/bank_churn_dataops.py

The task modules remain independently runnable. This file only provides the
orchestration layer required for activity 1.5 of the assignment.
"""

import sys
from pathlib import Path

# Make the project root importable when this file is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact

from tasks import data_ingestion, data_preprocessing
from tasks import eda_correlation, eda_feature_engineering
from tasks import eda_feature_importance, eda_visualization
from tasks.config import NUMERIC_FEATURES, TARGET

FLOW_NAME = "bank-customer-churn-dataops"
DEPLOYMENT_NAME = "bank-customer-churn-local"
SCHEDULE_INTERVAL_SECONDS = 120
TASK_SEQUENCE = [
    "data-ingestion",
    "data-preprocessing",
    "eda-correlation",
    "eda-feature-engineering",
    "eda-feature-importance",
    "eda-visualization",
]


@task(name="data-ingestion", retries=1, retry_delay_seconds=5)
def run_data_ingestion():
    """Load and validate the source dataset."""
    logger = get_run_logger()
    logger.info("Starting data ingestion")
    result = data_ingestion.run()
    logger.info("Data ingestion complete: shape=%s", result.shape)
    return result


@task(name="data-preprocessing", retries=1, retry_delay_seconds=5)
def run_data_preprocessing():
    """Summarize, clean, impute and normalize the source dataset."""
    logger = get_run_logger()
    logger.info("Starting data preprocessing")
    result = data_preprocessing.run()
    logger.info("Data preprocessing complete: shape=%s", result.shape)
    return result


@task(name="eda-correlation")
def run_eda_correlation():
    """Run numeric and categorical association analysis."""
    logger = get_run_logger()
    logger.info("Starting correlation and statistical EDA")
    result = eda_correlation.run()
    logger.info("Correlation and statistical EDA complete")
    return result


@task(name="eda-feature-engineering")
def run_eda_feature_engineering():
    """Run binning and categorical encoding demonstrations."""
    logger = get_run_logger()
    logger.info("Starting binning and encoding EDA")
    result = eda_feature_engineering.run()
    logger.info("Binning and encoding EDA complete: shape=%s", result.shape)
    return result


@task(name="eda-feature-importance")
def run_eda_feature_importance():
    """Train the tree models and report feature importance."""
    logger = get_run_logger()
    logger.info("Starting feature-importance EDA")
    result = eda_feature_importance.run()
    logger.info("Feature-importance EDA complete: %s", result)
    return result


@task(name="eda-visualization")
def run_eda_visualization():
    """Generate and save univariate and bivariate charts."""
    logger = get_run_logger()
    logger.info("Starting visualization EDA")
    result = eda_visualization.run()
    logger.info("Visualization EDA complete: %d charts", len(result["charts"]))
    return result


def publish_dashboard_artifacts(raw, processed, correlation, importance, visualization):
    """Publish run outputs to the Prefect Cloud Artifacts tab (activity 1.5)."""
    rows, cols = raw.shape
    churn_rate = round(float(raw[TARGET].mean() * 100), 2)

    create_table_artifact(
        key="dataset-summary",
        table=[
            {"metric": "Rows", "value": int(rows)},
            {"metric": "Columns", "value": int(cols)},
            {"metric": "Churn rate (%)", "value": churn_rate},
            {"metric": "Missing values", "value": int(processed.isnull().sum().sum())},
            {"metric": "Normalized numeric features", "value": len(NUMERIC_FEATURES)},
        ],
        description="Dataset and preprocessing summary (activities 1.2, 1.3).",
    )

    create_table_artifact(
        key="chi-square-associations",
        table=[
            {
                "feature": feature,
                "chi2": round(float(result["chi2"]), 3),
                "p_value": round(float(result["p_value"]), 4),
                "conclusion": result["conclusion"],
            }
            for feature, result in correlation["chi_square"].items()
        ],
        description="Chi-square test of each categorical feature vs churn (activity 1.4).",
    )

    create_table_artifact(
        key="model-feature-importance",
        table=[
            {
                "model": name,
                "accuracy": round(float(result["accuracy"]), 4),
                "top_feature": str(result["top_feature"]),
            }
            for name, result in importance.items()
        ],
        description="Model accuracy and most important churn driver (activity 1.4).",
    )

    pearson = correlation["pearson"]
    chart_list = "\n".join(
        f"- `{Path(chart).name}`" for chart in visualization["charts"]
    )
    create_markdown_artifact(
        key="run-summary",
        markdown=(
            "# Bank Customer Churn - Run Summary\n\n"
            f"- **Records processed:** {int(rows)} rows x {int(cols)} columns\n"
            f"- **Churn rate:** {churn_rate}%\n"
            f"- **Pearson r({pearson['pair'][0]}, {pearson['pair'][1]}):** "
            f"{round(float(pearson['r']), 4)} ({pearson['interpretation']})\n\n"
            "## Generated charts\n"
            f"{chart_list}\n"
        ),
        description="High-level summary of the DataOps run (activity 1.5).",
    )


@flow(name=FLOW_NAME, log_prints=True)
def bank_customer_churn_flow():
    """Execute the full ingestion, preprocessing and EDA pipeline in order."""
    logger = get_run_logger()
    logger.info("Starting Bank Customer Churn DataOps pipeline")

    raw = run_data_ingestion()
    processed = run_data_preprocessing()
    correlation = run_eda_correlation()
    run_eda_feature_engineering()
    importance = run_eda_feature_importance()
    visualization = run_eda_visualization()

    publish_dashboard_artifacts(raw, processed, correlation, importance, visualization)

    logger.info("Bank Customer Churn DataOps pipeline complete")


if __name__ == "__main__":
    bank_customer_churn_flow.serve(
        name=DEPLOYMENT_NAME,
        interval=SCHEDULE_INTERVAL_SECONDS,
    )
