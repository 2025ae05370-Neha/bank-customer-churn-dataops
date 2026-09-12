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

from tasks import data_ingestion, data_preprocessing
from tasks import eda_correlation, eda_feature_engineering
from tasks import eda_feature_importance, eda_visualization

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


@flow(name=FLOW_NAME, log_prints=True)
def bank_customer_churn_flow():
    """Execute the full ingestion, preprocessing and EDA pipeline in order."""
    logger = get_run_logger()
    logger.info("Starting Bank Customer Churn DataOps pipeline")

    run_data_ingestion()
    run_data_preprocessing()
    run_eda_correlation()
    run_eda_feature_engineering()
    run_eda_feature_importance()
    run_eda_visualization()

    logger.info("Bank Customer Churn DataOps pipeline complete")


if __name__ == "__main__":
    bank_customer_churn_flow.serve(
        name=DEPLOYMENT_NAME,
        interval=SCHEDULE_INTERVAL_SECONDS,
    )
