"""Step 1.2 -- Data Ingestion.

Loads the Bank Customer Churn dataset (10,000 customer records, Kaggle) and
performs the sanity checks every downstream task relies on: the file exists,
it parses, and the expected columns are present.
"""

from pathlib import Path

import pandas as pd

from tasks.config import CATEGORICAL_FEATURES, DATASET_PATH, NUMERIC_FEATURES, TARGET, get_logger

logger = get_logger("DataIngestion")

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def load_bank_customer_churn_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Read the bank customer churn CSV and return it as a DataFrame."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns from %s", *data.shape, path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    return data


def run() -> pd.DataFrame:
    """Entry point used both standalone and by the DataOps flow."""
    data = load_bank_customer_churn_data()
    logger.info("Dataset columns: %s", ", ".join(data.columns))
    logger.info("Churn rate in the raw data: %.2f%%", data[TARGET].mean() * 100)
    logger.info("Preview:\n%s", data.head(3).to_string())
    return data


if __name__ == "__main__":
    run()
