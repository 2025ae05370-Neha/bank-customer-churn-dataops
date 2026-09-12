"""Shared configuration for the Bank Customer Churn data pipeline.

Every task module imports its paths, column groupings and logger from here so
the pipeline stays consistent and each step remains independently runnable.
"""

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "BankCustomerChurnPrediction.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PROCESSED_DATA_PATH = OUTPUT_DIR / "processed_bank_churn.csv"
ENCODED_DATA_PATH = OUTPUT_DIR / "encoded_bank_churn.csv"

# Columns that identify a customer but carry no predictive signal.
IDENTIFIER_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# Continuous numeric features -- candidates for imputation, scaling and binning.
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]

# Already 0/1 encoded, so they are excluded from scaling but used by models.
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]

# Free-text categories that must be encoded before modelling.
CATEGORICAL_FEATURES = ["Geography", "Gender"]

# 1 = the customer churned (left the bank), 0 = retained.
TARGET = "Exited"

RANDOM_STATE = 42


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints to stdout and to the Prefect run log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger(name)
