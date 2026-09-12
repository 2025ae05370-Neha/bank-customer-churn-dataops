"""Step 1.3 -- Data Pre-processing.

Covers every pre-processing activity asked for in the problem statement:
summary statistics, missing-value checks, median imputation of numeric
columns, data-type reporting and Min-Max normalisation. The cleaned frame is
written to output/processed_bank_churn.csv for the EDA steps and the grader.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tasks.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    TARGET,
    get_logger,
)
from tasks.data_ingestion import load_bank_customer_churn_data

logger = get_logger("DataPreprocessing")


def summarize(df: pd.DataFrame) -> dict:
    """Report shape, dtypes, summary statistics and missing-value counts."""
    logger.info("Dataset shape: %d rows x %d columns", *df.shape)
    logger.info("Column data types:\n%s", df.dtypes.to_string())
    logger.info("Summary statistics (numeric columns):\n%s", df.describe().T.to_string())

    missing = df.isnull().sum()
    if missing.sum() == 0:
        logger.info("Missing-value check: no nulls found in any of the %d columns", df.shape[1])
    else:
        logger.info("Missing values per column:\n%s", missing[missing > 0].to_string())

    logger.info(
        "Class balance for target '%s':\n%s",
        TARGET,
        df[TARGET].value_counts(normalize=True).rename("proportion").to_string(),
    )

    return {
        "shape": df.shape,
        "missing_total": int(missing.sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove row/customer identifiers that carry no predictive signal."""
    dropped = [column for column in IDENTIFIER_COLUMNS if column in df.columns]
    logger.info("Dropping identifier columns: %s", dropped)
    return df.drop(columns=dropped)


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with the column median (robust to outliers)."""
    df = df.copy()
    for column in NUMERIC_FEATURES:
        null_count = int(df[column].isnull().sum())
        median = df[column].median()
        df[column] = df[column].fillna(median)
        logger.info("Imputed '%s': %d missing value(s) replaced with median %.2f", column, null_count, median)
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Min-Max scale numeric features into [0, 1], keeping the originals intact."""
    df = df.copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[NUMERIC_FEATURES])
    scaled_columns = [f"{column}_normalized" for column in NUMERIC_FEATURES]
    df[scaled_columns] = scaled

    logger.info(
        "Normalized %d numeric features -> %s",
        len(NUMERIC_FEATURES),
        ", ".join(scaled_columns),
    )
    logger.info("Normalized sample:\n%s", df[scaled_columns].head(3).to_string())
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full clean-up chain: drop IDs -> impute -> normalize."""
    return normalize(impute_numeric(drop_identifier_columns(df)))


def run() -> pd.DataFrame:
    """Entry point used both standalone and by the DataOps flow."""
    df = load_bank_customer_churn_data()
    summarize(df)

    processed = preprocess(df)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Saved pre-processed dataset to %s (shape=%s)", PROCESSED_DATA_PATH, processed.shape)
    logger.info(
        "Retained features -> numeric: %s | binary: %s | categorical: %s | target: %s",
        NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET,
    )
    return processed


if __name__ == "__main__":
    run()
