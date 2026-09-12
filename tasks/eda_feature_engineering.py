"""Step 1.4b -- EDA: binning and encoding (feature engineering).

Turns raw columns into model-ready features:
    * Equal-width (distance) binning of Age via pd.cut.
    * Equal-depth (frequency) binning of CreditScore via pd.qcut.
    * One-hot encoding of Geography (nominal, no natural order).
    * Binary encoding of Gender (two categories).
    * Label encoding of the Age band (ordinal: Young < ... < Senior).

The engineered frame is written to output/encoded_bank_churn.csv.
"""

import pandas as pd

from tasks.config import ENCODED_DATA_PATH, get_logger
from tasks.data_ingestion import load_bank_customer_churn_data

logger = get_logger("EDAFeatureEngineering")

N_BINS = 5
AGE_BAND_LABELS = ["Young", "Young Adult", "Mid Age", "Senior", "Elder"]


def distance_binning(df: pd.DataFrame, column: str = "Age", bins: int = N_BINS) -> pd.Series:
    """Equal-width binning: every bin spans the same value range."""
    binned = pd.cut(df[column], bins=bins)
    logger.info(
        "Equal-width bins for '%s' (counts are unbalanced by design):\n%s",
        column,
        binned.value_counts().sort_index().to_string(),
    )
    return binned.rename(f"{column}_bin_width")


def frequency_binning(df: pd.DataFrame, column: str = "CreditScore", bins: int = N_BINS) -> pd.Series:
    """Equal-depth binning: every bin holds roughly the same number of rows."""
    binned = pd.qcut(df[column], q=bins, duplicates="drop")
    logger.info(
        "Equal-depth bins for '%s' (counts are balanced by construction):\n%s",
        column,
        binned.value_counts().sort_index().to_string(),
    )
    return binned.rename(f"{column}_bin_freq")


def one_hot_encode(df: pd.DataFrame, column: str = "Geography") -> pd.DataFrame:
    """One-hot encode a nominal column -- no ordering is implied."""
    encoded = pd.get_dummies(df[column], prefix=column).astype(int)
    logger.info("One-hot encoded '%s' -> %s", column, list(encoded.columns))
    return encoded


def binary_encode(df: pd.DataFrame, column: str = "Gender") -> pd.Series:
    """Map a two-category column onto 0/1."""
    categories = sorted(df[column].dropna().unique())
    mapping = {categories[0]: 0, categories[1]: 1}
    logger.info("Binary encoded '%s' using mapping %s", column, mapping)
    return df[column].map(mapping).rename(f"{column}_binary")


def label_encode_age_band(df: pd.DataFrame, column: str = "Age") -> pd.Series:
    """Label-encode Age bands so the ordinal Young < ... < Elder order is preserved."""
    bands = pd.cut(df[column], bins=len(AGE_BAND_LABELS), labels=AGE_BAND_LABELS)
    mapping = {label: index for index, label in enumerate(AGE_BAND_LABELS)}
    logger.info("Label encoded '%s' bands using order %s", column, AGE_BAND_LABELS)
    return bands.map(mapping).astype(int).rename(f"{column}_band_label")


def run() -> pd.DataFrame:
    """Entry point used both standalone and by the DataOps flow."""
    df = load_bank_customer_churn_data()

    engineered = pd.concat(
        [
            df,
            distance_binning(df),
            frequency_binning(df),
            one_hot_encode(df),
            binary_encode(df),
            label_encode_age_band(df),
        ],
        axis=1,
    )

    engineered.to_csv(ENCODED_DATA_PATH, index=False)
    logger.info("Engineered dataframe shape: %s (was %s)", engineered.shape, df.shape)
    logger.info("Saved encoded dataset to %s", ENCODED_DATA_PATH)
    return engineered


if __name__ == "__main__":
    run()
