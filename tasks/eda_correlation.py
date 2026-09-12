"""Step 1.4a -- EDA: correlation analysis.

Answers "which features move together, and which categories are associated
with churn?" using three complementary measures:
    * Pearson correlation matrix across all numeric columns (+ heatmap).
    * Pearson r with a p-value for a single hypothesis-driven pair.
    * Chi-square test of independence for categorical vs the churn target.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so the task runs inside an orchestrator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from tasks.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET,
    get_logger,
)
from tasks.data_ingestion import load_bank_customer_churn_data

logger = get_logger("EDACorrelation")

ALPHA = 0.05


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation across numeric features, binary flags and the target."""
    columns = NUMERIC_FEATURES + BINARY_FEATURES + [TARGET]
    corr = df[columns].corr()
    logger.info("Correlation matrix:\n%s", corr.round(3).to_string())

    churn_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    logger.info("Features ranked by |correlation| with churn:\n%s", churn_corr.round(3).to_string())
    return corr


def save_heatmap(corr: pd.DataFrame, filename: str = "correlation_heatmap.png") -> str:
    """Render the correlation matrix as an annotated heatmap."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap - Bank Customer Churn")
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path)
    plt.close()
    logger.info("Saved correlation heatmap to %s", path)
    return str(path)


def pearson_pair(df: pd.DataFrame, col_a: str = "Age", col_b: str = "Balance") -> dict:
    """Pearson r with significance test for one numeric pair."""
    r, p_value = stats.pearsonr(df[col_a], df[col_b])

    strength = "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.6 else "strong"
    direction = "positive" if r > 0 else "negative"
    interpretation = f"{strength} {direction} linear relationship"

    logger.info("Pearson r(%s, %s) = %.4f, p-value = %.4g -> %s", col_a, col_b, r, p_value, interpretation)
    return {"pair": (col_a, col_b), "r": r, "p_value": p_value, "interpretation": interpretation}


def chi_square_tests(df: pd.DataFrame) -> dict:
    """Test each categorical/binary column for independence from churn."""
    results = {}
    for column in CATEGORICAL_FEATURES + BINARY_FEATURES:
        contingency = pd.crosstab(df[column], df[TARGET])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

        associated = p_value < ALPHA
        conclusion = (
            f"'{column}' is associated with churn (reject H0)"
            if associated
            else f"'{column}' appears independent of churn (fail to reject H0)"
        )

        logger.info("Contingency table (%s x %s):\n%s", column, TARGET, contingency.to_string())
        logger.info("Chi-square = %.4f, dof = %d, p-value = %.4g -> %s", chi2, dof, p_value, conclusion)

        results[column] = {"chi2": chi2, "p_value": p_value, "dof": dof, "conclusion": conclusion}
    return results


def run() -> dict:
    """Entry point used both standalone and by the DataOps flow."""
    df = load_bank_customer_churn_data()
    corr = correlation_matrix(df)
    save_heatmap(corr)
    pearson = pearson_pair(df)
    chi_square = chi_square_tests(df)
    return {"pearson": pearson, "chi_square": chi_square}


if __name__ == "__main__":
    run()
