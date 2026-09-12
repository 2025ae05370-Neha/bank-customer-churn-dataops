"""Step 1.4d -- EDA: visualisation (univariate and bivariate).

Saves five figures to output/:
    Univariate  - churn_distribution.png, numeric_distributions.png
    Bivariate   - age_vs_churn_boxplot.png, balance_vs_salary_scatter.png,
                  churn_by_geography_gender.png
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so the task runs inside an orchestrator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tasks.config import NUMERIC_FEATURES, OUTPUT_DIR, TARGET, get_logger
from tasks.data_ingestion import load_bank_customer_churn_data

logger = get_logger("EDAVisualization")

sns.set_theme(style="whitegrid")


def _save(fig, filename: str) -> str:
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return str(path)


def plot_churn_distribution(df: pd.DataFrame, filename: str = "churn_distribution.png") -> str:
    """Univariate: how many customers stayed vs left."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette="Set2", legend=False, ax=ax)
    ax.set_title("Customer Churn Distribution (0 = Retained, 1 = Churned)")
    ax.set_xlabel("Exited")
    return _save(fig, filename)


def plot_numeric_distributions(df: pd.DataFrame, filename: str = "numeric_distributions.png") -> str:
    """Univariate: histogram for each continuous feature."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.flatten(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=column, bins=30, kde=True, color="steelblue", ax=ax)
        ax.set_title(f"Distribution of {column}")
    fig.suptitle("Univariate Analysis - Numeric Features", y=1.02)
    fig.tight_layout()
    return _save(fig, filename)


def plot_age_vs_churn(df: pd.DataFrame, filename: str = "age_vs_churn_boxplot.png") -> str:
    """Bivariate: age spread of churned vs retained customers."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=TARGET, y="Age", hue=TARGET, palette="Set3", legend=False, ax=ax)
    ax.set_title("Age Distribution by Churn Status")
    return _save(fig, filename)


def plot_balance_vs_salary(df: pd.DataFrame, filename: str = "balance_vs_salary_scatter.png") -> str:
    """Bivariate: account balance against estimated salary, coloured by churn."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="EstimatedSalary", y="Balance", hue=TARGET, alpha=0.4, ax=ax)
    ax.set_title("Balance vs Estimated Salary by Churn Status")
    return _save(fig, filename)


def plot_churn_by_geography_gender(df: pd.DataFrame, filename: str = "churn_by_geography_gender.png") -> str:
    """Bivariate: churn rate faceted by geography and gender."""
    grid = sns.catplot(
        data=df, x="Gender", y=TARGET, col="Geography", kind="bar",
        hue="Gender", palette="pastel", legend=False, height=3.5,
    )
    grid.set_axis_labels("Gender", "Churn rate")
    grid.figure.suptitle("Churn Rate by Geography and Gender", y=1.05)
    return _save(grid.figure, filename)


def run() -> dict:
    """Entry point used both standalone and by the DataOps flow."""
    df = load_bank_customer_churn_data()
    charts = [
        plot_churn_distribution(df),
        plot_numeric_distributions(df),
        plot_age_vs_churn(df),
        plot_balance_vs_salary(df),
        plot_churn_by_geography_gender(df),
    ]
    logger.info("Generated %d charts in %s", len(charts), OUTPUT_DIR)
    return {"output_dir": str(OUTPUT_DIR), "charts": charts}


if __name__ == "__main__":
    run()
