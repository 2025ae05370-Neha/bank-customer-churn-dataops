"""Step 1.4c -- EDA: feature importance.

Trains a Decision Tree and a Random Forest to predict churn (Exited) and
reports which customer attributes actually drive the outcome. Importance
rankings are logged and saved as bar charts in output/.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so the task runs inside an orchestrator

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from tasks.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET,
    get_logger,
)
from tasks.data_ingestion import load_bank_customer_churn_data

logger = get_logger("EDAFeatureImportance")

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df: pd.DataFrame):
    """Label-encode categorical columns so both tree models can consume them."""
    X = df[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        X[column] = LabelEncoder().fit_transform(X[column])

    y = df[TARGET]
    logger.info("Feature matrix %s built from: %s", X.shape, FEATURES)
    return X, y


def train_and_rank(X, y):
    """Fit both models on an 80/20 stratified split and rank their features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=200, max_depth=10
        ),
    }

    rankings = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        logger.info("%s test accuracy: %.3f", name, accuracy)
        logger.info("%s classification report:\n%s", name, classification_report(y_test, model.predict(X_test)))
        logger.info("%s feature importances:\n%s", name, importances.round(4).to_string())

        rankings[name] = {"accuracy": accuracy, "importances": importances}
    return rankings


def save_importance_plot(importances: pd.Series, title: str, filename: str) -> str:
    """Horizontal bar chart of feature importances."""
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh", color="teal")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path)
    plt.close()
    logger.info("Saved %s", path)
    return str(path)


def run() -> dict:
    """Entry point used both standalone and by the DataOps flow."""
    df = load_bank_customer_churn_data()
    X, y = prepare_features(df)
    rankings = train_and_rank(X, y)

    save_importance_plot(
        rankings["Decision Tree"]["importances"],
        "Decision Tree - Feature Importance (Churn)",
        "dt_feature_importance.png",
    )
    save_importance_plot(
        rankings["Random Forest"]["importances"],
        "Random Forest - Feature Importance (Churn)",
        "rf_feature_importance.png",
    )

    return {
        name: {
            "accuracy": round(result["accuracy"], 4),
            "top_feature": result["importances"].idxmax(),
        }
        for name, result in rankings.items()
    }


if __name__ == "__main__":
    run()
