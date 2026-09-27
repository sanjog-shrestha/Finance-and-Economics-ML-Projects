"""
Online Payment Fraud Detection — Model Comparison
--------------------------------------------------------
Cleans and encodes a mobile-money transaction dataset, then trains and
compares three classifiers (Logistic Regression, XGBoost, Random Forest)
on predicting fraudulent transactions. Covers exploratory analysis,
categorical encoding, train/test splitting, ROC-AUC comparison, and a
confusion matrix.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as ras, ConfusionMatrixDisplay
from xgboost import XGBClassifier


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "new_data.csv"
    target_column: str = "isFraud"
    drop_columns: tuple = ("isFraud", "type", "nameOrig", "nameDest")
    test_size: float = 0.3
    random_state: int = 42
    rf_n_estimators: int = 7
    rf_criterion: str = "entropy"
    rf_random_state: int = 7


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the transaction dataset."""
    data = pd.read_csv(cfg.csv_path)
    print(data.head())
    data.info()
    print(data.describe())
    return data


def summarize_column_types(data: pd.DataFrame) -> None:
    """Print counts of categorical, integer, and float columns."""
    object_cols = list(data.dtypes[data.dtypes == "object"].index)
    print("Categorical variables:", len(object_cols))

    num_cols = list(data.dtypes[data.dtypes == "int"].index)
    print("Integer variables:", len(num_cols))

    fl_cols = list(data.dtypes[data.dtypes == "float"].index)
    print("Float variables:", len(fl_cols))


def plot_transaction_type_counts(data: pd.DataFrame) -> None:
    """Bar plot of transaction type counts."""
    sns.countplot(x="type", data=data)
    plt.show()


def plot_fraud_counts(data: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Print fraud vs. non-fraud counts."""
    print(data[cfg.target_column].value_counts())


def plot_step_distribution(data: pd.DataFrame) -> None:
    """Plot the distribution of the 'step' (time) column."""
    plt.figure(figsize=(15, 6))
    sns.histplot(data["step"], bins=50, kde=True)
    plt.show()


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    """Heatmap of factorized-column correlations (handles categorical columns too)."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        data.apply(lambda x: pd.factorize(x)[0]).corr(),
        cmap="BrBG", fmt=".2f", linewidths=2, annot=True,
    )
    plt.show()


def encode_and_prepare(data: pd.DataFrame, cfg: PipelineConfig):
    """One-hot encode transaction type, drop identifier columns, and clean up dtypes."""
    type_dummies = pd.get_dummies(data["type"], drop_first=True)
    data_new = pd.concat([data, type_dummies], axis=1)
    print(data_new.head())

    X = data_new.drop(list(cfg.drop_columns), axis=1)
    y = data_new[cfg.target_column]

    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop(cfg.target_column, axis=1)
    y = combined[cfg.target_column]

    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)

    print(X.shape, y.shape)
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, cfg: PipelineConfig):
    """Train/test split."""
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)


def train_and_compare_models(X_train, y_train, X_test, y_test, cfg: PipelineConfig) -> list:
    """Train Logistic Regression, XGBoost, and Random Forest, printing ROC-AUC for each."""
    models = [
        LogisticRegression(),
        XGBClassifier(),
        RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            criterion=cfg.rf_criterion,
            random_state=cfg.rf_random_state,
        ),
    ]

    for model in models:
        model.fit(X_train, y_train)
        print(f"{model} : ")

        train_preds = model.predict_proba(X_train)[:, 1]
        print("Training Accuracy : ", ras(y_train, train_preds))

        test_preds = model.predict_proba(X_test)[:, 1]
        print("Validation Accuracy : ", ras(y_test, test_preds))
        print()

    return models


def plot_confusion_matrix(model, X_test, y_test) -> None:
    """Confusion matrix for the given model on the test set."""
    cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    cm.plot(cmap="Blues")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, explore it, encode features, train models, and evaluate."""
    data = load_data(cfg)
    summarize_column_types(data)

    plot_transaction_type_counts(data)
    plot_fraud_counts(data, cfg)
    plot_step_distribution(data)
    plot_correlation_heatmap(data)

    X, y = encode_and_prepare(data, cfg)
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)

    models = train_and_compare_models(X_train, y_train, X_test, y_test, cfg)
    plot_confusion_matrix(models[1], X_test, y_test)  # XGBoost


if __name__ == "__main__":
    run_pipeline()