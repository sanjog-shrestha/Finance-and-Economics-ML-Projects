"""
Bitcoin Price Direction Classification — Model Comparison
----------------------------------------------------------------
Predicts whether Bitcoin's next-day closing price will go up or down,
using engineered price features and comparing three classifiers
(Logistic Regression, SVM, XGBoost). Covers data loading, exploration,
date feature engineering, target construction, feature scaling,
training/evaluation, and a confusion matrix.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "bitcoin.csv"
    correlation_threshold: float = 0.9
    test_size: float = 0.3
    random_state: int = 42


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the Bitcoin price CSV."""
    df = pd.read_csv(cfg.csv_path)
    print(df.head())
    print(df.shape)
    print(df.describe())
    df.info()
    return df


def plot_close_price(df: pd.DataFrame) -> None:
    """Plot closing price over the full history."""
    plt.figure(figsize=(15, 5))
    plt.plot(df["Close"])
    plt.title("Bitcoin Close price.", fontsize=15)
    plt.ylabel("Price in dollars.")
    plt.show()


def drop_redundant_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 'Adj Close' after confirming it's identical to 'Close'."""
    print(df[df["Close"] == df["Adj Close"]].shape, df.shape)
    df = df.drop(["Adj Close"], axis=1)
    print(df.isnull().sum())
    return df


def plot_feature_distributions(df: pd.DataFrame, features: list) -> None:
    """Plot distribution and boxplot for each price feature."""
    plt.subplots(figsize=(20, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 2, i + 1)
        sn.histplot(df[col], kde=True)
    plt.show()

    plt.subplots(figsize=(20, 10))
    for i, col in enumerate(features):
        plt.subplot(2, 2, i + 1)
        sn.boxplot(x=df[col])
    plt.show()


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split the Date string into year/month/day columns, then parse it as a real date."""
    df = df.copy()
    splitted = df["Date"].str.split("-", expand=True)
    df["year"] = splitted[0].astype("int")
    df["month"] = splitted[1].astype("int")
    df["day"] = splitted[2].astype("int")
    df["Date"] = pd.to_datetime(df["Date"])
    print(df.head())
    return df


def plot_yearly_averages(df: pd.DataFrame) -> None:
    """Plot mean Open/High/Low/Close per year."""
    data_grouped = df.groupby("year").mean(numeric_only=True)
    plt.subplots(figsize=(20, 10))
    for i, col in enumerate(["Open", "High", "Low", "Close"]):
        plt.subplot(2, 2, i + 1)
        data_grouped[col].plot.bar()
    plt.show()


def engineer_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the modeling features and the next-day up/down target."""
    df = df.copy()
    df["is_quarter_end"] = np.where(df["month"] % 3 == 0, 1, 0)
    df["open-close"] = df["Open"] - df["Close"]
    df["low-high"] = df["Low"] - df["High"]
    df["target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    print(df.head())
    return df


def plot_target_balance(df: pd.DataFrame) -> None:
    """Pie chart of the up/down target class balance."""
    plt.pie(df["target"].value_counts().values, labels=[0, 1], autopct="%1.1f%%")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Heatmap of strongly correlated feature pairs."""
    plt.figure(figsize=(10, 10))
    sn.heatmap(df.corr(numeric_only=True) > cfg.correlation_threshold, annot=True, cbar=False)
    plt.show()


def prepare_features(df: pd.DataFrame):
    """Select and scale the final modeling features."""
    features = df[["open-close", "low-high", "is_quarter_end"]]
    target = df["target"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, target


def train_and_compare_models(X_train, Y_train, X_valid, Y_valid) -> list:
    """Train Logistic Regression, SVM, and XGBoost, printing ROC-AUC for each."""
    models = [LogisticRegression(), SVC(kernel="poly", probability=True), XGBClassifier()]

    for model in models:
        model.fit(X_train, Y_train)
        print(f"{model} : ")
        print("Training Accuracy : ", roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
        print("Validation Accuracy : ", roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]))
        print()

    return models


def plot_confusion_matrix(model, X_valid, Y_valid) -> None:
    """Confusion matrix for the first trained model (Logistic Regression) on the validation set."""
    ConfusionMatrixDisplay.from_estimator(model, X_valid, Y_valid, cmap="Blues")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, explore it, engineer features/target, train models, and evaluate."""
    df = load_data(cfg)
    plot_close_price(df)
    df = drop_redundant_column(df)

    price_features = ["Open", "High", "Low", "Close"]
    plot_feature_distributions(df, price_features)

    df = engineer_date_features(df)
    plot_yearly_averages(df)

    df = engineer_target_features(df)
    plot_target_balance(df)
    plot_correlation_heatmap(df, cfg)

    features, target = prepare_features(df)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features, target, test_size=cfg.test_size, random_state=cfg.random_state
    )

    models = train_and_compare_models(X_train, Y_train, X_valid, Y_valid)
    plot_confusion_matrix(models[0], X_valid, Y_valid)


if __name__ == "__main__":
    run_pipeline()