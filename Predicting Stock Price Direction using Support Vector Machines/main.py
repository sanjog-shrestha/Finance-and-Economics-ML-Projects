"""
Stock Direction Prediction + Trading Strategy Backtest (SVM)
------------------------------------------------------------------
Predicts whether a stock's next-day closing price will go up or down
using an SVM classifier on simple price-spread features, compares
several SVM kernels, and backtests a basic long/flat trading strategy
built from the model's predictions — evaluated on the held-out test
period only, to avoid grading the strategy on data the model trained on.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "RELIANCE.csv"
    train_fraction: float = 0.8
    kernels: tuple = ("rbf", "linear", "poly", "sigmoid")
    poly_degree: int = 3


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the price CSV and set Date as the index."""
    df = pd.read_csv(cfg.csv_path)
    print(df.head())

    df.index = pd.to_datetime(df["Date"])
    df = df.drop(["Date"], axis="columns")
    return df


def engineer_features_and_target(df: pd.DataFrame):
    """Build simple price-spread features and the next-day up/down target."""
    df = df.copy()
    df["Open-Close"] = df.Open - df.Close
    df["High-Low"] = df.High - df.Low

    X = df[["Open-Close", "High-Low"]]
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    print(X.head())
    return df, X, y


def split_data(X: pd.DataFrame, y: np.ndarray, cfg: PipelineConfig):
    """Chronological train/test split (no shuffling, since this is time series)."""
    split = int(cfg.train_fraction * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test, split


def train_default_svm(X_train, y_train) -> SVC:
    """Train the baseline SVC (default RBF kernel)."""
    return SVC().fit(X_train, y_train)


def backtest_strategy(df: pd.DataFrame, model: SVC, X: pd.DataFrame, split: int) -> pd.DataFrame:
    """Build a long/flat trading strategy from test-period predictions only.

    Predictions are generated only for the test period (rows after `split`)
    so the backtest reflects genuinely out-of-sample performance, rather
    than mixing in predictions the model already saw during training.
    """
    df = df.copy()
    df["Predicted_Signal"] = np.nan
    df.iloc[split:, df.columns.get_loc("Predicted_Signal")] = model.predict(X[split:])

    df["Return"] = df.Close.pct_change()
    df["Strategy_Return"] = df.Return * df.Predicted_Signal.shift(1)

    df["Cum_Ret"] = df["Return"].cumsum()
    df["Cum_Strategy"] = df["Strategy_Return"].cumsum()
    return df


def plot_strategy_returns(df: pd.DataFrame, split: int) -> None:
    """Plot buy-and-hold vs. strategy cumulative returns over the test period."""
    test_df = df.iloc[split:]
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["Cum_Ret"], color="red", label="Buy & Hold")
    plt.plot(test_df["Cum_Strategy"], color="blue", label="Strategy")
    plt.title("Cumulative Returns (Test Period Only)")
    plt.legend()
    plt.show()


def evaluate_model(model: SVC, X_train, y_train, X_test, y_test) -> dict:
    """Report training vs. test accuracy, plus ROC-AUC and F1 on the test set."""
    y_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    for name, value in metrics.items():
        print(f"{name}: {value}")
    return metrics


def compare_kernels(X_train, y_train, X_test, y_test, cfg: PipelineConfig) -> dict:
    """Train an SVC per kernel and compare test accuracy."""
    results = {}
    for kernel in cfg.kernels:
        kwargs = {"degree": cfg.poly_degree} if kernel == "poly" else {}
        model = SVC(kernel=kernel, **kwargs).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[kernel] = accuracy
        print(f"Accuracy with {kernel} kernel: {accuracy}")
    return results


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, engineer features, train/evaluate SVM models, and backtest a strategy."""
    df = load_data(cfg)
    df, X, y = engineer_features_and_target(df)

    X_train, X_test, y_train, y_test, split = split_data(X, y, cfg)

    model = train_default_svm(X_train, y_train)
    evaluate_model(model, X_train, y_train, X_test, y_test)

    df = backtest_strategy(df, model, X, split)
    plot_strategy_returns(df, split)

    compare_kernels(X_train, y_train, X_test, y_test, cfg)


if __name__ == "__main__":
    run_pipeline()