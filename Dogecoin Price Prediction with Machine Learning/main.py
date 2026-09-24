"""
DOGE-USD Price Forecasting with SARIMAX
--------------------------------------------
Forecasts DOGE-USD closing price using a SARIMAX time series model with
engineered price/volume features as exogenous regressors. Covers data
loading, cleaning, exploration, feature engineering, train/test
splitting, model fitting, and forecast visualization.
"""

from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "DOGE-USD.csv"
    tail_n: int = 30
    train_size: int = 11
    sarimax_order: tuple = (2, 1, 1)
    feature_columns: tuple = ("Close", "Volume", "gap", "a", "b")


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the price CSV."""
    data = pd.read_csv(cfg.csv_path)
    print(data.head())
    return data


def explore_data(data: pd.DataFrame) -> None:
    """Print correlation matrix and missing-value diagnostics."""
    print(data.corr(numeric_only=True))
    print(data.isnull().any())
    print(data.isnull().sum())


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, set as index, and drop missing rows."""
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    data = data.dropna()
    print(data.describe())
    return data


def plot_close_price(data: pd.DataFrame) -> None:
    """Plot mean closing price over time."""
    plt.figure(figsize=(20, 7))
    x = data.groupby("Date")["Close"].mean()
    x.plot(linewidth=2.5, color="b")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Date vs Close")
    plt.show()


def engineer_features(data: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Derive price/volume features and select the ones used for modeling."""
    data = data.copy()
    data["gap"] = (data["High"] - data["Low"]) * data["Volume"]
    data["y"] = data["High"] / data["Volume"]
    data["z"] = data["Low"] / data["Volume"]
    data["a"] = data["High"] / data["Low"]
    data["b"] = (data["High"] / data["Low"]) * data["Volume"]

    print(abs(data.corr(numeric_only=True)["Close"].sort_values(ascending=False)))

    data = data[list(cfg.feature_columns)]
    print(data.head())
    return data


def split_train_test(data: pd.DataFrame, cfg: PipelineConfig):
    """Take the last `tail_n` rows and split into train/test in chronological order."""
    recent = data.tail(cfg.tail_n)
    train = recent[: cfg.train_size]
    test = recent[cfg.train_size :]
    print(train.shape, test.shape)
    return train, test


def fit_sarimax(train: pd.DataFrame, cfg: PipelineConfig):
    """Fit a SARIMAX model on the training set using the other features as exogenous regressors."""
    model = SARIMAX(
        endog=train["Close"], exog=train.drop("Close", axis=1), order=cfg.sarimax_order
    )
    results = model.fit()
    print(results.summary())
    return results


def forecast(results, train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Predict Close price over the test period using the fitted model."""
    start = len(train)
    end = len(train) + len(test) - 1
    predictions = results.predict(start=start, end=end, exog=test.drop("Close", axis=1))
    print(predictions)
    return predictions


def plot_forecast(test: pd.DataFrame, predictions: pd.Series) -> None:
    """Plot actual vs. predicted Close price over the test period."""
    plt.figure(figsize=(12, 6))
    test["Close"].plot(legend=True, label="Actual")
    predictions.plot(label="Forecast", legend=True)
    plt.title("Actual vs. Forecast Close Price")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, clean it, engineer features, fit SARIMAX, and visualize the forecast."""
    data = load_data(cfg)
    explore_data(data)

    data = clean_data(data)
    plot_close_price(data)

    data = engineer_features(data, cfg)
    train, test = split_train_test(data, cfg)

    results = fit_sarimax(train, cfg)
    predictions = forecast(results, train, test)

    plot_forecast(test, predictions)


if __name__ == "__main__":
    run_pipeline()