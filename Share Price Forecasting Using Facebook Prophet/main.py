"""
Share Price Forecasting with Facebook Prophet.

Fits a Prophet time-series model on the historical adjusted closing price of a
stock (Amazon by default), forecasts the held-out test period, plots the
forecast and its trend/seasonality components, and reports MSE, MAE and MAPE.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "Share-Price-Forecasting-Using-Facebook-Prophet---Dataset.csv"          # price data downloaded from Yahoo Finance
    date_column: str = "Date"
    target_column: str = "Adj Close"
    split_date: str = "2019-07-21"      # train: <= split_date, test: > split_date
    plot_style: str = "fivethirtyeight"
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    """Read the raw price CSV."""
    df = pd.read_csv(config.csv_path)
    print(f"Loaded {len(df)} rows from {config.csv_path}")
    print(df.head())
    return df


def prepare_prophet_frame(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Rename columns to the `ds` (date) / `y` (value) format Prophet expects."""
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df[config.date_column]),
        "y": df[config.target_column].astype(float),
    })
    return prophet_df


def split_by_date(df: pd.DataFrame, config: PipelineConfig):
    """Chronological split: everything up to split_date trains, the rest tests."""
    cutoff = pd.Timestamp(config.split_date)
    train = df.loc[df["ds"] <= cutoff].copy()
    test = df.loc[df["ds"] > cutoff].copy()
    print(f"Train: {len(train)} rows ({train['ds'].min().date()} -> {train['ds'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['ds'].min().date()} -> {test['ds'].max().date()})")
    return train, test


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def train_prophet(train: pd.DataFrame) -> Prophet:
    """Fit a default Prophet model on the training period."""
    model = Prophet()
    model.fit(train)
    return model


def forecast_test_period(model: Prophet, test: pd.DataFrame) -> pd.DataFrame:
    """Predict prices on the exact trading dates of the test period."""
    forecast = model.predict(test[["ds"]])
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    return forecast


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate_forecast(test: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Compare actual test prices against Prophet's point forecast (yhat)."""
    y_true = test["y"].to_numpy()
    y_pred = forecast["yhat"].to_numpy()

    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred),
    }
    print("\nForecast error on test period")
    for name, value in metrics.items():
        print(f"  {name:<9}: {value:.4f}")
    return metrics


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_forecast(model: Prophet, forecast: pd.DataFrame) -> None:
    """Prophet's built-in plot: training history, forecast and uncertainty band."""
    fig = model.plot(forecast)
    fig.suptitle("Prophet forecast")


def plot_components(model: Prophet, forecast: pd.DataFrame) -> None:
    """Trend, weekly and yearly seasonality components."""
    fig = model.plot_components(forecast)
    fig.suptitle("Forecast components")


def plot_actual_vs_forecast(test: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Overlay actual test-period prices on the forecast."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test["ds"], test["y"], label="Actual", linewidth=1.5)
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast (yhat)", linewidth=1.5)
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.2, label="Uncertainty interval")
    ax.set_title("Test period: actual vs. forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> dict:
    plt.style.use(config.plot_style)

    raw = load_data(config)
    data = prepare_prophet_frame(raw, config)
    train, test = split_by_date(data, config)

    model = train_prophet(train)
    forecast = forecast_test_period(model, test)

    plot_forecast(model, forecast)
    plot_components(model, forecast)
    plot_actual_vs_forecast(test, forecast)

    metrics = evaluate_forecast(test, forecast)

    if config.show_plots:
        plt.show()
    return metrics


if __name__ == "__main__":
    run_pipeline(PipelineConfig())