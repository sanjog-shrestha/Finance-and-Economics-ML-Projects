"""
Multi-Stock Exploration + Apple Close Price Forecasting (LSTM)
--------------------------------------------------------------------
Explores 5 years of daily prices for a handful of large-cap stocks, then
builds and trains an LSTM to forecast Apple's closing price from its own
price history. Covers data loading, per-company visualization, sequence
windowing, model training, and forecast evaluation/visualization.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "all_stocks_5yr.csv"
    companies: tuple = ("AAPL", "AMD", "FB", "GOOGL", "AMZN", "NVDA", "EBAY", "CSCO", "IBM")
    target_company: str = "AAPL"
    window_size: int = 60
    train_fraction: float = 0.95
    lstm_units: int = 64
    dense_units: int = 32
    dropout_rate: float = 0.5
    epochs: int = 10
    batch_size: int = 32
    range_start: datetime = field(default_factory=lambda: datetime(2013, 1, 1))
    range_end: datetime = field(default_factory=lambda: datetime(2018, 1, 1))


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the multi-stock CSV and parse dates."""
    data = pd.read_csv(cfg.csv_path, delimiter=",", on_bad_lines="skip")
    print(data.shape)
    print(data.sample(7))

    data["date"] = pd.to_datetime(data["date"])
    data.info()
    return data


def plot_company_prices(data: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Plot open/close price for each configured company."""
    plt.figure(figsize=(15, 8))
    for index, company in enumerate(cfg.companies, 1):
        plt.subplot(3, 3, index)
        c = data[data["Name"] == company]
        plt.plot(c["date"], c["close"], c="r", label="close", marker="+")
        plt.plot(c["date"], c["open"], c="g", label="open", marker="^")
        plt.title(company)
        plt.legend()
        plt.tight_layout()
    plt.show()


def plot_company_volumes(data: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Plot trading volume for each configured company."""
    plt.figure(figsize=(15, 8))
    for index, company in enumerate(cfg.companies, 1):
        plt.subplot(3, 3, index)
        c = data[data["Name"] == company]
        plt.plot(c["date"], c["volume"], c="purple", marker="*")
        plt.title(f"{company} Volume")
        plt.tight_layout()
    plt.show()


def get_target_company_data(data: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Isolate the target company's rows and preview the configured date range."""
    company_df = data[data["Name"] == cfg.target_company]

    prediction_range = company_df.loc[
        (company_df["date"] > cfg.range_start) & (company_df["date"] < cfg.range_end)
    ]
    print(f"Rows in {cfg.range_start.date()}–{cfg.range_end.date()}: {len(prediction_range)}")

    plt.plot(company_df["date"], company_df["close"])
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title(f"{cfg.target_company} Stock Prices")
    plt.show()

    return company_df


def build_sequences(scaled_data: np.ndarray, window_size: int):
    """Turn a scaled 1D price series into (X, y) sliding-window sequences."""
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def prepare_train_test(company_df: pd.DataFrame, cfg: PipelineConfig):
    """Scale the close price series and build train/test sequence windows."""
    close_data = company_df.filter(["close"])
    dataset = close_data.values
    training_size = int(np.ceil(len(dataset) * cfg.train_fraction))
    print("Training rows:", training_size)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_size, :]
    x_train, y_train = build_sequences(train_data, cfg.window_size)

    test_data = scaled_data[training_size - cfg.window_size:, :]
    x_test, _ = build_sequences(test_data, cfg.window_size)
    y_test = dataset[training_size:, :]

    return x_train, y_train, x_test, y_test, scaler, training_size


def build_model(cfg: PipelineConfig, input_shape: tuple) -> keras.Model:
    """Build a two-layer LSTM regressor for the price-forecasting sequences."""
    model = keras.models.Sequential([
        keras.layers.LSTM(units=cfg.lstm_units, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(units=cfg.lstm_units),
        keras.layers.Dense(cfg.dense_units),
        keras.layers.Dropout(cfg.dropout_rate),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


def train_model(model: keras.Model, x_train, y_train, cfg: PipelineConfig):
    """Train the LSTM on the training sequences."""
    return model.fit(x_train, y_train, epochs=cfg.epochs, batch_size=cfg.batch_size)


def evaluate_and_forecast(model: keras.Model, x_test, y_test, scaler: MinMaxScaler):
    """Predict on the test sequences, invert scaling, and report MSE/RMSE."""
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    print("MSE", mse)
    print("RMSE", rmse)

    return predictions


def plot_forecast(company_df: pd.DataFrame, training_size: int, predictions: np.ndarray, cfg: PipelineConfig) -> None:
    """Plot train/actual/predicted close price."""
    train = company_df[:training_size].copy()
    test = company_df[training_size:].copy()
    test["Predictions"] = predictions

    plt.figure(figsize=(10, 8))
    plt.plot(train["date"], train["close"])
    plt.plot(test["date"], test[["close", "Predictions"]])
    plt.title(f"{cfg.target_company} Stock Close Price")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend(["Train", "Test", "Predictions"])
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, explore multiple companies, then train and evaluate an LSTM on the target company."""
    data = load_data(cfg)
    plot_company_prices(data, cfg)
    plot_company_volumes(data, cfg)

    company_df = get_target_company_data(data, cfg)

    x_train, y_train, x_test, y_test, scaler, training_size = prepare_train_test(company_df, cfg)

    model = build_model(cfg, input_shape=(x_train.shape[1], 1))
    train_model(model, x_train, y_train, cfg)

    predictions = evaluate_and_forecast(model, x_test, y_test, scaler)
    plot_forecast(company_df, training_size, predictions, cfg)


if __name__ == "__main__":
    run_pipeline()