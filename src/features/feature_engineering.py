"""
feature_engineering.py

Creates engineered features for PPO trading.

Pipeline

master_data.parquet
        ↓
Feature Engineering
        ↓
master_features.parquet
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = Path("data/processed")

INPUT_FILE = DATA_DIR / "master_data.parquet"
FEATURE_FILE = DATA_DIR / "master_features.parquet"
SCALER_FILE = DATA_DIR / "feature_scaler.pkl"
TENSOR_FILE = DATA_DIR / "training_tensors.npz"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Load Dataset
# ==========================================================


def load_master_data() -> pd.DataFrame:

    logger.info("Loading master dataset...")

    df = pd.read_parquet(INPUT_FILE)

    logger.info(f"Rows : {len(df):,}")

    logger.info(f"Columns : {list(df.columns)}")

    return df


# ==========================================================
# Return Features
# ==========================================================


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Creating return features...")

    assets = [
        "BTC_Close",
        "SP500_Close",
        "Gold_Close",
        "USDX_Close",
    ]

    for asset in assets:

        name = asset.replace("_Close", "_Return")

        df[name] = np.log(df[asset] / df[asset].shift(1))

    return df


# ==========================================================
# Moving Average Features
# ==========================================================


def add_sma_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Creating SMA features...")

    for window in [10, 20, 50]:

        df[f"SMA_{window}"] = (
            df["BTC_Close"]
            / df["BTC_Close"].rolling(window).mean()
        ) - 1.0

    return df


# ==========================================================
# Volatility Features
# ==========================================================


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Creating volatility features...")

    btc_return = df["BTC_Return"]

    df["Volatility_20"] = btc_return.rolling(20).std()

    df["Volatility_50"] = btc_return.rolling(50).std()

    return df


# ==========================================================
# RSI
# ==========================================================


def compute_rsi(
    series: pd.Series,
    window: int = 14,
) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()

    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-12)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ==========================================================
# MACD
# ==========================================================


def compute_macd(
    series: pd.Series,
):

    ema12 = series.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = series.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    hist = macd - signal

    return macd, signal, hist


# ==========================================================
# Bollinger Bands
# ==========================================================


def add_bollinger_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Creating Bollinger Bands...")

    ma20 = df["BTC_Close"].rolling(20).mean()

    std20 = df["BTC_Close"].rolling(20).std()

    df["BB_Upper"] = ma20 + (2 * std20)

    df["BB_Lower"] = ma20 - (2 * std20)

    df["BB_Width"] = (
        df["BB_Upper"] - df["BB_Lower"]
    ) / ma20

    return df


# ==========================================================
# Technical Indicators
# ==========================================================


def add_technical_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Creating technical indicators...")

    df["RSI"] = compute_rsi(df["BTC_Close"])

    (
        df["MACD"],
        df["MACD_SIGNAL"],
        df["MACD_HIST"],
    ) = compute_macd(df["BTC_Close"])

    df = add_bollinger_features(df)

    return df

    # ==========================================================
# Lag Features
# ==========================================================

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Creating lag features...")

    lag_assets = [
        "BTC_Return",
        "SP500_Return",
        "Gold_Return",
        "USDX_Return",
    ]

    for asset in lag_assets:

        for lag in [1, 3, 5]:

            df[f"{asset}_Lag{lag}"] = df[asset].shift(lag)

    return df


# ==========================================================
# Cross-Market Correlation Features
# ==========================================================

def add_cross_market_features(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:

    logger.info("Creating rolling correlations...")

    df["BTC_SP500_Corr"] = (
        df["BTC_Return"]
        .rolling(window)
        .corr(df["SP500_Return"])
    )

    df["BTC_Gold_Corr"] = (
        df["BTC_Return"]
        .rolling(window)
        .corr(df["Gold_Return"])
    )

    df["BTC_USDX_Corr"] = (
        df["BTC_Return"]
        .rolling(window)
        .corr(df["USDX_Return"])
    )

    return df


# ==========================================================
# Market Regime
# ==========================================================

def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Detecting market regimes...")

    q1 = df["Volatility_50"].quantile(0.33)
    q2 = df["Volatility_50"].quantile(0.66)

    conditions = [
        df["Volatility_50"] <= q1,
        (df["Volatility_50"] > q1) &
        (df["Volatility_50"] <= q2),
        df["Volatility_50"] > q2,
    ]

    choices = [
        0,
        1,
        2,
    ]

    df["Market_Regime"] = np.select(
        conditions,
        choices,
        default=1,
    )

    return df


# ==========================================================
# Clean Dataset
# ==========================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Cleaning dataset...")

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna()

    logger.info(f"Remaining rows : {len(df):,}")

    return df


# ==========================================================
# Train / Test Split
# ==========================================================

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
):

    logger.info("Creating train/test split...")

    split = int(len(df) * train_ratio)

    train_df = df.iloc[:split].copy()

    test_df = df.iloc[split:].copy()

    logger.info(
        f"Train : {len(train_df):,}"
    )

    logger.info(
        f"Test  : {len(test_df):,}"
    )

    return train_df, test_df


# ==========================================================
# Feature Scaling
# ==========================================================

def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "BTC_Return",
):

    logger.info("Scaling features...")

    feature_columns = [
        c for c in train_df.columns
        if c != target_column
    ]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_df[feature_columns]
    )

    X_test = scaler.transform(
        test_df[feature_columns]
    )

    y_train = train_df[target_column].values

    y_test = test_df[target_column].values

    with open(
        SCALER_FILE,
        "wb",
    ) as f:

        pickle.dump(
            scaler,
            f,
        )

    logger.info("Scaler saved.")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
    )


# ==========================================================
# Feature Engineering Pipeline
# ==========================================================

def engineer_features() -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)

    df = load_master_data()

    df = add_return_features(df)

    df = add_sma_features(df)

    df = add_volatility_features(df)

    df = add_technical_indicators(df)

    df = add_lag_features(df)

    df = add_cross_market_features(df)

    df = add_market_regime(df)

    df = clean_dataset(df)

    df.to_parquet(FEATURE_FILE)

    logger.info(f"Saved features -> {FEATURE_FILE}")

    return df

    # ==========================================================
# Sequence Generation
# ==========================================================

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 12,
):

    logger.info(
        f"Creating {lookback}-step sequences..."
    )

    X_seq = []
    y_seq = []

    for i in range(lookback, len(X)):

        X_seq.append(
            X[i - lookback:i]
        )

        y_seq.append(
            y[i]
        )

    X_seq = np.asarray(
        X_seq,
        dtype=np.float32,
    )

    y_seq = np.asarray(
        y_seq,
        dtype=np.float32,
    )

    logger.info(
        f"Created {len(X_seq):,} sequences."
    )

    return X_seq, y_seq


# ==========================================================
# Save Training Tensors
# ==========================================================

def save_training_tensors(
    X_train,
    y_train,
    X_test,
    y_test,
):

    logger.info("Saving tensors...")

    np.savez_compressed(
        TENSOR_FILE,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    logger.info(
        f"Tensors saved -> {TENSOR_FILE}"
    )


# ==========================================================
# Main Pipeline
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("Starting Feature Engineering")
    logger.info("=" * 60)

    # --------------------------------------
    # Engineer Features
    # --------------------------------------

    df = engineer_features()

    # --------------------------------------
    # Split Dataset
    # --------------------------------------

    train_df, test_df = split_dataset(df)

    # --------------------------------------
    # Scale Features
    # --------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
    ) = scale_features(
        train_df,
        test_df,
    )

    logger.info(
        f"Number of Features : {len(feature_columns)}"
    )

    # --------------------------------------
    # Convert to Sequential Tensors
    # --------------------------------------

    X_train, y_train = create_sequences(
        X_train,
        y_train,
    )

    X_test, y_test = create_sequences(
        X_test,
        y_test,
    )

    # --------------------------------------
    # Save
    # --------------------------------------

    save_training_tensors(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    logger.info("=" * 60)
    logger.info("Feature Engineering Completed")
    logger.info("=" * 60)

    logger.info(
        f"Train Shape : {X_train.shape}"
    )

    logger.info(
        f"Test Shape  : {X_test.shape}"
    )

    logger.info(
        f"Feature Count : {X_train.shape[2]}"
    )

    logger.info(
        f"Lookback : {X_train.shape[1]}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()