"""
data_loader.py

Author: Yash Ganjare
Project: Turnover-Cost Gap Research

Handles downloading, preprocessing, and storing
multi-asset financial datasets.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Helper Functions
# ==========================================================


def _convert_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts dataframe index to UTC timezone.
    """

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


# ==========================================================
# Binance Downloader
# ==========================================================


def fetch_binance_btc(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    days: int = 365,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Downloads BTC price data from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair.

    timeframe : str
        Candle interval.

    days : int
        Number of days.

    limit : int
        Maximum candles per request.

    Returns
    -------
    pd.DataFrame
    """

    logger.info(f"Downloading {symbol} ({days} days)...")

    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
        }
    )

    since = exchange.parse8601(
        (
            pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    target_bars = days * 24 * 60

    all_ohlcv = []

    while len(all_ohlcv) < target_bars:

        try:

            batch = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            if len(batch) == 0:
                break

            all_ohlcv.extend(batch)

            since = batch[-1][0] + 60_000

            logger.info(
                f"Downloaded {len(all_ohlcv):,} candles..."
            )

            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:

            logger.warning(f"Retrying... {e}")

            time.sleep(5)

    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    df.index = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df = (
        df[["close"]]
        .rename(columns={"close": "BTC_Close"})
        .sort_index()
    )

    df = df[~df.index.duplicated(keep="first")]

    logger.info(f"BTC Bars: {len(df):,}")

    return df


# ==========================================================
# Yahoo Finance Downloader
# ==========================================================


def fetch_yahoo_asset(
    ticker: str,
    asset_name: str,
    days: int = 365,
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Downloads minute-level data from Yahoo Finance.

    Yahoo only allows a few days of minute data
    per request, so we download in chunks.
    """

    logger.info(f"Downloading {asset_name}...")

    chunks = []

    end_date = datetime.utcnow()

    chunk_days = 5

    for i in range(0, days, chunk_days):

        chunk_end = end_date - timedelta(days=i)

        chunk_start = chunk_end - timedelta(days=chunk_days)

        try:

            temp = yf.download(
                ticker,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if not temp.empty:

                chunks.append(temp)

                logger.info(
                    f"{asset_name}: downloaded chunk "
                    f"{chunk_start.date()} -> {chunk_end.date()}"
                )

        except Exception as e:

            logger.warning(e)

        time.sleep(0.5)

    if len(chunks) == 0:

        logger.warning(
            f"{asset_name} unavailable. "
            "Creating synthetic series."
        )

        idx = pd.date_range(
            end=datetime.utcnow(),
            periods=days * 24 * 60,
            freq="1min",
            tz="UTC",
        )

        returns = np.random.normal(
            0.00001,
            0.001,
            len(idx),
        )

        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                f"{asset_name}_Close": prices,
            },
            index=idx,
        )

    df = pd.concat(chunks)

    df = df.sort_index()

    df = df[~df.index.duplicated(keep="first")]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = _convert_to_utc(df)

    df = df[["Close"]].rename(
        columns={
            "Close": f"{asset_name}_Close"
        }
    )

    logger.info(f"{asset_name} Bars: {len(df):,}")

    return df

    # ==========================================================
# Dataset Utilities
# ==========================================================

def merge_market_data(
    btc_df: pd.DataFrame,
    assets: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge multiple market datasets using the BTC timestamp
    as the master timeline.

    Parameters
    ----------
    btc_df : pd.DataFrame
        BTC dataframe.

    assets : list[pd.DataFrame]
        Additional market dataframes.

    Returns
    -------
    pd.DataFrame
    """

    logger.info("Merging datasets...")

    master_df = btc_df.copy()

    for asset in assets:
        master_df = master_df.join(asset, how="left")

    master_df = (
        master_df
        .sort_index()
        .ffill()
        .bfill()
    )

    logger.info(
        f"Merged dataset contains {len(master_df):,} rows "
        f"and {len(master_df.columns)} columns."
    )

    return master_df


# ==========================================================
# Save / Load
# ==========================================================

def save_master_data(
    df: pd.DataFrame,
    filename: str = "master_data.parquet",
) -> Path:
    """
    Save processed dataset as parquet.
    """

    path = DATA_DIR / filename

    df.to_parquet(path)

    logger.info(f"Dataset saved to {path}")

    return path


def load_master_data(
    filename: str = "master_data.parquet",
) -> pd.DataFrame:
    """
    Load processed dataset.
    """

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    logger.info(f"Loading dataset: {path}")

    return pd.read_parquet(path)


# ==========================================================
# Complete Pipeline
# ==========================================================

def build_master_dataset(
    days: int = 365,
) -> pd.DataFrame:
    """
    Complete market-data pipeline.

    Downloads

    - BTC
    - S&P500
    - Gold
    - USD Index

    Merges everything and saves to disk.
    """

    logger.info("=" * 60)
    logger.info("Building Master Dataset")
    logger.info("=" * 60)

    btc = fetch_binance_btc(days=days)

    sp500 = fetch_yahoo_asset(
        ticker="^GSPC",
        asset_name="SP500",
        days=days,
    )

    gold = fetch_yahoo_asset(
        ticker="GC=F",
        asset_name="Gold",
        days=days,
    )

    usd = fetch_yahoo_asset(
        ticker="DX-Y.NYB",
        asset_name="USDX",
        days=days,
    )

    master = merge_market_data(
        btc,
        [
            sp500,
            gold,
            usd,
        ],
    )

    save_master_data(master)

    logger.info("Dataset creation complete.")

    return master


# ==========================================================
# Script Entry
# ==========================================================

def main():

    df = build_master_dataset(days=365)

    print("\nDataset Preview\n")

    print(df.head())

    print("\nDataset Info\n")

    print(df.info())

    print("\nMissing Values\n")

    print(df.isna().sum())

    print(f"\nTotal Rows : {len(df):,}")

    print(f"Columns    : {list(df.columns)}")


if __name__ == "__main__":
    main()