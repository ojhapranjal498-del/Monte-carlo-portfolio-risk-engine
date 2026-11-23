"""Market-data retrieval and annualized return/covariance estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GLD"]


@dataclass
class MarketData:
    prices: pd.DataFrame
    log_returns: pd.DataFrame
    annual_mean: pd.Series
    annual_cov: pd.DataFrame
    weights: pd.Series


def load_market_data(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    period: str = "3y",
    weights: Sequence[float] | None = None,
) -> MarketData:
    """Download adjusted daily closes and estimate annualized log-return moments.

    ``auto_adjust=True`` makes Yahoo Finance's ``Close`` column adjusted for
    splits and distributions. Weights are normalized and held constant in the
    daily-rebalanced portfolio used by the simulation and historical method.
    """
    tickers = list(tickers)
    if not tickers:
        raise ValueError("At least one ticker is required.")

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Check connectivity/tickers and retry.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Adjusted Close prices were not present in Yahoo Finance response.")
        prices = raw["Close"].copy()
    else:
        # yfinance returns a flat frame for a single ticker in some versions.
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.reindex(columns=tickers).dropna(how="all").ffill().dropna()
    prices = prices.dropna(axis=1, how="any")
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        raise RuntimeError(f"No complete price history returned for: {', '.join(missing)}")
    if len(prices) < 253:
        raise RuntimeError(f"Need at least 253 daily prices; received {len(prices)}.")

    returns = np.log(prices / prices.shift(1)).dropna()
    if weights is None:
        weight_values = np.full(len(tickers), 1.0 / len(tickers))
    else:
        weight_values = np.asarray(weights, dtype=float)
        if weight_values.shape != (len(tickers),) or np.any(weight_values < 0):
            raise ValueError("Weights must be non-negative and match the ticker count.")
        if weight_values.sum() <= 0:
            raise ValueError("At least one portfolio weight must be positive.")
        weight_values = weight_values / weight_values.sum()

    weight_series = pd.Series(weight_values, index=tickers, name="weight")
    annual_mean = returns.mean() * 252
    annual_cov = returns.cov() * 252
    return MarketData(prices, returns, annual_mean, annual_cov, weight_series)
