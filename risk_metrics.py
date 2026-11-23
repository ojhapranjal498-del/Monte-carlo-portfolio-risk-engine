"""Comparable one-year loss VaR and CVaR calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _tail_metrics(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


def monte_carlo_metrics(paths: np.ndarray, initial_value: float = 100_000.0,
                        confidence_levels=(0.95, 0.99)) -> dict:
    """Calculate VaR/CVaR from simulated terminal portfolio losses."""
    losses = initial_value - np.asarray(paths)[-1]
    return {c: _tail_metrics(losses, c) for c in confidence_levels}


def historical_metrics(log_returns: pd.DataFrame, weights: pd.Series,
                       initial_value: float = 100_000.0, horizon_days: int = 252,
                       confidence_levels=(0.95, 0.99)) -> dict:
    """Historical simulation using rolling, compounded one-year log returns."""
    w = weights.reindex(log_returns.columns).to_numpy(dtype=float)
    daily_portfolio_log_returns = log_returns.to_numpy() @ w
    if daily_portfolio_log_returns.size < horizon_days:
        raise ValueError("Insufficient daily observations for the requested historical horizon.")
    rolling_log_returns = np.convolve(
        daily_portfolio_log_returns, np.ones(horizon_days), mode="valid"
    )
    terminal_values = initial_value * np.exp(rolling_log_returns)
    losses = initial_value - terminal_values
    return {c: _tail_metrics(losses, c) for c in confidence_levels}


def parametric_metrics(annual_mean: pd.Series, annual_cov: pd.DataFrame,
                       weights: pd.Series, initial_value: float = 100_000.0,
                       horizon_years: float = 1.0,
                       confidence_levels=(0.95, 0.99)) -> dict:
    """Normal variance-covariance VaR/CVaR on annual portfolio log returns."""
    w = weights.reindex(annual_mean.index).to_numpy(dtype=float)
    w = w / w.sum()
    mu = float(w @ annual_mean.to_numpy(dtype=float)) * horizon_years
    variance = float(w @ annual_cov.to_numpy(dtype=float) @ w) * horizon_years
    sigma = np.sqrt(max(variance, 0.0))
    result = {}
    for confidence in confidence_levels:
        z = norm.ppf(confidence)
        # Loss = V0 * (1 - exp(R)); its confidence quantile maps to the
        # lower-tail (1-confidence) quantile of normally distributed R.
        return_cutoff = mu - z * sigma
        var = initial_value * (1.0 - np.exp(return_cutoff))
        # Exact conditional expectation of exp(R) in the lower normal tail:
        # E[exp(R) | R <= mu-z*sigma] = exp(mu+sigma^2/2) * Phi(-z-sigma)/Phi(-z).
        expected_terminal_ratio = (
            np.exp(mu + 0.5 * sigma**2)
            * norm.cdf(-z - sigma) / (1.0 - confidence)
        )
        cvar = initial_value * (1.0 - expected_terminal_ratio)
        result[confidence] = (float(var), float(cvar))
    return result


def comparison_table(mc: dict, hist: dict, param: dict) -> pd.DataFrame:
    rows = []
    for method, metrics in (("Monte Carlo", mc), ("Historical", hist), ("Parametric Normal", param)):
        for confidence, (var, cvar) in metrics.items():
            rows.append({"Method": method, "Confidence": f"{confidence:.0%}",
                         "VaR ($)": var, "CVaR ($)": cvar})
    return pd.DataFrame(rows)
