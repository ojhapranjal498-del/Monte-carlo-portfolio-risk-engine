"""Vectorized correlated Geometric Brownian Motion Monte Carlo engine."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_portfolio_paths(
    annual_mean: pd.Series | np.ndarray,
    annual_cov: pd.DataFrame | np.ndarray,
    weights: pd.Series | np.ndarray,
    initial_value: float = 100_000.0,
    n_sims: int = 10_000,
    horizon_days: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """Simulate daily-rebalanced portfolio values under correlated GBM.

    Asset log-return increments are multivariate normal. Cholesky factorization
    imposes the estimated cross-asset covariance. Constant portfolio weights
    are applied to each simulated daily asset log return (daily rebalancing).

    Returns an array shaped ``(horizon_days + 1, n_sims)`` including day zero.
    """
    mu = np.asarray(annual_mean, dtype=float).reshape(-1)
    cov = np.asarray(annual_cov, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if cov.shape != (mu.size, mu.size) or w.size != mu.size:
        raise ValueError("Mean, covariance, and weight dimensions do not agree.")
    if initial_value <= 0 or n_sims < 1 or horizon_days < 1:
        raise ValueError("Initial value, simulation count, and horizon must be positive.")
    w = w / w.sum()

    # Symmetrize and add a tiny diagonal jitter to tolerate floating-point noise.
    cov = (cov + cov.T) / 2
    chol = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * 1e-12)
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    z = rng.standard_normal((horizon_days, n_sims, mu.size))
    correlated = z @ chol.T
    log_increments = (mu - 0.5 * np.diag(cov))[None, None, :] * dt
    log_increments = log_increments + correlated * np.sqrt(dt)
    portfolio_log_returns = np.einsum("tni,i->tn", log_increments, w)

    paths = np.empty((horizon_days + 1, n_sims), dtype=float)
    paths[0] = initial_value
    paths[1:] = initial_value * np.exp(np.cumsum(portfolio_log_returns, axis=0))
    return paths
