"""Run the multi-asset Monte Carlo portfolio risk pipeline."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import DEFAULT_TICKERS, load_market_data
from mc_engine import simulate_portfolio_paths
from risk_metrics import (comparison_table, historical_metrics,
                          monte_carlo_metrics, parametric_metrics)

INITIAL_VALUE = 100_000.0
N_SIMS = 10_000
HORIZON_DAYS = 252
CONFIDENCE_LEVELS = (0.95, 0.99)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def make_path_chart(paths: np.ndarray, output_path: Path) -> None:
    days = np.arange(paths.shape[0])
    mean_path = paths.mean(axis=1)
    p05, p95 = np.percentile(paths, [5, 95], axis=1)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(days, paths[:, :100], color="steelblue", alpha=0.16, linewidth=0.8)
    ax.fill_between(days, p05, p95, color="orange", alpha=0.22, label="5th–95th percentile band")
    ax.plot(days, mean_path, color="darkred", linewidth=2.2, label="Mean path")
    ax.set(title="Monte Carlo Portfolio Value Paths (100 of 10,000)",
           xlabel="Trading days", ylabel="Portfolio value ($)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_distribution_chart(paths: np.ndarray, mc_metrics: dict, output_path: Path) -> None:
    terminal = paths[-1]
    losses = INITIAL_VALUE - terminal
    var95, cvar95 = mc_metrics[0.95]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(terminal, bins=70, color="slateblue", alpha=0.78, edgecolor="white")
    # VaR is a loss threshold; translate it to the equivalent terminal value.
    var_value = INITIAL_VALUE - var95
    cvar_value = INITIAL_VALUE - cvar95
    ax.axvline(var_value, color="darkorange", linewidth=2.2,
               label=f"95% VaR threshold: ${var_value:,.0f}")
    ax.axvline(cvar_value, color="crimson", linewidth=2.2, linestyle="--",
               label=f"95% CVaR (mean tail loss): ${cvar_value:,.0f}")
    ax.axvline(INITIAL_VALUE, color="black", linewidth=1.2, linestyle=":", label="Initial value")
    ax.set(title="Simulated 1-Year Terminal Portfolio Value Distribution",
           xlabel="Terminal portfolio value ($)", ylabel="Frequency")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading 3 years of adjusted daily prices for:", ", ".join(DEFAULT_TICKERS))
    market = load_market_data()
    print(f"Loaded {len(market.prices):,} daily price observations; "
          f"{len(market.log_returns):,} daily returns.")
    paths = simulate_portfolio_paths(
        market.annual_mean, market.annual_cov, market.weights,
        initial_value=INITIAL_VALUE, n_sims=N_SIMS,
        horizon_days=HORIZON_DAYS, seed=42,
    )
    mc = monte_carlo_metrics(paths, INITIAL_VALUE, CONFIDENCE_LEVELS)
    hist = historical_metrics(market.log_returns, market.weights, INITIAL_VALUE,
                              HORIZON_DAYS, CONFIDENCE_LEVELS)
    param = parametric_metrics(market.annual_mean, market.annual_cov,
                               market.weights, INITIAL_VALUE, 1.0, CONFIDENCE_LEVELS)
    table = comparison_table(mc, hist, param)
    print("\nONE-YEAR PORTFOLIO RISK (USD; loss relative to $100,000 initial value)")
    print(table.to_string(index=False, formatters={"VaR ($)": "${:,.2f}".format,
                                                   "CVaR ($)": "${:,.2f}".format}))

    path_chart = OUTPUT_DIR / "simulated_paths.png"
    dist_chart = OUTPUT_DIR / "terminal_distribution.png"
    make_path_chart(paths, path_chart)
    make_distribution_chart(paths, mc, dist_chart)
    print("\nCharts saved:")
    print(f" - {path_chart}")
    print(f" - {dist_chart}")
    print("\nNote: VaR/CVaR are model estimates, not guarantees or investment advice.")


if __name__ == "__main__":
    main()
