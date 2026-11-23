# Multi-Asset Monte Carlo Portfolio Risk & VaR Engine

A modular Python project that estimates one-year portfolio loss risk using Monte Carlo, historical, and parametric methods. The default portfolio is equally weighted across AAPL, MSFT, GOOGL, JPM, and GLD, with a $100,000 starting value.

## Project structure

```text
multi_asset_var_engine/
├── requirements.txt
├── data_loader.py
├── mc_engine.py
├── risk_metrics.py
├── main.py
└── output/
    ├── simulated_paths.png
    └── terminal_distribution.png
```

## Methodology

### Historical inputs

`data_loader.py` downloads three years of daily Yahoo Finance prices with `auto_adjust=True`, so the `Close` series is adjusted for splits and distributions. Daily continuously compounded returns are

\[
r_{i,t}=\ln(P_{i,t}/P_{i,t-1}).
\]

The annualized estimates use 252 trading days:

\[
\hat\mu=252\,\bar r,\qquad \hat\Sigma=252\,\operatorname{Cov}(r).
\]

The portfolio uses normalized, constant weights. The simulation and historical calculation apply those weights to daily asset log returns, representing daily rebalancing.

### Correlated GBM Monte Carlo

For asset *i*, a GBM step is represented in log-return form:

\[
\Delta X_i=(\mu_i-\tfrac12\Sigma_{ii})\Delta t + L_i Z\sqrt{\Delta t},
\]

where \(Z\) is a vector of independent standard normal draws and \(LL^T=\Sigma\). The Cholesky factor \(L\), computed with `np.linalg.cholesky`, introduces the estimated cross-asset correlations. The portfolio value evolves as

\[
V_t=V_0\exp\left(\sum_{s=1}^{t}\sum_i w_i\Delta X_{i,s}\right).
\]

`mc_engine.py` vectorizes 10,000 paths over 252 trading days. The random seed is fixed for reproducibility.

### VaR and CVaR / Expected Shortfall

Define the one-year dollar loss as \(L=V_0-V_T\), so a positive number is a loss and a negative number is a gain. At confidence level \(c\):

- **VaR** is the \(c\)-quantile of the loss distribution.
- **CVaR / Expected Shortfall** is the average loss in the worst \((1-c)\) tail, i.e. losses at or above VaR.

The three methods are:

1. **Monte Carlo:** calculate terminal losses from simulated paths and estimate their empirical quantile and tail average.
2. **Historical simulation:** compound rolling 252-trading-day portfolio log returns from the observed history, then calculate the loss quantile and tail average. With three years of data, the number of overlapping annual windows is limited; this is an important sample-size limitation.
3. **Parametric variance-covariance:** approximate annual portfolio log return as normal with mean \(w^T\mu\) and variance \(w^T\Sigma w\). Convert the normal return quantile to a dollar-loss threshold. The CVaR uses the normal lower-tail expected return.

The parametric method assumes normally distributed annual portfolio log returns, making terminal value lognormal. Its dollar CVaR uses the exact truncated-lognormal conditional expectation for the lower return tail.

## Run

Python 3.10+ is recommended.

```bash
# From this project directory
python -m pip install -r requirements.txt
python main.py
```

The script prints 95% and 99% VaR/CVaR for all three methods and writes 300-DPI charts to `output/`:

- `simulated_paths.png`: 100 sample paths, the mean path, and 5th/95th percentile bands.
- `terminal_distribution.png`: terminal-value histogram, the 95% VaR terminal-value threshold, and the terminal-value equivalent of 95% CVaR.

## Configuration

Edit constants in `main.py` to change starting value, path count, horizon, or seed. Pass alternative tickers and weights to `load_market_data()` in `data_loader.py`.

## Important limitations

This is an educational risk model, not a trading system. It assumes stable estimated moments, normally distributed daily shocks, constant weights, and no transaction costs, taxes, liquidity constraints, jumps, or volatility clustering. Historical annual windows overlap and provide relatively few independent observations. Results are sensitive to the estimation window and model assumptions. Yahoo Finance availability and its data may change.
