import numpy as np
import pandas as pd
import statsmodels.api as sm


def rolling_beta(y, x, window=60):
    betas = []
    for i in range(window, len(y)):
        y_window = y.iloc[i-window:i]
        x_window = x.iloc[i-window:i]
        x_const = sm.add_constant(x_window)
        model = sm.OLS(y_window, x_const).fit()
        betas.append(model.params[1])
    return pd.Series(betas, index=y.index[window:])


def calculate_half_life(spread):
    spread_lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()

    spread_lag = spread_lag.loc[delta.index]

    model = sm.OLS(delta, sm.add_constant(spread_lag)).fit()
    lambda_coef = model.params[1]

    half_life = -np.log(2) / lambda_coef
    return half_life


def calculate_hurst(ts):
    lags = range(2, 20)
    tau = [np.std(ts.diff(lag).dropna()) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def performance_metrics(strategy_returns):
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1

    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

    downside = strategy_returns[strategy_returns < 0]
    sortino = (strategy_returns.mean() / downside.std()) * np.sqrt(252)

    cagr = cumulative.iloc[-1] ** (252 / len(strategy_returns)) - 1

    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd
    }
