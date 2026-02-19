import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------
# HEDGE RATIO (OLS)
# ---------------------------------------------------

def estimate_hedge_ratio(y, x):
    """
    Estimate static hedge ratio using OLS regression.
    """
    df = pd.concat([y, x], axis=1).dropna()
    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1]

    x_const = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, x_const).fit()

    beta = model.params[1]
    return beta


# ---------------------------------------------------
# SPREAD
# ---------------------------------------------------

def compute_spread(y, x, beta):
    df = pd.concat([y, x], axis=1).dropna()
    spread = df.iloc[:, 0] - beta * df.iloc[:, 1]
    return spread


# ---------------------------------------------------
# Z-SCORE
# ---------------------------------------------------

def compute_zscore(series):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


# ---------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------

def generate_signals(zscore, threshold=2.0):
    signals = pd.DataFrame(index=zscore.index)
    signals["long"] = zscore < -threshold
    signals["short"] = zscore > threshold
    signals["position"] = signals["long"].astype(int) - signals["short"].astype(int)
    return signals


# ---------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------

def backtest(spread, signals):
    spread_returns = spread.pct_change().fillna(0)
    strategy_returns = signals["position"].shift(1) * spread_returns
    strategy_returns = strategy_returns.fillna(0)

    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1

    sharpe = 0
    if strategy_returns.std() != 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

    return cumulative, drawdown, sharpe


# ---------------------------------------------------
# ROLLING BETA
# ---------------------------------------------------

def rolling_beta(y, x, window=60):
    betas = []
    index_vals = []

    df = pd.concat([y, x], axis=1).dropna()
    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1]

    for i in range(window, len(df)):
        y_window = y_clean.iloc[i-window:i]
        x_window = x_clean.iloc[i-window:i]

        x_const = sm.add_constant(x_window)
        model = sm.OLS(y_window, x_const).fit()

        betas.append(model.params[1])
        index_vals.append(df.index[i])

    return pd.Series(betas, index=index_vals)


# ---------------------------------------------------
# HALF-LIFE OF MEAN REVERSION
# ---------------------------------------------------

def calculate_half_life(spread):
    spread = spread.dropna()
    spread_lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()

    spread_lag = spread_lag.loc[delta.index]

    if len(spread_lag) < 20:
        return np.nan

    model = sm.OLS(delta, sm.add_constant(spread_lag)).fit()
    lambda_coef = model.params[1]

    if lambda_coef >= 0:
        return np.nan

    half_life = -np.log(2) / lambda_coef
    return half_life


# ---------------------------------------------------
# HURST EXPONENT
# ---------------------------------------------------

def calculate_hurst(ts):
    ts = ts.dropna()

    if len(ts) < 100:
        return np.nan

    lags = range(2, 20)
    tau = [np.std(ts.diff(lag).dropna()) for lag in lags]

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2.0

    return hurst


# ---------------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------------

def performance_metrics(strategy_returns):
    strategy_returns = strategy_returns.fillna(0)

    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1

    sharpe = 0
    sortino = 0

    if strategy_returns.std() != 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

    downside = strategy_returns[strategy_returns < 0]
    if downside.std() != 0:
        sortino = (strategy_returns.mean() / downside.std()) * np.sqrt(252)

    cagr = 0
    if len(strategy_returns) > 0:
        cagr = cumulative.iloc[-1] ** (252 / len(strategy_returns)) - 1

    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd
    }
