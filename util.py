import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm


# ---------------------------
# DATA LOADER
# ---------------------------

def load_price_data(tickers, start="2015-01-01"):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    data = data["Close"]
    data = data.dropna(how="all")
    return data


# ---------------------------
# COINTEGRATION SCAN
# ---------------------------

def scan_cointegrated_pairs(price_df, p_threshold=0.05):
    results = []
    tickers = price_df.columns

    for t1, t2 in combinations(tickers, 2):
        s1 = price_df[t1].dropna()
        s2 = price_df[t2].dropna()
        df = pd.concat([s1, s2], axis=1).dropna()

        if len(df) < 250:
            continue

        score, pvalue, _ = coint(df.iloc[:, 0], df.iloc[:, 1])

        if pvalue < p_threshold:
            results.append((t1, t2, pvalue))

    return pd.DataFrame(results, columns=["Stock1", "Stock2", "p-value"]).sort_values("p-value")


# ---------------------------
# HEDGE RATIO (OLS)
# ---------------------------

def estimate_hedge_ratio(y, x):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    beta = model.params[1]
    return beta


# ---------------------------
# SPREAD + Z-SCORE
# ---------------------------

def compute_spread(y, x, beta):
    spread = y - beta * x
    return spread


def compute_zscore(series):
    return (series - series.mean()) / series.std()


# ---------------------------
# SIGNAL GENERATION
# ---------------------------

def generate_signals(zscore, threshold=2.0):
    signals = pd.DataFrame(index=zscore.index)
    signals["long"] = zscore < -threshold
    signals["short"] = zscore > threshold
    signals["position"] = signals["long"].astype(int) - signals["short"].astype(int)
    return signals


# ---------------------------
# BACKTEST ENGINE
# ---------------------------

def backtest(spread, signals):
    returns = spread.pct_change().fillna(0)
    strategy_returns = signals["position"].shift(1) * returns

    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1

    sharpe = (
        strategy_returns.mean() / strategy_returns.std()
    ) * np.sqrt(252)

    return cumulative, drawdown, sharpe
