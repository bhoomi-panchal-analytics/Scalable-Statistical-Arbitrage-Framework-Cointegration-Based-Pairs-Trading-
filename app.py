import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations

from util import (
    estimate_hedge_ratio,
    compute_spread,
    compute_zscore,
    generate_signals,
    backtest,
    rolling_beta,
    calculate_half_life,
    calculate_hurst,
    performance_metrics,
)

st.set_page_config(layout="wide")
st.title("Scalable Statistical Arbitrage – Cointegration Framework")

# ---------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------

st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input(
    "Enter tickers (comma separated)",
    "JPM,BAC,C,GS,MS"
)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2015-01-01")
)

p_threshold = st.sidebar.slider("Cointegration p-value", 0.01, 0.1, 0.05)
z_threshold = st.sidebar.slider("Z-score Threshold", 1.0, 3.0, 2.0)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

@st.cache_data
def load_data(tickers, start):
    data = yf.download(
        tickers,
        start=str(start),
        auto_adjust=True,
        progress=False
    )["Close"]

    return data.dropna(how="all")

data = load_data(tickers, start_date)

if data.shape[0] < 250:
    st.error("Minimum 250 observations required.")
    st.stop()

# ---------------------------------------------------
# COINTEGRATION SCANNER
# ---------------------------------------------------

from statsmodels.tsa.stattools import coint

pairs = []
for t1, t2 in combinations(data.columns, 2):
    s1 = data[t1]
    s2 = data[t2]
    df = pd.concat([s1, s2], axis=1).dropna()

    if len(df) < 250:
        continue

    score, pvalue, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    pairs.append((t1, t2, pvalue))

pairs_df = pd.DataFrame(pairs, columns=["Stock1", "Stock2", "p-value"])
pairs_df = pairs_df.sort_values("p-value")

st.subheader("Cointegration Scan")
st.dataframe(pairs_df)

valid_pairs = pairs_df[pairs_df["p-value"] < p_threshold]

if valid_pairs.empty:
    st.warning("No cointegrated pairs found under selected threshold.")
    st.stop()

selected_pair = st.selectbox(
    "Select Pair",
    valid_pairs.apply(lambda r: f"{r['Stock1']} - {r['Stock2']}", axis=1)
)

stock1, stock2 = selected_pair.split(" - ")

# ---------------------------------------------------
# PAIR ANALYSIS
# ---------------------------------------------------

y = data[stock1]
x = data[stock2]

beta = estimate_hedge_ratio(y, x)
spread = compute_spread(y, x, beta)
zscore = compute_zscore(spread)

signals = generate_signals(zscore, z_threshold)

spread_returns = spread.pct_change().fillna(0)
strategy_returns = signals["position"].shift(1) * spread_returns
strategy_returns = strategy_returns.fillna(0)

cum_returns, drawdown, sharpe = backtest(spread, signals)

metrics = performance_metrics(strategy_returns)
half_life = calculate_half_life(spread)
hurst = calculate_hurst(spread)

# ---------------------------------------------------
# METRICS DISPLAY
# ---------------------------------------------------

st.subheader("Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("CAGR", round(metrics["CAGR"], 4))
col2.metric("Sharpe", round(metrics["Sharpe"], 3))
col3.metric("Sortino", round(metrics["Sortino"], 3))
col4.metric("Max Drawdown", round(metrics["Max Drawdown"], 3))

col5, col6 = st.columns(2)
col5.metric("Half-Life (days)", round(half_life, 2))
col6.metric("Hurst Exponent", round(hurst, 3))

# ---------------------------------------------------
# GRAPHS
# ---------------------------------------------------

# 1 Price
st.plotly_chart(
    px.line(data[[stock1, stock2]], title="Price Series"),
    use_container_width=True
)

# 2 Spread
st.plotly_chart(
    px.line(spread, title="Spread"),
    use_container_width=True
)

# 3 Z-score
fig_z = go.Figure()
fig_z.add_trace(go.Scatter(x=zscore.index, y=zscore))
fig_z.add_hline(y=z_threshold)
fig_z.add_hline(y=-z_threshold)
fig_z.update_layout(title="Z-score with Thresholds")
st.plotly_chart(fig_z, use_container_width=True)

# 4 Rolling Correlation
rolling_corr = y.rolling(60).corr(x)
st.plotly_chart(
    px.line(rolling_corr, title="Rolling 60-Day Correlation"),
    use_container_width=True
)

# 5 Rolling Hedge Ratio
beta_roll = rolling_beta(y, x)
st.plotly_chart(
    px.line(beta_roll, title="Rolling Hedge Ratio (60-day)"),
    use_container_width=True
)

# 6 Cumulative Returns
st.plotly_chart(
    px.line(cum_returns, title="Cumulative Strategy Returns"),
    use_container_width=True
)

# 7 Drawdown
st.plotly_chart(
    px.area(drawdown, title="Drawdown"),
    use_container_width=True
)

# 8 Rolling Volatility
rolling_vol = spread_returns.rolling(60).std() * np.sqrt(252)
st.plotly_chart(
    px.line(rolling_vol, title="Rolling Annualized Volatility of Spread"),
    use_container_width=True
)

# 9 Rolling Sharpe
rolling_sharpe = (
    strategy_returns.rolling(60).mean() /
    strategy_returns.rolling(60).std()
) * np.sqrt(252)

st.plotly_chart(
    px.line(rolling_sharpe, title="Rolling Sharpe (60-day)"),
    use_container_width=True
)

# 10 Spread Distribution
st.plotly_chart(
    px.histogram(spread, nbins=50, title="Spread Distribution"),
    use_container_width=True
)

# 11 Trade Markers
fig_trade = go.Figure()
fig_trade.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))

long_entries = spread[signals["long"]]
short_entries = spread[signals["short"]]

fig_trade.add_trace(go.Scatter(
    x=long_entries.index,
    y=long_entries,
    mode="markers",
    marker=dict(symbol="triangle-up", size=8),
    name="Long"
))

fig_trade.add_trace(go.Scatter(
    x=short_entries.index,
    y=short_entries,
    mode="markers",
    marker=dict(symbol="triangle-down", size=8),
    name="Short"
))

fig_trade.update_layout(title="Trade Signals on Spread")
st.plotly_chart(fig_trade, use_container_width=True)
