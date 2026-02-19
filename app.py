import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from util import (
    load_price_data,
    scan_cointegrated_pairs,
    estimate_hedge_ratio,
    compute_spread,
    compute_zscore,
    generate_signals,
    backtest,
)

st.set_page_config(layout="wide")
st.title("Statistical Arbitrage – Cointegration Framework")

# Sidebar
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input(
    "Enter tickers (comma separated)", "JPM,BAC,C,GS,MS"
)

start_date = st.sidebar.date_input(
    "Start Date", pd.to_datetime("2015-01-01")
)

p_threshold = st.sidebar.slider("Cointegration p-value", 0.01, 0.1, 0.05)
z_threshold = st.sidebar.slider("Z-score Threshold", 1.0, 3.0, 2.0)

tickers = [t.strip().upper() for t in tickers_input.split(",")]

@st.cache_data
def load_data_cached(tickers, start_date):
    return load_price_data(tickers, str(start_date))

data = load_data_cached(tickers, start_date)

if data.shape[0] < 250:
    st.error("Minimum 250 observations required.")
    st.stop()

# ---------------------------
# PAIR SCANNER
# ---------------------------

st.subheader("Cointegrated Pairs")
pairs_df = scan_cointegrated_pairs(data, p_threshold)

if pairs_df.empty:
    st.warning("No cointegrated pairs found.")
    st.stop()

st.dataframe(pairs_df)

selected_pair = st.selectbox(
    "Select Pair",
    pairs_df.apply(lambda row: f"{row['Stock1']} - {row['Stock2']}", axis=1)
)

stock1, stock2 = selected_pair.split(" - ")

y = data[stock1]
x = data[stock2]

beta = estimate_hedge_ratio(y, x)
spread = compute_spread(y, x, beta)
zscore = compute_zscore(spread)
signals = generate_signals(zscore, z_threshold)
cum_returns, drawdown, sharpe = backtest(spread, signals)

st.metric("Sharpe Ratio", round(sharpe, 3))

# ---------------------------
# 10 GRAPHS
# ---------------------------

# 1 Price
st.plotly_chart(px.line(data[[stock1, stock2]], title="Price Series"), use_container_width=True)

# 2 Spread
st.plotly_chart(px.line(spread, title="Spread"), use_container_width=True)

# 3 Z-score
fig_z = go.Figure()
fig_z.add_trace(go.Scatter(x=zscore.index, y=zscore))
fig_z.add_hline(y=z_threshold)
fig_z.add_hline(y=-z_threshold)
fig_z.update_layout(title="Z-score")
st.plotly_chart(fig_z, use_container_width=True)

# 4 Rolling Correlation
rolling_corr = y.rolling(60).corr(x)
st.plotly_chart(px.line(rolling_corr, title="Rolling 60-Day Correlation"), use_container_width=True)

# 5 Cumulative Returns
st.plotly_chart(px.line(cum_returns, title="Cumulative Returns"), use_container_width=True)

# 6 Drawdown
st.plotly_chart(px.area(drawdown, title="Drawdown"), use_container_width=True)

# 7 Histogram Spread
st.plotly_chart(px.histogram(spread, nbins=50, title="Spread Distribution"), use_container_width=True)

# 8 Rolling Sharpe
rolling_sharpe = signals["position"].shift(1) * spread.pct_change()
rolling_sharpe = rolling_sharpe.rolling(60).mean() / rolling_sharpe.rolling(60).std()
st.plotly_chart(px.line(rolling_sharpe, title="Rolling Sharpe"), use_container_width=True)

# 9 Position Exposure
st.plotly_chart(px.line(signals["position"], title="Position Exposure"), use_container_width=True)

# 10 Coint Heatmap
pivot = pairs_df.pivot(index="Stock1", columns="Stock2", values="p-value")
st.plotly_chart(px.imshow(pivot, title="Cointegration P-Value Heatmap"), use_container_width=True)
