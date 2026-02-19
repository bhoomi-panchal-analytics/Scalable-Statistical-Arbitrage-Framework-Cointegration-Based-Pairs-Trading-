import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import coint
import yfinance as yf

st.set_page_config(layout="wide")
st.title("Statistical Arbitrage – Cointegration Framework")

# Sidebar
st.sidebar.header("Configuration")

ticker1 = st.sidebar.text_input("Stock 1", "JPM")
ticker2 = st.sidebar.text_input("Stock 2", "BAC")
start_date = st.sidebar.date_input("Start Date")
z_threshold = st.sidebar.slider("Z-score Threshold", 1.0, 3.0, 2.0)
p_threshold = st.sidebar.slider("Cointegration p-value", 0.01, 0.1, 0.05)

# Data Load
@st.cache_data
def load_data(t1, t2):
    data = yf.download([t1, t2], start="2018-01-01", auto_adjust=True)
    data = data["Close"].dropna()
    return data

data = load_data(ticker1, ticker2)

# Cointegration
score, pvalue, _ = coint(data[ticker1], data[ticker2])
st.write(f"Cointegration p-value: {pvalue:.5f}")

if pvalue < p_threshold:

    # OLS Hedge Ratio
    beta = np.polyfit(data[ticker2], data[ticker1], 1)[0]
    spread = data[ticker1] - beta * data[ticker2]

    zscore = (spread - spread.mean()) / spread.std()

    # GRAPH 1 – Price
    fig1 = px.line(data, title="Price Series")
    st.plotly_chart(fig1, use_container_width=True)

    # GRAPH 2 – Spread
    fig2 = px.line(spread, title="Spread")
    st.plotly_chart(fig2, use_container_width=True)

    # GRAPH 3 – Z-score
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
    fig3.add_hline(y=z_threshold)
    fig3.add_hline(y=-z_threshold)
    fig3.update_layout(title="Z-score with Thresholds")
    st.plotly_chart(fig3, use_container_width=True)

    # Signals
    signals = pd.DataFrame(index=zscore.index)
    signals["long"] = zscore < -z_threshold
    signals["short"] = zscore > z_threshold

    # Backtest
    position = signals["long"].astype(int) - signals["short"].astype(int)
    returns = spread.pct_change().fillna(0)
    strategy_returns = position.shift(1) * returns

    cum_returns = (1 + strategy_returns).cumprod()

    # GRAPH 4 – Cumulative Returns
    fig4 = px.line(cum_returns, title="Cumulative Strategy Returns")
    st.plotly_chart(fig4, use_container_width=True)

    # GRAPH 5 – Drawdown
    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1
    fig5 = px.area(drawdown, title="Drawdown")
    st.plotly_chart(fig5, use_container_width=True)

    # Add remaining graphs similarly:
    # Rolling Sharpe
    rolling_sharpe = strategy_returns.rolling(60).mean() / strategy_returns.rolling(60).std()
    fig6 = px.line(rolling_sharpe, title="Rolling Sharpe Ratio")
    st.plotly_chart(fig6, use_container_width=True)

else:
    st.warning("Pair not cointegrated under selected threshold.")
