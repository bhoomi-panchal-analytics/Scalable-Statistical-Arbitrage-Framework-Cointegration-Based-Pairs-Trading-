import streamlit as st
import pandas as pd
import yfinance as yf
from util import rolling_beta, calculate_half_life, calculate_hurst, performance_metrics

strategy_returns = signals["position"].shift(1) * spread.pct_change().fillna(0)

metrics = performance_metrics(strategy_returns)
half_life = calculate_half_life(spread)
hurst = calculate_hurst(spread)

st.subheader("Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("CAGR", round(metrics["CAGR"], 4))
col2.metric("Sharpe", round(metrics["Sharpe"], 3))
col3.metric("Sortino", round(metrics["Sortino"], 3))
col4.metric("Max Drawdown", round(metrics["Max Drawdown"], 3))

st.metric("Half-Life (days)", round(half_life, 2))
st.metric("Hurst Exponent", round(hurst, 3))

# ----------------------------------
# USER INPUT SECTION
# ----------------------------------

st.sidebar.header("Universe Selection")

input_mode = st.sidebar.radio(
    "Select Input Mode",
    ["Manual Input", "Predefined US Banks"]
)

if input_mode == "Manual Input":
    tickers_input = st.sidebar.text_input(
        "Enter tickers (comma separated)",
        "JPM,BAC,C,GS,MS"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

else:
    tickers = ["JPM", "BAC", "C", "GS", "MS"]
    st.sidebar.write("Universe: US Major Banks")

# ----------------------------------
# SANITIZATION & VALIDATION
# ----------------------------------

def validate_tickers(ticker_list):
    valid = []
    invalid = []

    for ticker in ticker_list:
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                valid.append(ticker)
            else:
                invalid.append(ticker)
        except Exception:
            invalid.append(ticker)

    return valid, invalid


if st.sidebar.button("Validate Tickers"):
    valid, invalid = validate_tickers(tickers)

    if invalid:
        st.sidebar.warning(f"Invalid Tickers Removed: {invalid}")

    if len(valid) < 2:
        st.error("Minimum 2 valid tickers required.")
        st.stop()

    st.session_state["tickers"] = valid
    st.success(f"Validated Universe: {valid}")

# ----------------------------------
# LOAD DATA BUTTON
# ----------------------------------

if "tickers" in st.session_state:
    start_date = st.sidebar.date_input(
        "Start Date",
        pd.to_datetime("2015-01-01")
    )

    if st.sidebar.button("Load Data"):
        data = yf.download(
            st.session_state["tickers"],
            start=str(start_date),
            auto_adjust=True,
            progress=False
        )["Close"]

        data = data.dropna(how="all")

        if data.shape[0] < 250:
            st.error("Not enough historical data (minimum 250 observations).")
            st.stop()

        st.session_state["data"] = data
        st.success("Data Loaded Successfully")

beta_roll = rolling_beta(y, x)

st.plotly_chart(
    px.line(beta_roll, title="Rolling Hedge Ratio (60-day)"),
    use_container_width=True
)


rolling_vol = spread.pct_change().rolling(60).std() * np.sqrt(252)

st.plotly_chart(
    px.line(rolling_vol, title="Rolling Annualized Volatility of Spread"),
    use_container_width=True
)


fig_trade = go.Figure()
fig_trade.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))

long_entries = spread[signals["long"]]
short_entries = spread[signals["short"]]

fig_trade.add_trace(go.Scatter(
    x=long_entries.index,
    y=long_entries,
    mode="markers",
    marker=dict(symbol="triangle-up", size=8),
    name="Long Entry"
))

fig_trade.add_trace(go.Scatter(
    x=short_entries.index,
    y=short_entries,
    mode="markers",
    marker=dict(symbol="triangle-down", size=8),
    name="Short Entry"
))

fig_trade.update_layout(title="Trade Signals on Spread")

st.plotly_chart(fig_trade, use_container_width=True)

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

fig_acf, ax = plt.subplots()
plot_acf(spread.dropna(), ax=ax, lags=40)
st.pyplot(fig_acf)


# ----------------------------------
# DISPLAY DATA SUMMARY
# ----------------------------------

if "data" in st.session_state:
    st.subheader("Data Summary")
    st.write("Tickers:", st.session_state["data"].columns.tolist())
    st.write("Observations:", len(st.session_state["data"]))
    st.dataframe(st.session_state["data"].tail())
