import streamlit as st
import pandas as pd
import yfinance as yf

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

# ----------------------------------
# DISPLAY DATA SUMMARY
# ----------------------------------

if "data" in st.session_state:
    st.subheader("Data Summary")
    st.write("Tickers:", st.session_state["data"].columns.tolist())
    st.write("Observations:", len(st.session_state["data"]))
    st.dataframe(st.session_state["data"].tail())
