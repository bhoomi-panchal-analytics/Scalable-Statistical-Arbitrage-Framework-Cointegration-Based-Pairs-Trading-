import yfinance as yf
import pandas as pd
import os

DATA_DIR = "data"

def download_price_data(tickers, start="2015-01-01", end=None):
    """
    Downloads adjusted close prices for a list of tickers.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']

    data = data.dropna(how='all')

    file_path = os.path.join(DATA_DIR, "prices.csv")
    data.to_csv(file_path)

    return data


def load_local_data():
    """
    Loads locally saved price data.
    """
    file_path = os.path.join(DATA_DIR, "prices.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError("No local data found. Run download_price_data first.")

    return pd.read_csv(file_path, index_col=0, parse_dates=True)
