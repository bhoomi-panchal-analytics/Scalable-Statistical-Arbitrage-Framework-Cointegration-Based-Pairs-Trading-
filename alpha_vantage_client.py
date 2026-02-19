import requests

API_KEY = "YOUR_API_KEY"

def fetch_alpha_vantage(symbol):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "full"
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"],
        orient="index"
    )

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df["5. adjusted close"].astype(float)

    return df
