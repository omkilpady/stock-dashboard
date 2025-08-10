
import os
import requests
import datetime as dt

API_URL = "https://www.alphavantage.co/query"

def _api_key():
    try:
        import streamlit as st
        if "ALPHA_VANTAGE_API_KEY" in st.secrets:
            return st.secrets["ALPHA_VANTAGE_API_KEY"]
    except Exception:
        pass
    return os.getenv("ALPHA_VANTAGE_API_KEY", "")

def get_daily_series(symbol: str):
    key = _api_key()
    if not key:
        return None
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": key
    }
    r = requests.get(API_URL, params=params, timeout=20)
    if r.status_code != 200:
        return None
    js = r.json()
    series = js.get("Time Series (Daily)")
    if not series:
        return None
    out = {}
    for d, row in series.items():
        try:
            out[dt.date.fromisoformat(d)] = float(row["5. adjusted close"])
        except Exception:
            continue
    return out

def get_period_return(symbol: str, start_date: dt.date, end_date: dt.date):
    data = get_daily_series(symbol)
    if not data:
        return None
    available = sorted(data.keys())
    if not available:
        return None
    def nearest(target):
        return min(available, key=lambda d: abs(d - target))
    s = nearest(start_date)
    e = nearest(end_date)
    try:
        return data[e] / data[s] - 1.0
    except Exception:
        return None
