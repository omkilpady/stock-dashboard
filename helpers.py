import datetime as dt

import yfinance as yf
import requests
from typing import List


def fx_to_usd(value: float, currency: str) -> float:
    """Convert value to USD if a FX rate is available."""
    if currency == "USD":
        return value
    try:
        pair = f"{currency}USD=X"
        rate = yf.download(pair, period="1d", auto_adjust=True)["Close"].iloc[-1]
        return value * rate
    except Exception:
        return value


def price_on_date(symbol: str, date: dt.date) -> float:
    """Get the first available closing price on or after the date."""
    try:
        data = yf.download(
            symbol,
            start=date,
            end=date + dt.timedelta(days=5),
            auto_adjust=True,
        )["Close"]
        if not data.empty:
            value = data.iloc[0]
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
    except Exception:
        pass
    return float("nan")


def search_tickers(query: str) -> List[str]:
    """Return a list of matching ticker symbols from Yahoo Finance."""
    if not query:
        return []
    url = (
        "https://query2.finance.yahoo.com/v1/finance/search"
        f"?q={query}&quotes_count=10&news_count=0"
    )
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        data = resp.json().get("quotes", [])
        return [item["symbol"] for item in data if "symbol" in item]
    except Exception:
        return []
