import datetime as dt

import yfinance as yf


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
                return float(value.item())
            return float(value)
    except Exception:
        pass
    return float("nan")
