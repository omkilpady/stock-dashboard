import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

st.title("📊 Stock Beta & Volatility Analyzer")

# ── Inputs ──────────────────────────────────────────────────────────────────
ticker    = st.text_input("Enter Stock Ticker", "AAPL")
benchmark = st.text_input("Enter Benchmark (e.g. ^GSPC)", "^GSPC")

today       = dt.date.today()
default_end = today - dt.timedelta(days=1)          # yesterday so data exists
start       = st.date_input("Start Date", dt.date(2023, 1, 1))
end         = st.date_input("End Date", default_end)

# ── Helper: fetch daily % returns ───────────────────────────────────────────
@st.cache_data
def fetch_returns(tick, start, end):
    px = yf.download(tick, start=start, end=end, auto_adjust=True)["Close"]
    return px.pct_change().dropna()

try:
    stock_ret = fetch_returns(ticker,    start, end)
    bench_ret = fetch_returns(benchmark, start, end)

    if stock_ret.empty or bench_ret.empty:
        st.warning("No price data for one or both tickers in that date range.")
        st.stop()

    # Combine and enforce column names explicitly
    df = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    df.columns = ["Stock", "Benchmark"]  # <-- bullet-proof rename

    if df.empty:
        st.warning("No overlapping trading days—try a different date range.")
        st.stop()

    # ── Stats ───────────────────────────────────────────────────────────────
    beta  = np.cov(df["Stock"], df["Benchmark"])[0, 1] / np.var(df["Benchmark"])
    alpha = df["Stock"].mean() - beta * df["Benchmark"].mean()
    r2    = np.corrcoef(df["Stock"], df["Benchmark"])[0, 1] ** 2
    sigma = df["Stock"].std()

    st.write(f"**Beta**: {beta:.2f}")
    st.write(f"**Alpha**: {alpha:.4%}")
    st.write(f"**R²**: {r2:.2f}")
    st.write(f"**Standard Deviation (σ)**: {sigma:.2%}")

    # ── Chart ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    ax.scatter(df["Benchmark"], df["Stock"], alpha=0.5)
    ax.plot(df["Benchmark"],
            alpha + beta * df["Benchmark"],
            color="red")
    ax.set_xlabel(f"{benchmark} Return")
    ax.set_ylabel(f"{ticker} Return")
    ax.set_title("Regression Line (Beta)")
    st.pyplot(fig)

except Exception as e:
    st.warning("Something went wrong. Double-check tickers and date range.")
    st.exception(e)
