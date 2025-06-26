import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

st.title("📊 Stock Beta & Volatility Analyzer")

# ── Inputs ─────────────────────────────────────────────────────────────────
ticker    = st.text_input("Enter Stock Ticker", "AAPL")
benchmark = st.text_input("Enter Benchmark (e.g. ^GSPC)", "^GSPC")

today       = dt.date.today()
default_end = today - dt.timedelta(days=1)
start       = st.date_input("Start Date", dt.date(2023, 1, 1))
end         = st.date_input("End Date", default_end)

# ── Helper: fetch adjusted-close prices ────────────────────────────────────
@st.cache_data
def fetch_prices(tick, start, end):
    px = yf.download(tick, start=start, end=end, auto_adjust=True)["Close"]
    px.index = px.index.date  # drop intraday time part
    return px

try:
    stock_px  = fetch_prices(ticker,    start, end)
    bench_px  = fetch_prices(benchmark, start, end)

    # Align dates (inner join) and drop blanks
    prices = pd.concat([stock_px, bench_px], axis=1, join="inner").dropna()
    prices.columns = ["Stock_Price", "Benchmark_Price"]

    if prices.empty:
        st.warning("No overlapping price data—try a different date range.")
        st.stop()

    # Compute returns
    returns = prices.pct_change().dropna()
    returns.columns = ["Stock", "Benchmark"]

    # Combined DataFrame (for download)
    df_full = prices.join(returns, how="inner")
    df_full.index.name = "Date"

    # ── Stats ──────────────────────────────────────────────────────────────
    beta  = np.cov(returns["Stock"], returns["Benchmark"])[0, 1] / np.var(returns["Benchmark"])
    alpha = returns["Stock"].mean() - beta * returns["Benchmark"].mean()
    r2    = np.corrcoef(returns["Stock"], returns["Benchmark"])[0, 1] ** 2
    sigma = returns["Stock"].std()

    cols = st.columns(4)
    cols[0].metric("Beta", f"{beta:.2f}")
    cols[1].metric("Alpha", f"{alpha:.4%}")
    cols[2].metric("R²", f"{r2:.2f}")
    cols[3].metric("σ (Std Dev)", f"{sigma:.2%}")

    # ── Download buttons ──────────────────────────────────────────────────
    st.download_button(
        "Download prices + returns CSV",
        data=df_full.to_csv().encode("utf-8"),
        file_name="prices_and_returns.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download returns-only CSV",
        data=returns.to_csv().encode("utf-8"),
        file_name="returns.csv",
        mime="text/csv",
    )

    # ── Scatter plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    ax.scatter(returns["Benchmark"], returns["Stock"], alpha=0.5)
    ax.plot(
        returns["Benchmark"],
        alpha + beta * returns["Benchmark"],
        color="red"
    )
    ax.set_xlabel(f"{benchmark} Return")
    ax.set_ylabel(f"{ticker} Return")
    ax.set_title("Regression Line (Beta)")
    st.pyplot(fig)

except Exception as e:
    st.warning("Something went wrong. Double-check tickers and date range.")
    st.exception(e)


