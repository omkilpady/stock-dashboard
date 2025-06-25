import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

st.title("📊 Stock Beta & Volatility Analyzer")

# ----- Inputs --------------------------------------------------------------
ticker     = st.text_input("Enter Stock Ticker", "AAPL")
benchmark  = st.text_input("Enter Benchmark (e.g. ^GSPC)", "^GSPC")

today       = dt.date.today()
default_end = today - dt.timedelta(days=1)          # yesterday to ensure data exists
start       = st.date_input("Start Date", dt.date(2023, 1, 1))
end         = st.date_input("End Date", default_end)

# ----- Data pull -----------------------------------------------------------
@st.cache_data
def get_data(tick, start, end):
    df = yf.download(tick, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {tick} in this date range.")
    return df['Close'].pct_change().dropna()

try:
    stock_ret = get_data(ticker, start, end)
    bench_ret = get_data(benchmark, start, end)

    # keep only overlapping trading days and drop any NaNs
    df = (
        pd.concat([stock_ret, bench_ret], axis=1, join="inner")
        .dropna()
        .rename(columns={0: "Stock", 1: "Benchmark"})
    )

    if df.empty:
        st.warning("No overlapping data in that date range. Try different dates.")
        st.stop()

    # ----- Stats -----------------------------------------------------------
    beta  = np.cov(df["Stock"], df["Benchmark"])[0, 1] / np.var(df["Benchmark"])
    alpha = df["Stock"].mean() - beta * df["Benchmark"].mean()
    r2    = np.corrcoef(df["Stock"], df["Benchmark"])[0, 1] ** 2
    sigma = df["Stock"].std()

    st.write(f"**Beta**: {beta:.2f}")
    st.write(f"**Alpha**: {alpha:.4%}")
    st.write(f"**R²**: {r2:.2f}")
    st.write(f"**Standard Deviation (σ)**: {sigma:.2%}")

    # ----- Chart -----------------------------------------------------------
    fig, ax = plt.subplots()
    ax.scatter(df["Benchmark"], df["Stock"], alpha=0.5)
    ax.plot(df["Benchmark"], alpha + beta * df["Benchmark"], color="red")
    ax.set_xlabel("Benchmark Return")
    ax.set_ylabel("Stock Return")
    ax.set_title("Regression Line (Beta)")
    st.pyplot(fig)

except Exception as e:
    st.warning("Please enter valid tickers and make sure data exists for the selected range.")
    st.exception(e)
