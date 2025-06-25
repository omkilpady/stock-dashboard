import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📊 Stock Beta & Volatility Analyzer")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
benchmark = st.text_input("Enter Benchmark (e.g. ^GSPC)", "^GSPC")
start = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end = st.date_input("End Date", pd.to_datetime("today"))

@st.cache_data
def get_data(tick):
    df = yf.download(tick, start=start, end=end)["Adj Close"]
    return df.pct_change().dropna()

try:
    stock_ret = get_data(ticker)
    bench_ret = get_data(benchmark)
    df = pd.concat([stock_ret, bench_ret], axis=1)
    df.columns = ["Stock", "Benchmark"]
    
    beta = np.cov(df["Stock"], df["Benchmark"])[0,1] / np.var(df["Benchmark"])
    alpha = df["Stock"].mean() - beta * df["Benchmark"].mean()
    r2 = np.corrcoef(df["Stock"], df["Benchmark"])[0,1]**2
    sigma = df["Stock"].std()

    st.write(f"**Beta**: {beta:.2f}")
    st.write(f"**Alpha**: {alpha:.4%}")
    st.write(f"**R²**: {r2:.2f}")
    st.write(f"**Standard Deviation (σ)**: {sigma:.2%}")

    # Chart
    fig, ax = plt.subplots()
    ax.scatter(df["Benchmark"], df["Stock"], alpha=0.5)
    ax.plot(df["Benchmark"], alpha + beta * df["Benchmark"], color='red')
    ax.set_xlabel("Benchmark Return")
    ax.set_ylabel("Stock Return")
    ax.set_title("Regression Line (Beta)")
    st.pyplot(fig)

except:
    st.warning("Please enter valid tickers and make sure data exists for the selected range.")
