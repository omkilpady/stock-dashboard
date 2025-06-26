from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit page config
st.set_page_config(page_title="Stock Beta Analyzer", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #f2f2f2;
        }
        .css-18e3th9, .css-1d391kg {
            background-color: #1a1a1a;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Stock Beta & Volatility Analyzer")

# Inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL")
benchmark = st.text_input("Enter Benchmark (e.g. ^GSPC)", "^GSPC")
today = date.today()
default_end = today - timedelta(days=1)
start = st.date_input("Start Date", date(2023, 1, 1))
end = st.date_input("End Date", default_end)

# Fetch returns
@st.cache_data
def fetch_returns(tick, start, end):
    px = yf.download(tick, start=start, end=end, auto_adjust=True)["Close"]
    return px.pct_change().dropna()

try:
    stock_ret = fetch_returns(ticker, start, end)
    bench_ret = fetch_returns(benchmark, start, end)

    if stock_ret.empty or bench_ret.empty:
        st.warning("No price data for one or both tickers in that date range.")
        st.stop()

    df = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    df.columns = ["Stock", "Benchmark"]

    if df.empty:
        st.warning("No overlapping trading days—try a different date range.")
        st.stop()

    # Metrics
    beta = np.cov(df["Stock"], df["Benchmark"])[0, 1] / np.var(df["Benchmark"])
    alpha = df["Stock"].mean() - beta * df["Benchmark"].mean()
    r2 = np.corrcoef(df["Stock"], df["Benchmark"])[0, 1] ** 2
    sigma = df["Stock"].std()
    rho = df.corr().iloc[0, 1]
    mean = df["Stock"].mean()
    sharpe = mean / sigma if sigma != 0 else 0

    st.subheader("📌 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### Beta")
        st.markdown(f"<span style='font-size: 28px'>{beta:.2f}</span>", unsafe_allow_html=True)
        st.markdown("ℹ️", help=f"When {benchmark} moves 1%, {ticker} tends to move {beta:.2f}%.")

    with col2:
        st.markdown("### Alpha")
        st.markdown(f"<span style='font-size: 28px'>{alpha:.4%}</span>", unsafe_allow_html=True)
        st.markdown("ℹ️", help=f"When {benchmark} is flat, {ticker} returns {alpha:.4%} on average.")

    with col3:
        st.markdown("### R²")
        st.markdown(f"<span style='font-size: 28px'>{r2:.2f}</span>", unsafe_allow_html=True)
        st.markdown("ℹ️", help="R² indicates how well the stock's returns are explained by the benchmark.")

    with col4:
        st.markdown("### σ")
        st.markdown(f"<span style='font-size: 28px'>{sigma:.2%}</span>", unsafe_allow_html=True)
        st.markdown("ℹ️", help="σ (standard deviation) shows day-to-day volatility.")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown("### ρ (Correlation)")
        st.markdown(f"<span style='font-size: 28px'>{rho:.2f}</span>", unsafe_allow_html=True)
        st.markdown("ℹ️", help="ρ shows how closely the stock and benchmark move together.")

    with col6:
        st.markdown("### Mean Daily Return")
        st.markdown(f"<span style='font-size: 28px'>{mean:.4%}</span>", unsafe_allow_html=True)
        st.markdown("ℹ️", help="The average daily return of the stock.")

    with col7:
        st.markdown("### Sharpe Ratio")
        st.markdown(f"<span style='font-size: 28px'>{sharpe:.2f}</span>", unsafe_allow_html=True)
        st.markdown("ℹ️", help="Sharpe ratio = mean return ÷ volatility (risk-adjusted return).")

    # Chart
    st.subheader("📉 Regression Scatter Plot")
    fig, ax = plt.subplots()
    ax.scatter(df["Benchmark"], df["Stock"], alpha=0.5)
    ax.plot(df["Benchmark"], alpha + beta * df["Benchmark"], color="red")
    ax.set_xlabel(f"{benchmark} Return")
    ax.set_ylabel(f"{ticker} Return")
    ax.set_title("Regression Line (Beta)")
    st.pyplot(fig)

    st.caption(f"""
        Each dot shows daily returns for {ticker} vs {benchmark}.  
        The red line is a regression line. Its slope shows how sensitive {ticker} is to {benchmark} during the selected timeframe ({start} to {end}).
    """)

except Exception as e:
    st.warning("Something went wrong. Double-check tickers and date range.")
    st.exception(e)

