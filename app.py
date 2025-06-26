import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import datetime as dt

st.set_page_config(page_title="Stock Beta & Vol Analyzer", layout="centered")

# ── Title style ────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; font-size:2.5rem'>📊 Stock Beta & Volatility Analyzer</h1>",
    unsafe_allow_html=True,
)

# ── Inputs ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker", "AAPL")
with col2:
    benchmark = st.text_input("Benchmark (e.g. ^GSPC)", "^GSPC")

today = dt.date.today()
default_end = today - dt.timedelta(days=1)
start = st.date_input("Start Date", dt.date(2023, 1, 1))
end = st.date_input("End Date", default_end)

# ── Helper: prices → returns ───────────────────────────────────────────────
@st.cache_data
def fetch_prices(tick, start, end):
    px = yf.download(tick, start=start, end=end, auto_adjust=True)["Close"]
    px.index = px.index.date
    return px

try:
    px_stock = fetch_prices(ticker, start, end)
    px_bench = fetch_prices(benchmark, start, end)

    prices = pd.concat([px_stock, px_bench], axis=1, join="inner").dropna()
    prices.columns = ["Stock_Price", "Benchmark_Price"]
    returns = prices.pct_change().dropna().rename(
        columns={"Stock_Price": "Stock", "Benchmark_Price": "Benchmark"}
    )

    if returns.empty:
        st.warning("No overlapping trading days. Adjust the date range.")
        st.stop()

    # ── Stats ──────────────────────────────────────────────────────────────
    beta = np.cov(returns["Stock"], returns["Benchmark"])[0, 1] / np.var(
        returns["Benchmark"]
    )
    alpha = returns["Stock"].mean() - beta * returns["Benchmark"].mean()
    r2 = returns["Stock"].corr(returns["Benchmark"]) ** 2
    sigma = returns["Stock"].std()

    # ── Metric cards with Explain buttons ──────────────────────────────────
    explainer = {
        "Beta": f"{ticker} moves about **{beta:.2f}×** the benchmark’s daily move.",
        "Alpha": f"On a day the benchmark is flat, {ticker} averages **{alpha:.3%}**.",
        "R²": f"About **{r2:.0%}** of {ticker}'s daily swings are explained by {benchmark}.",
        "σ": f"The stock’s typical one-day move is **±{sigma:.2%}**.",
    }

    col_a, col_b, col_c, col_d = st.columns(4)
    metric_cols = [col_a, col_b, col_c, col_d]
    labels = ["Beta", "Alpha", "R²", "σ"]
    values = [f"{beta:.2f}", f"{alpha:.3%}", f"{r2:.2f}", f"{sigma:.2%}"]

    for col, lab, val in zip(metric_cols, labels, values):
        col.metric(lab, val)
        # show explanation when button clicked
        if col.button("ℹ️ Explain", key=lab):
            col.info(explainer[lab])

    # ── Download buttons ───────────────────────────────────────────────────
    st.download_button(
        "Download prices + returns CSV",
        data=pd.concat([prices, returns], axis=1)
        .to_csv(index=True)
        .encode(),
        file_name="prices_and_returns.csv",
        mime="text/csv",
    )

    # ── Interactive scatter (Altair) ───────────────────────────────────────
    scatter = (
        alt.Chart(returns.reset_index())
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("Benchmark", title=f"{benchmark} Return"),
            y=alt.Y("Stock", title=f"{ticker} Return"),
            tooltip=["Date:T", "Stock", "Benchmark"],
        )
    )

    reg_line = (
        alt.Chart(returns.reset_index())
        .transform_regression("Benchmark", "Stock")
        .mark_line(color="red")
        .encode(x="Benchmark", y="Stock")
    )

    st.altair_chart((scatter + reg_line).interactive(), use_container_width=True)

except Exception as e:
    st.error("Error: " + str(e))


