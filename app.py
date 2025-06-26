import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import datetime as dt

st.set_page_config(page_title="Stock Beta & Vol Analyzer", layout="centered")

# ── Title ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; font-size:2.4rem'>📊 Stock Beta & Volatility Analyzer</h1>",
    unsafe_allow_html=True,
)

# ── Inputs ─────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    ticker = st.text_input("Stock Ticker", "AAPL")
with c2:
    benchmark = st.text_input("Benchmark (e.g. ^GSPC)", "^GSPC")

today = dt.date.today()
default_end = today - dt.timedelta(days=1)
start = st.date_input("Start Date", dt.date(2023, 1, 1))
end = st.date_input("End Date", default_end)

# ── Fetch prices ───────────────────────────────────────────────────────────
@st.cache_data
def fetch_px(tick, start, end):
    px = yf.download(tick, start=start, end=end, auto_adjust=True)["Close"]
    px.index = px.index.date
    return px

try:
    px_stock = fetch_px(ticker, start, end)
    px_bench = fetch_px(benchmark, start, end)

    # Align prices and name columns
    prices = pd.concat([px_stock, px_bench], axis=1, join="inner").dropna()
    prices.columns = ["Stock_Price", "Benchmark_Price"]
    prices.index.name = "Date"

    # Daily returns
    returns = prices.pct_change().dropna().rename(
        columns={"Stock_Price": "Stock", "Benchmark_Price": "Benchmark"}
    )
    returns.index.name = "Date"

    if returns.empty:
        st.warning("No overlapping trading days. Adjust the date range.")
        st.stop()

    # ── Stats ──────────────────────────────────────────────────────────────
    cov = np.cov(returns["Stock"], returns["Benchmark"])[0, 1]
    var_b = np.var(returns["Benchmark"])
    beta = cov / var_b
    alpha = returns["Stock"].mean() - beta * returns["Benchmark"].mean()
    r2 = returns["Stock"].corr(returns["Benchmark"]) ** 2
    sigma = returns["Stock"].std()
    corr = returns["Stock"].corr(returns["Benchmark"])
    mean_ret = returns["Stock"].mean()
    sharpe = (mean_ret / sigma) * np.sqrt(252)

    # ── Metric cards + tooltips ────────────────────────────────────────────
    metrics = {
        "Beta": (
            f"{beta:.2f}",
            f"A beta of {beta:.2f} means **{ticker}** tends to move "
            f"{abs(beta):.2f}× the benchmark each day "
            f"{'in the same direction' if beta >= 0 else 'in the opposite direction'}.",
        ),
        "Alpha": (
            f"{alpha:.3%}",
            f"When **{benchmark}** is flat, **{ticker}** averages {alpha:.3%} extra return.",
        ),
        "R²": (
            f"{r2:.2f}",
            f"{r2:.0%} of **{ticker}**’s daily moves are explained by **{benchmark}**.",
        ),
        "σ": (
            f"{sigma:.2%}",
            f"Typical one-day move (volatility) for **{ticker}** is ±{sigma:.2%}.",
        ),
        "ρ": (
            f"{corr:.2f}",
            "Correlation ranges −1 to +1; closer to +1 means they often move together.",
        ),
        "Mean": (
            f"{mean_ret:.3%}",
            f"Average daily return for **{ticker}** over the selected period.",
        ),
        "Sharpe": (
            f"{sharpe:.2f}",
            "Risk-adjusted return (annualized). Above 1 is generally considered good.",
        ),
    }

    st.subheader("Key Metrics")
    rows = [st.columns(4), st.columns(3)]
    for idx, (label, (val, expl)) in enumerate(metrics.items()):
        col = rows[0][idx] if idx < 4 else rows[1][idx - 4]
        col.metric(label, val, help=expl)

    # ── Optional Charts ────────────────────────────────────────────────────
    st.subheader("📈 Optional Charts")
    show_price = st.checkbox("Show price history")
    show_ret   = st.checkbox("Show cumulative return")

    if show_price:
        price_df = prices.reset_index().melt(id_vars="Date", value_name="Price")
        price_chart = (
            alt.Chart(price_df)
            .mark_line()
            .encode(
                x="Date:T",
                y=alt.Y("Price:Q", title="Adjusted Close Price"),
                color="variable:N",
                tooltip=["Date:T", "variable:N", "Price:Q"],
            )
            .interactive()
        )
        st.altair_chart(price_chart, use_container_width=True)

    if show_ret:
        cum_ret = (returns + 1).cumprod() - 1
        cum_df = cum_ret.reset_index().melt(id_vars="Date", value_name="CumReturn")
        ret_chart = (
            alt.Chart(cum_df)
            .mark_line()
            .encode(
                x="Date:T",
                y=alt.Y("CumReturn:Q", title="Cumulative Return"),
                color="variable:N",
                tooltip=[
                    "Date:T",
                    "variable:N",
                    alt.Tooltip("CumReturn:Q", format=".2%"),
                ],
            )
            .interactive()
        )
        st.altair_chart(ret_chart, use_container_width=True)

    # ── Download button ────────────────────────────────────────────────────
    full_df = prices.join(returns, how="inner")
    st.download_button(
        "Download prices + returns CSV",
        data=full_df.to_csv().encode(),
        file_name="prices_and_returns.csv",
        mime="text/csv",
    )

    # ── Scatter + regression line ──────────────────────────────────────────
    scatter = (
        alt.Chart(returns.reset_index())
        .mark_circle(size=60, opacity=0.5)
        .encode(
            x=alt.X("Benchmark", title=f"{benchmark} Daily Return"),
            y=alt.Y("Stock", title=f"{ticker} Daily Return"),
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

    # ── Friendly chart caption ─────────────────────────────────────────────
    st.markdown(
        f"""
**Chart guide**

* **Dots** – each trading day’s paired returns for **{ticker}** and **{benchmark}** from **{start}** to **{end}**.  
* **Red line** – “best-fit” trend; its slope is the Beta above.  
* Steeper line ⇒ higher sensitivity; flat or downward line ⇒ little or opposite sensitivity.
""",
        unsafe_allow_html=True,
    )

except Exception as e:
    st.error(f"Error: {e}")
