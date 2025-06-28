import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import datetime as dt
from typing import List, Dict

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

    prices = pd.concat([px_stock, px_bench], axis=1, join="inner").dropna()
    prices.columns = ["Stock_Price", "Benchmark_Price"]
    prices.index.name = "Date"

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
    show_ret = st.checkbox("Show cumulative return")

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
                tooltip=["Date:T", "variable:N", alt.Tooltip("CumReturn:Q", format=".2%")],
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

    st.markdown(
        f"""
**Chart guide**

* **Dots** – each trading day’s paired returns for **{ticker}** and **{benchmark}** from **{start}** to **{end}**.  
* **Red line** – “best-fit” trend; its slope is the Beta above.  
* Steeper line ⇒ higher sensitivity; flat or downward line ⇒ little or opposite sensitivity.
""",
        unsafe_allow_html=True,
    )

    # ── Portfolio Tracker ─────────────────────────────────────────────────
    st.subheader("📒 Portfolio Tracker")

    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    def validate_symbol(symbol: str):
        """Check if ticker exists and return metadata."""
        try:
            info = yf.Ticker(symbol).info
            name = info.get("shortName") or info.get("longName")
            if name:
                return {"currency": info.get("currency", "USD"), "name": name}
        except Exception:
            pass
        return None

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
                return data.iloc[0]
        except Exception:
            pass
        return float("nan")

    with st.form("add_asset"):
        a_cols = st.columns(4)
        sym = a_cols[0].text_input("Symbol")
        date_bought = a_cols[1].date_input("Buy Date", today)
        shares = a_cols[2].number_input("Shares", min_value=0.0, step=0.01)
        track_usd = a_cols[3].checkbox("Track in USD", value=True)
        submitted = st.form_submit_button("Add")

        if submitted and sym:
            info = validate_symbol(sym)
            if not info:
                st.error("Invalid ticker symbol")
            else:
                px = price_on_date(sym, date_bought)
                if np.isnan(px):
                    st.error("Price not available for that date")
                else:
                    invested = shares * px
                    asset_currency = info["currency"]
                    currency = "USD" if track_usd else asset_currency
                    if track_usd:
                        invested = fx_to_usd(invested, asset_currency)
                    st.session_state["portfolio"].append(
                        {
                            "symbol": sym.upper(),
                            "date": date_bought,
                            "shares": shares,
                            "invested": invested,
                            "asset_currency": asset_currency,
                            "currency": currency,
                        }
                    )

    if st.session_state["portfolio"]:
        pf_df = pd.DataFrame(st.session_state["portfolio"])

        try:
            latest = (
                yf.download(
                    list(pf_df["symbol"].unique()),
                    period="1d",
                    auto_adjust=True,
                )["Close"].iloc[-1]
            )
        except Exception:
            latest = pd.Series(dtype=float)

        pf_df["Current"] = pf_df["symbol"].map(latest)

        def cur_price(row: pd.Series) -> float:
            price = row["Current"]
            if row["currency"] == "USD":
                price = fx_to_usd(price, row["asset_currency"])
            return price

        pf_df["Current"] = pf_df.apply(cur_price, axis=1)
        pf_df["Current Value"] = pf_df["shares"] * pf_df["Current"]
        pf_df["P/L"] = pf_df["Current Value"] - pf_df["invested"]

        st.dataframe(
            pf_df[["symbol", "date", "shares", "currency", "invested", "Current Value", "P/L"]],
            use_container_width=True,
        )

        for i in pf_df.index:
            e_col, d_col = st.columns(2)
            if e_col.button("Edit", key=f"edit_{i}"):
                st.session_state["edit_idx"] = int(i)
            if d_col.button("Delete", key=f"del_{i}"):
                st.session_state["portfolio"].pop(int(i))
                st.experimental_rerun()

        if "edit_idx" in st.session_state:
            idx = st.session_state.pop("edit_idx")
            asset = st.session_state["portfolio"][idx]
            with st.form("edit_asset"):
                e_cols = st.columns(4)
                sym_e = e_cols[0].text_input("Symbol", asset["symbol"])
                date_e = e_cols[1].date_input("Buy Date", asset["date"])
                shares_e = e_cols[2].number_input("Shares", value=asset["shares"], min_value=0.0, step=0.01)
                usd_e = e_cols[3].checkbox("Track in USD", value=asset["currency"] == "USD")
                save = st.form_submit_button("Save")

                if save:
                    info = validate_symbol(sym_e)
                    if info:
                        px = price_on_date(sym_e, date_e)
                        if not np.isnan(px):
                            invested = shares_e * px
                            asset_currency = info["currency"]
                            currency = "USD" if usd_e else asset_currency
                            if usd_e:
                                invested = fx_to_usd(invested, asset_currency)
                            st.session_state["portfolio"][idx] = {
                                "symbol": sym_e.upper(),
                                "date": date_e,
                                "shares": shares_e,
                                "invested": invested,
                                "asset_currency": asset_currency,
                                "currency": currency,
                            }
                            st.experimental_rerun()

        try:
            price_series = []
            for asset in st.session_state["portfolio"]:
                hist = yf.download(
                    asset["symbol"],
                    start=asset["date"],
                    end=today + dt.timedelta(days=1),
                    auto_adjust=True,
                )["Close"]
                if asset["currency"] == "USD":
                    fx_hist = yf.download(
                        f"{asset['asset_currency']}USD=X",
                        start=asset["date"],
                        end=today + dt.timedelta(days=1),
                        auto_adjust=True,
                    )["Close"]
                    hist = hist.mul(fx_hist, fill_value=np.nan)
                price_series.append(hist * asset["shares"])

            if price_series:
                portfolio_val = pd.concat(price_series, axis=1).fillna(method="ffill").sum(axis=1)
                chart_df = portfolio_val.reset_index().rename(columns={0: "Value", "index": "Date"})
                chart = (
                    alt.Chart(chart_df)
                    .mark_line()
                    .encode(
                        x="Date:T",
                        y=alt.Y("Value:Q", title="Portfolio Value"),
                        tooltip=["Date:T", alt.Tooltip("Value:Q", format=".2f")],
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
        except Exception:
            pass

except Exception as e:
    st.error(f"Error: {e}")
