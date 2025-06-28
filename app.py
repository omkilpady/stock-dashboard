 import streamlit as st
 import yfinance as yf
 import pandas as pd
 import numpy as np
 import altair as alt
 import datetime as dt
+from typing import List, Dict
 
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
@@ -161,27 +162,213 @@ try:
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
 
+    # ── Portfolio Tracker ─────────────────────────────────────────────────
+    st.subheader("📒 Portfolio Tracker")
+
+    if "portfolio" not in st.session_state:
+        st.session_state["portfolio"] = []
+
+    def validate_symbol(symbol: str):
+        """Check if ticker exists and return metadata."""
+        try:
+            info = yf.Ticker(symbol).info
+            name = info.get("shortName") or info.get("longName")
+            if name:
+                return {"currency": info.get("currency", "USD"), "name": name}
+        except Exception:
+            pass
+        return None
+
+    def fx_to_usd(value: float, currency: str) -> float:
+        """Convert value to USD if a FX rate is available."""
+        if currency == "USD":
+            return value
+        try:
+            pair = f"{currency}USD=X"
+            rate = (
+                yf.download(pair, period="1d", auto_adjust=True)["Close"].iloc[-1]
+            )
+            return value * rate
+        except Exception:
+            return value
+
+    def price_on_date(symbol: str, date: dt.date) -> float:
+        """Get the first available closing price on or after the date."""
+        try:
+            data = yf.download(
+                symbol,
+                start=date,
+                end=date + dt.timedelta(days=5),
+                auto_adjust=True,
+            )["Close"]
+            if not data.empty:
+                return data.iloc[0]
+        except Exception:
+            pass
+        return float("nan")
+
+    with st.form("add_asset"):
+        a_cols = st.columns(4)
+        sym = a_cols[0].text_input("Symbol")
+        date_bought = a_cols[1].date_input("Buy Date", today)
+        shares = a_cols[2].number_input("Shares", min_value=0.0, step=0.01)
+        track_usd = a_cols[3].checkbox("Track in USD", value=True)
+        submitted = st.form_submit_button("Add")
+
+        if submitted and sym:
+            info = validate_symbol(sym)
+            if not info:
+                st.error("Invalid ticker symbol")
+            else:
+                px = price_on_date(sym, date_bought)
+                if np.isnan(px):
+                    st.error("Price not available for that date")
+                else:
+                    invested = shares * px
+                    asset_currency = info["currency"]
+                    currency = "USD" if track_usd else asset_currency
+                    if track_usd:
+                        invested = fx_to_usd(invested, asset_currency)
+                    st.session_state["portfolio"].append(
+                        {
+                            "symbol": sym.upper(),
+                            "date": date_bought,
+                            "shares": shares,
+                            "invested": invested,
+                            "asset_currency": asset_currency,
+                            "currency": currency,
+                        }
+                    )
+
+    if st.session_state["portfolio"]:
+        pf_df = pd.DataFrame(st.session_state["portfolio"])
+
+        try:
+            latest = (
+                yf.download(
+                    list(pf_df["symbol"].unique()),
+                    period="1d",
+                    auto_adjust=True,
+                )["Close"].iloc[-1]
+            )
+        except Exception:
+            latest = pd.Series(dtype=float)
+
+        pf_df["Current"] = pf_df["symbol"].map(latest)
+
+        def cur_price(row: pd.Series) -> float:
+            price = row["Current"]
+            if row["currency"] == "USD":
+                price = fx_to_usd(price, row["asset_currency"])
+            return price
+
+        pf_df["Current"] = pf_df.apply(cur_price, axis=1)
+        pf_df["Current Value"] = pf_df["shares"] * pf_df["Current"]
+        pf_df["P/L"] = pf_df["Current Value"] - pf_df["invested"]
+
+        st.dataframe(
+            pf_df[["symbol", "date", "shares", "currency", "invested", "Current Value", "P/L"]],
+            use_container_width=True,
+        )
+
+        # ----- Edit/Delete Buttons -----
+        for i in pf_df.index:
+            e_col, d_col = st.columns(2)
+            if e_col.button("Edit", key=f"edit_{i}"):
+                st.session_state["edit_idx"] = int(i)
+            if d_col.button("Delete", key=f"del_{i}"):
+                st.session_state["portfolio"].pop(int(i))
+                st.experimental_rerun()
+
+        if "edit_idx" in st.session_state:
+            idx = st.session_state.pop("edit_idx")
+            asset = st.session_state["portfolio"][idx]
+            with st.form("edit_asset"):
+                e_cols = st.columns(4)
+                sym_e = e_cols[0].text_input("Symbol", asset["symbol"])
+                date_e = e_cols[1].date_input("Buy Date", asset["date"])
+                shares_e = e_cols[2].number_input("Shares", value=asset["shares"], min_value=0.0, step=0.01)
+                usd_e = e_cols[3].checkbox("Track in USD", value=asset["currency"] == "USD")
+                save = st.form_submit_button("Save")
+
+                if save:
+                    info = validate_symbol(sym_e)
+                    if info:
+                        px = price_on_date(sym_e, date_e)
+                        if not np.isnan(px):
+                            invested = shares_e * px
+                            asset_currency = info["currency"]
+                            currency = "USD" if usd_e else asset_currency
+                            if usd_e:
+                                invested = fx_to_usd(invested, asset_currency)
+                            st.session_state["portfolio"][idx] = {
+                                "symbol": sym_e.upper(),
+                                "date": date_e,
+                                "shares": shares_e,
+                                "invested": invested,
+                                "asset_currency": asset_currency,
+                                "currency": currency,
+                            }
+                            st.experimental_rerun()
+
+        # ----- Return Chart -----
+        try:
+            price_series = []
+            for asset in st.session_state["portfolio"]:
+                hist = yf.download(
+                    asset["symbol"],
+                    start=asset["date"],
+                    end=today + dt.timedelta(days=1),
+                    auto_adjust=True,
+                )["Close"]
+                if asset["currency"] == "USD":
+                    fx_hist = yf.download(
+                        f"{asset['asset_currency']}USD=X",
+                        start=asset["date"],
+                        end=today + dt.timedelta(days=1),
+                        auto_adjust=True,
+                    )["Close"]
+                    hist = hist.mul(fx_hist, fill_value=np.nan)
+                price_series.append(hist * asset["shares"])
+
+            if price_series:
+                portfolio_val = pd.concat(price_series, axis=1).fillna(method="ffill").sum(axis=1)
+                chart_df = portfolio_val.reset_index().rename(columns={0: "Value", "index": "Date"})
+                chart = (
+                    alt.Chart(chart_df)
+                    .mark_line()
+                    .encode(
+                        x="Date:T",
+                        y=alt.Y("Value:Q", title="Portfolio Value"),
+                        tooltip=["Date:T", alt.Tooltip("Value:Q", format=".2f")],
+                    )
+                    .interactive()
+                )
+                st.altair_chart(chart, use_container_width=True)
+        except Exception:
+            pass
+
 except Exception as e:
     st.error(f"Error: {e}")
 
EOF
)
