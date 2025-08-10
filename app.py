
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import re
import io

from av_client import get_period_return
from report_builder import build_pdf_report, date_range_from_phrase, compute_metrics_table_alpha

st.set_page_config(page_title="Stock Report Generator (Alpha Vantage)", layout="wide")

st.title("Stock Report Generator")
st.caption("Type a request, get a clean report. Powered by Alpha Vantage free API.")

tab1, tab2, tab3 = st.tabs(["Ask in plain English", "Single Ticker Report", "Transcript to Summary (paste text)"])

@st.cache_data
def load_sector_csv(name: str):
    path = f"data/{name}.csv"
    return pd.read_csv(path)

def parse_query(q: str):
    ql = q.lower()

    m_count = re.search(r'(\d+)\s+(stocks|companies)', ql)
    count = int(m_count.group(1)) if m_count else 3

    if "tech" in ql:
        sector_name = "US Tech"
        sector_csv = "sectors_us_tech"
    else:
        sector_name = "US Healthcare"
        sector_csv = "sectors_us_healthcare"

    period_phrase = "last 3 months"
    if "last quarter" in ql or "previous quarter" in ql:
        period_phrase = "last quarter"
    elif "last month" in ql:
        period_phrase = "last month"
    elif "ytd" in ql or "year to date" in ql:
        period_phrase = "ytd"
    elif "last 3 months" in ql or "last three months" in ql:
        period_phrase = "last 3 months"

    benchmark = "SPY"

    m_by = re.search(r'by\s+(\d+)\s*%?', ql)
    min_outperf = float(m_by.group(1))/100.0 if m_by else None

    return {
        "count": count,
        "sector_name": sector_name,
        "sector_csv": sector_csv,
        "period_phrase": period_phrase,
        "benchmark": benchmark,
        "min_outperf": min_outperf
    }

with tab1:
    st.subheader("Ask in plain English")
    example = "List 5 companies in healthcare that beat the S&P 500 in the last 3 months"
    q = st.text_input("Your request", value=example)

    if st.button("Generate picks and report"):
        params = parse_query(q)
        start_date, end_date = date_range_from_phrase(params["period_phrase"])

        sector_df = load_sector_csv(params["sector_csv"])
        tickers = sector_df["Ticker"].dropna().unique().tolist()

        bench_ret = get_period_return(params["benchmark"], start_date, end_date)

        rows = []
        for t in tickers:
            r = get_period_return(t, start_date, end_date)
            if r is not None:
                rows.append({"Ticker": t, "Return": r})
        if not rows:
            st.error("Could not fetch data. Check your API key in Secrets, or try again later.")
        else:
            df = pd.DataFrame(rows)
            if bench_ret is not None:
                df["Outperformance"] = df["Return"] - bench_ret
                if params["min_outperf"] is not None:
                    df = df[df["Outperformance"] >= params["min_outperf"]]
            df = df.sort_values("Outperformance", ascending=False if bench_ret is not None else True)
            topn = df.head(params["count"]).reset_index(drop=True)

            st.success(f"Top {len(topn)} in {params['sector_name']} for {params['period_phrase']}")
            st.dataframe(topn)

            buff = io.BytesIO()
            title = f"{params['sector_name']} outperformers vs {params['benchmark']}"
            subtitle = f"Window {start_date} to {end_date}"
            build_pdf_report(buffer=buff, title=title, subtitle=subtitle, tickers=topn["Ticker"].tolist(),
                             benchmark=params["benchmark"], start_date=start_date, end_date=end_date,
                             metrics_fn=compute_metrics_table_alpha)
            st.download_button("Download PDF report", data=buff.getvalue(), file_name="stock_report.pdf")

with tab2:
    st.subheader("Single Ticker Report")
    tkr = st.text_input("Ticker", value="JNJ")
    period_choice = st.selectbox("Period", ["last quarter", "last 3 months", "last month", "ytd"])
    benchmark = st.selectbox("Benchmark proxy", ["SPY"])

    if st.button("Generate ticker PDF"):
        start_date, end_date = date_range_from_phrase(period_choice)
        buff = io.BytesIO()
        title = f"Report for {tkr}"
        subtitle = f"Window {start_date} to {end_date} vs {benchmark}"
        build_pdf_report(buffer=buff, title=title, subtitle=subtitle, tickers=[tkr],
                         benchmark=benchmark, start_date=start_date, end_date=end_date,
                         metrics_fn=compute_metrics_table_alpha)
        st.download_button("Download PDF report", data=buff.getvalue(), file_name=f"{tkr}_report.pdf")

with tab3:
    st.subheader("Transcript to key takeaways")
    st.caption("Paste earnings call text and I will summarize it heuristically. We can add AI later.")
    text = st.text_area("Paste transcript text", height=200)

    if st.button("Summarize transcript"):
        if not text.strip():
            st.error("Please paste text.")
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            bullets = lines[:8]
            summary = "Heuristic summary:\n" + "\n".join(f"- {b}" for b in bullets)
            st.text_area("Summary", value=summary, height=200)
            st.download_button("Download summary", data=summary, file_name="transcript_summary.txt")
