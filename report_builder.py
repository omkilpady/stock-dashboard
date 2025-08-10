
import datetime as dt
import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import numpy as np

from av_client import get_period_return, get_daily_series

def date_range_from_phrase(phrase: str):
    today = dt.date.today()

    def quarter_start_end(day: dt.date):
        q_end_months = [3,6,9,12]
        last_end = None
        for m in reversed(q_end_months):
            year = day.year if day.month > m else day.year - 1
            if m == 12:
                last_day = 31
            else:
                tmp = dt.date(year, m, 28) + dt.timedelta(days=4)
                last_day = (tmp - dt.timedelta(days=tmp.day)).day
            cand = dt.date(year, m, last_day)
            if cand < day:
                last_end = cand
                break
        prev_m = {3:12, 6:3, 9:6, 12:9}[last_end.month]
        prev_y = last_end.year - 1 if prev_m == 12 else last_end.year
        if prev_m == 12:
            prev_last_day = 31
        else:
            tmp = dt.date(prev_y, prev_m, 28) + dt.timedelta(days=4)
            prev_last_day = (tmp - dt.timedelta(days=tmp.day)).day
        prev_end = dt.date(prev_y, prev_m, prev_last_day)
        start = prev_end + dt.timedelta(days=1)
        return start, last_end

    pl = phrase.lower()
    if "last quarter" in pl:
        return quarter_start_end(today)
    if "last 3 months" in pl or "last three months" in pl:
        return today - dt.timedelta(days=90), today
    if "last month" in pl:
        return today - dt.timedelta(days=30), today
    if "ytd" in pl or "year to date" in pl:
        start = dt.date(today.year, 1, 1)
        return start, today
    return quarter_start_end(today)

def compute_metrics_table_alpha(ticker: str, benchmark: str, start_date: dt.date, end_date: dt.date):
    data = {"Metric": [], "Value": []}

    r = get_period_return(ticker, start_date, end_date)
    data["Metric"].append("Period return")
    data["Value"].append(f"{r:.2%}" if r is not None else "N/A")

    br = get_period_return(benchmark, start_date, end_date)
    data["Metric"].append("Benchmark return (SPY)")
    data["Value"].append(f"{br:.2%}" if br is not None else "N/A")

    if r is not None and br is not None:
        data["Metric"].append("Outperformance vs SPY")
        data["Value"].append(f"{(r-br):.2%}")
    else:
        data["Metric"].append("Outperformance vs SPY")
        data["Value"].append("N/A")

    series = get_daily_series(ticker)
    if series:
        dates = sorted(d for d in series.keys() if start_date <= d <= end_date)
        closes = [series[d] for d in dates]
        if len(closes) >= 2:
            rets = [closes[i]/closes[i-1]-1.0 for i in range(1, len(closes))]
            vol = float(np.std(rets)) * (252**0.5)
            data["Metric"].append("Ann. volatility (proxy)")
            data["Value"].append(f"{vol:.2%}")
        else:
            data["Metric"].append("Ann. volatility (proxy)")
            data["Value"].append("N/A")
    else:
        data["Metric"].append("Ann. volatility (proxy)")
        data["Value"].append("N/A")

    return pd.DataFrame(data)

def wrap_text(c, text, x, y, max_width, leading=14):
    lines = simpleSplit(text, c._fontname, c._fontsize, max_width)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y

def build_pdf_report(buffer, title: str, subtitle: str, tickers: list, benchmark: str, start_date: dt.date, end_date: dt.date, metrics_fn):
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    margin = 0.7*inch
    y = height - margin

    c.setFont("Helvetica-Bold", 18)
    y = wrap_text(c, title, margin, y, width - 2*margin, leading=22)
    c.setFont("Helvetica", 12)
    y = wrap_text(c, subtitle, margin, y-6, width - 2*margin, leading=16)
    c.setFont("Helvetica", 10)
    y = wrap_text(c, f"Benchmark {benchmark}   Period {start_date} to {end_date}", margin, y-6, width - 2*margin, leading=14)
    c.showPage()

    for t in tickers:
        c.setFont("Helvetica-Bold", 14)
        y = height - margin
        y = wrap_text(c, f"Ticker {t}", margin, y, width - 2*margin, leading=18)

        c.setFont("Helvetica", 10)
        df = metrics_fn(t, benchmark, start_date, end_date)
        col1_x = margin
        col2_x = margin + 2.6*inch
        row_y = y - 10
        for _, row in df.iterrows():
            c.drawString(col1_x, row_y, str(row["Metric"]))
            c.drawString(col2_x, row_y, str(row["Value"]))
            row_y -= 14

        row_y -= 10
        c.setFont("Helvetica-Oblique", 10)
        row_y = wrap_text(c, "Notes: paste transcript highlights or fundamentals in future versions.", margin, row_y, width - 2*margin, leading=14)

        c.showPage()

    c.save()
    buffer.seek(0)
