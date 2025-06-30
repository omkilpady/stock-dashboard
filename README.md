# stock-dashboard
# Stock Dashboard

This Streamlit app analyzes a stock's beta and volatility versus a benchmark. It also contains a **Portfolio Tracker** for monitoring an entire portfolio.

The tracker verifies each ticker and automatically fetches the price on the purchase date. Enter the number of shares and choose whether to track the holding in its native currency or convert values to USD. Holdings can be edited or removed and a chart displays the portfolio’s value over time.

## Installation

Run these commands in a fresh virtual environment to install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Multi-Stock Analysis

Select multiple tickers and a benchmark in the sidebar to compare their daily returns. The dashboard displays a correlation heatmap, betas and R² values, rolling correlations, and simple trend regressions so you can explore how groups of assets move together.
