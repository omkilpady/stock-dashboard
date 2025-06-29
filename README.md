# stock-dashboard
# Stock Dashboard

This Streamlit app analyzes a stock's beta and volatility versus a benchmark. It also contains a **Portfolio Tracker** for monitoring an entire portfolio.

The tracker verifies each ticker and automatically fetches the price on the purchase date. Enter the number of shares and choose whether to track the holding in its native currency or convert values to USD. Holdings can be edited or removed and a chart displays the portfolio’s value over time.

## Development

Run the unit tests with:

```bash
pytest -q
```

Tests are also executed automatically in GitHub Actions.
