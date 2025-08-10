# Stock Report Generator — Alpha Vantage Edition

Uses the free Alpha Vantage API so it deploys cleanly on Streamlit Cloud.

What it does
- Natural language screen for US Healthcare or US Tech, ranked by outperformance vs SPY.
- 2 page PDF per ticker with period return, outperformance vs SPY, and volatility proxy.
- Single-ticker PDF.
- Transcript paste -> simple summary.

Setup on Streamlit Cloud
1) Upload these files to a GitHub repo.
2) In Streamlit, set app file to app.py.
3) Add your Alpha Vantage key in Secrets as:
ALPHA_VANTAGE_API_KEY = "sk_from_alpha_vantage"
4) Deploy.

Notes
- Free tier is ~5 calls/min. Keep sector lists small.
- Benchmarks use SPY.