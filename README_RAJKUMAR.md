# Quick Start: One-Click Backtesting App

A simple Streamlit app built on vectorbt for an SMA crossover backtest.

How to run locally

1) Create venv (recommended):
```bash
python -m venv .venv && .venv\Scripts\activate
```

2) Install requirements:
```bash
pip install -r requirements-streamlit.txt
```

3) Start the app:
```bash
streamlit run streamlit_app.py
```

The app opens in your browser at http://localhost:8501

Usage
- Enter comma-separated symbols like `BTC-USD, ETH-USD`.
- Choose period and interval.
- Set Fast and Slow MA windows and click "Run Backtest".

Notes
- Built with vectorbt: https://github.com/polakowo/vectorbt
- Original candlestick patterns Dash demo remains under `apps/candlestick-patterns`.

Deploy
- Streamlit Cloud: deploy `streamlit_app.py` with `requirements-streamlit.txt`.
- Vercel: prefer Streamlit Cloud or a container; Streamlit is not first-class on Vercel.
