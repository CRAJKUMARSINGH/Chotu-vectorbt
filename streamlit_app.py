import os
import numpy as np
import pandas as pd
import streamlit as st
import vectorbt as vbt


st.set_page_config(page_title="VectorBT SMA Crossover", layout="wide")


@st.cache_data(show_spinner=False)
def load_price(symbols: list[str], period: str, interval: str) -> pd.DataFrame:


    data = vbt.YFData.download(
        symbols,
        period=period,
        interval=interval,
        missing_index="drop",
    ).get("Close")
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    return data


def run_strategy(price: pd.DataFrame, fast: int, slow: int) -> vbt.Portfolio:
    fast_ma = vbt.MA.run(price, window=fast)
    slow_ma = vbt.MA.run(price, window=slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10000.0)
    return pf


def main() -> None:
    st.title("One-Click SMA Crossover Backtest (vectorbt)")
    with st.sidebar:
        st.header("Data")
        symbols_str = st.text_input("Symbols (comma-separated)", value="BTC-USD, ETH-USD")
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "max"], index=1)
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

        st.header("Strategy")
        fast = st.number_input("Fast MA window", min_value=2, max_value=200, value=10, step=1)
        slow = st.number_input("Slow MA window", min_value=3, max_value=400, value=50, step=1)
        if slow <= fast:
            st.warning("Slow window should be greater than fast window.")

        st.header("Run")
        run_btn = st.button("Run Backtest", use_container_width=True)

    if run_btn:
        with st.status("Downloading data and running backtest...", expanded=False):
            price = load_price(symbols, period, interval)
            pf = run_strategy(price, fast, slow)

        st.subheader("Cumulative Return")
        st.plotly_chart(pf.total_return().vbt.heatmap(slider_level="symbol").figure, use_container_width=True)

        st.subheader("Portfolio Stats")
        stats = pf.stats()
        st.dataframe(stats.rename("Value"))

        st.subheader("Equity Curve")
        fig = pf.plot()
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trades Overview")
        st.dataframe(pf.trades.records)

    st.caption("Powered by vectorbt. See repo: https://github.com/polakowo/vectorbt")


if __name__ == "__main__":
    main()


