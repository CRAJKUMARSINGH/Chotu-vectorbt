import os
import numpy as np
import pandas as pd
import streamlit as st
import vectorbt as vbt


st.set_page_config(page_title="VectorBT SMA Crossover", layout="wide")


@st.cache_data(show_spinner=False)
def load_price(symbols: list[str], period: str, interval: str) -> pd.DataFrame:
    """Download and cache only the Close price frame."""
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
    """Build portfolio object for given parameters (not cached to avoid heavy object caching)."""
    fast_ma = vbt.MA.run(price, window=fast)
    slow_ma = vbt.MA.run(price, window=slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10000.0)
    return pf


@st.cache_data(show_spinner=False)
def compute_outputs(symbols_key: str, period: str, interval: str, fast: int, slow: int):
    """Cache only lightweight/serializable outputs keyed by parameters.

    symbols_key should be a stable string (e.g., 'BTC-USD,ETH-USD').
    Returns: dict(stats), total_return DataFrame, equity curve Series, trades head DataFrame
    """
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    price = load_price(symbols, period, interval)
    pf = run_strategy(price, fast, slow)
    stats_df = pf.stats()
    total_return_df = pf.total_return().to_frame()
    equity_curve = pf.value()
    trades_head = pf.trades.records.head(200)
    return (
        stats_df.to_dict(),
        total_return_df,
        equity_curve,
        trades_head,
    )


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
        if st.button("Clear cache"):
            st.cache_data.clear()

    if run_btn:
        with st.status("Running backtest...", expanded=False):
            stats_dict, total_return_df, equity_curve, trades_head = compute_outputs(
                ", ".join(symbols), period, interval, fast, slow
            )

        st.subheader("Cumulative Return")
        try:
            fig_heat = total_return_df.vbt.heatmap(slider_level="symbol").figure
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception:
            st.line_chart(total_return_df)

        st.subheader("Portfolio Stats")
        stats_df = pd.Series(stats_dict).to_frame(name="Value") if isinstance(stats_dict, dict) else pd.DataFrame(stats_dict)
        st.dataframe(stats_df)

        st.subheader("Equity Curve")
        st.line_chart(equity_curve)

        st.subheader("Trades Overview (first 200)")
        st.dataframe(trades_head)

    st.caption("Powered by vectorbt. See repo: https://github.com/polakowo/vectorbt")


if __name__ == "__main__":
    main()


