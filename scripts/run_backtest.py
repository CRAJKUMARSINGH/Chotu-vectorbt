import argparse
from typing import List

import pandas as pd
import vectorbt as vbt


def load_price(symbols: List[str], period: str, interval: str) -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(description="Run SMA crossover backtest (vectorbt)")
    parser.add_argument("--symbols", type=str, default="BTC-USD", help="Comma-separated symbols")
    parser.add_argument("--period", type=str, default="1y", help="Yahoo Finance period, e.g., 6mo,1y,5y,max")
    parser.add_argument("--interval", type=str, default="1d", help="Yahoo Finance interval, e.g., 1d,1wk,1mo")
    parser.add_argument("--fast", type=int, default=10, help="Fast MA window")
    parser.add_argument("--slow", type=int, default=50, help="Slow MA window")
    parser.add_argument("--out", type=str, default="", help="Optional path to save stats CSV")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.slow <= args.fast:
        raise SystemExit("--slow must be greater than --fast")

    price = load_price(symbols, args.period, args.interval)
    pf = run_strategy(price, args.fast, args.slow)
    stats = pf.stats()

    print("=== Summary (first 20 rows) ===")
    print(stats.head(20).to_string())

    if args.out:
        stats.to_csv(args.out)
        print(f"Saved stats to {args.out}")


if __name__ == "__main__":
    main()


