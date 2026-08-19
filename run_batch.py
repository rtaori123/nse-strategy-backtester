"""Headless universe scan: run a strategy across the Nifty 500 and rank the results.

Examples:
    python run_batch.py --strategy sma_crossover --fast 20 --slow 50
    python run_batch.py --strategy rsi_meanrev --period 3y --limit 100 --out rsi.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from backtester import data, engine, strategies, universe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest a strategy across the Nifty 500.")
    p.add_argument("--strategy", required=True,
                   help="Strategy key: " + ", ".join(s.key for s in strategies.list_strategies()))
    p.add_argument("--period", default="5y", help="History window (1y,2y,3y,5y,10y,max)")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost per side")
    p.add_argument("--limit", type=int, default=0, help="Max stocks to scan (0 = all)")
    p.add_argument("--refresh", action="store_true", help="Force re-download data")
    p.add_argument("--out", default="results.csv", help="Output CSV path")
    p.add_argument("--sort", default="Sharpe", help="Column to sort by")
    # Remaining --key value pairs are treated as strategy params.
    return p


def parse_strategy_params(strategy, extra: list[str]) -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    for prm in strategy.params:
        if prm.kind == "int":
            parser.add_argument(f"--{prm.name}", type=int)
        elif prm.kind == "float":
            parser.add_argument(f"--{prm.name}", type=float)
        else:
            parser.add_argument(f"--{prm.name}", type=str)
    ns, _ = parser.parse_known_args(extra)
    return {k: v for k, v in vars(ns).items() if v is not None}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    args, extra = build_parser().parse_known_args(argv)

    try:
        strategy = strategies.get_strategy(args.strategy)
    except KeyError as e:
        print(e)
        return 2
    params = parse_strategy_params(strategy, extra)

    symbols = universe.load_symbols()
    if args.limit > 0:
        symbols = symbols[: args.limit]

    print(f"Backtesting '{strategy.name}' on {len(symbols)} stocks "
          f"(period={args.period}, params={ {**strategy.defaults(), **params} })…")

    done_count = {"n": 0}
    def cb(done, total, sym):
        if done - done_count["n"] >= 25 or done == total:
            done_count["n"] = done
            print(f"  data {done}/{total}", end="\r", flush=True)

    ohlcv, failed = data.get_many(symbols, period=args.period,
                                  refresh=args.refresh, progress_cb=cb)
    print()

    records = []
    for sym, df in ohlcv.items():
        if len(df) < 30:
            continue
        res = engine.run(df, strategy, params, cost_bps=args.cost_bps, symbol=sym)
        m = res.metrics
        records.append({
            "Symbol": sym.replace(".NS", ""),
            "Return_%": round(m["total_return"] * 100, 2),
            "vs_BH_%": round((m["total_return"] - res.benchmark_metrics["total_return"]) * 100, 2),
            "CAGR_%": round(m["cagr"] * 100, 2),
            "Sharpe": round(m["sharpe"], 3),
            "MaxDD_%": round(m["max_drawdown"] * 100, 2),
            "Win_%": round(m["win_rate"] * 100, 1) if pd.notna(m["win_rate"]) else None,
            "Trades": m["num_trades"],
        })

    if not records:
        print("No results produced.")
        return 1

    table = pd.DataFrame(records)
    sort_col = args.sort if args.sort in table.columns else "Sharpe"
    table = table.sort_values(sort_col, ascending=False).reset_index(drop=True)
    table.to_csv(args.out, index=False)

    print(f"\nWrote {len(table)} rows to {args.out} "
          f"({len(failed)} symbols had no data). Top 10 by {sort_col}:\n")
    print(table.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
