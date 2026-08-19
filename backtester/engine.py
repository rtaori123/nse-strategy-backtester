"""Vectorized long/flat backtest engine.

Signals are generated on bar *t* and executed on bar *t+1* (``position =
signal.shift(1)``) to avoid lookahead. Per-bar strategy returns are the held
position times the close-to-close return, minus a transaction cost charged
whenever the position changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as _metrics


@dataclass
class BacktestResult:
    symbol: str
    strategy_key: str
    params: dict
    ohlcv: pd.DataFrame
    position: pd.Series          # held position actually applied per bar (0/1)
    returns: pd.Series           # per-bar strategy returns (net of costs)
    equity: pd.Series            # strategy equity curve (starts at 1.0)
    benchmark_equity: pd.Series  # buy & hold equity curve (starts at 1.0)
    trades: pd.DataFrame         # entry/exit/return_pct/bars_held
    metrics: dict
    benchmark_metrics: dict


def _extract_trades(
    close: pd.Series, position: pd.Series, cost_bps: float
) -> pd.DataFrame:
    """Turn a held 0/1 position into a per-trade log."""
    pos = position.fillna(0.0).values
    idx = position.index
    price = close.values
    cost = cost_bps / 10000.0

    trades = []
    in_pos = False
    entry_i = 0
    for i in range(len(pos)):
        if not in_pos and pos[i] > 0:
            in_pos, entry_i = True, i
        elif in_pos and pos[i] == 0:
            entry_p, exit_p = price[entry_i], price[i]
            gross = exit_p / entry_p - 1.0
            net = gross - 2 * cost  # entry + exit
            trades.append(
                {
                    "entry_date": idx[entry_i], "exit_date": idx[i],
                    "entry_price": float(entry_p), "exit_price": float(exit_p),
                    "return_pct": float(net), "bars_held": int(i - entry_i),
                }
            )
            in_pos = False
    if in_pos:  # still open at the end of the sample
        entry_p, exit_p = price[entry_i], price[-1]
        gross = exit_p / entry_p - 1.0
        trades.append(
            {
                "entry_date": idx[entry_i], "exit_date": idx[-1],
                "entry_price": float(entry_p), "exit_price": float(exit_p),
                "return_pct": float(gross - cost), "bars_held": int(len(pos) - 1 - entry_i),
                "open": True,
            }
        )
    return pd.DataFrame(trades)


def run(
    df: pd.DataFrame,
    strategy,
    params: dict | None = None,
    cost_bps: float = 5.0,
    symbol: str = "",
) -> BacktestResult:
    """Run ``strategy`` over OHLCV ``df`` and return a :class:`BacktestResult`."""
    params = params or {}
    close = df["Close"].astype(float)

    signal = strategy.signals(df, **params)
    # Execute next bar to avoid lookahead.
    position = signal.shift(1).fillna(0.0)

    bar_ret = close.pct_change().fillna(0.0)
    gross = position * bar_ret

    # Charge cost on position changes (a round trip costs 2x on entry+exit bars).
    turns = position.diff().abs().fillna(position.abs())
    cost = turns * (cost_bps / 10000.0)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    benchmark_equity = (1.0 + bar_ret).cumprod()

    trades = _extract_trades(close, position, cost_bps)
    m = _metrics.compute(net, trades)
    bench = _metrics.buy_and_hold(close)

    return BacktestResult(
        symbol=symbol,
        strategy_key=strategy.key,
        params={**strategy.defaults(), **params},
        ohlcv=df,
        position=position,
        returns=net,
        equity=equity,
        benchmark_equity=benchmark_equity,
        trades=trades,
        metrics=m,
        benchmark_metrics=bench,
    )
