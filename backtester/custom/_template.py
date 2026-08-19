"""TEMPLATE — copy this file, rename it (no leading underscore), and edit.

A custom strategy = a signal function + parameters, registered once. The signal
function receives the full OHLCV frame (Open/High/Low/Close/Volume, DatetimeIndex)
and your parameters, and returns a **long/flat position** as a pandas Series:

    * return 1.0 on bars you want to be LONG, 0.0 on bars you want to be FLAT, OR
    * return raw entry/exit booleans and wrap them with ``hold(entries, exits)``
      to "stay long until the exit fires".

Everything else — next-bar execution (no lookahead), transaction costs, the
sector/time/regime conditions, cadence, win/loss + expectancy stats and charts —
is handled by the engine. You only express the entry/exit logic here.

Files whose name starts with "_" (like this one) are NOT loaded, so this template
never registers anything.
"""
from __future__ import annotations

import pandas as pd

from backtester import indicators as ind          # sma, ema, rsi, macd, bollinger, atr
from backtester.strategies import Param, Strategy, hold, register


def signal(df: pd.DataFrame, lookback: int = 20, vol_mult: float = 1.5) -> pd.Series:
    """EXAMPLE logic — replace with your own rules.

    Long when today's close is the highest of the last ``lookback`` closes AND
    volume is at least ``vol_mult`` x its 20-day average; exit when price loses
    the ``lookback``-day low.
    """
    close, vol = df["Close"], df["Volume"]
    breakout = close >= close.rolling(int(lookback)).max()
    vol_ok = vol >= vol_mult * vol.rolling(20).mean()
    entries = breakout & vol_ok
    exits = close <= close.rolling(int(lookback)).min()
    return hold(entries, exits)


# Registering is what makes it show up in the app + CLI. Uncomment in your copy:
#
# register(Strategy(
#     key="my_strategy",                    # unique id (snake_case)
#     name="My Strategy",                   # shown in the UI
#     description="One line describing the edge.",
#     params=[
#         Param("lookback", "Lookback (bars)", "int", 20, 5, 200, 1),
#         Param("vol_mult", "Volume x avg", "float", 1.5, 1.0, 5.0, 0.1),
#     ],
#     signal_fn=signal,
#     # optional: overlay_fn=lambda df, **p: {"20d high": df["Close"].rolling(20).max()},
#     # optional: subplot_fn=lambda df, **p: ("Volume", {"Volume": df["Volume"]}),
# ))
