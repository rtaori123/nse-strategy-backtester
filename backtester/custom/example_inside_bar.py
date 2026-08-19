"""Worked example of a custom swing strategy: inside-bar breakout + volume surge.

Demonstrates the kind of bespoke candle/volume rule a discretionary trader uses
(not a standard indicator), expressed with raw OHLCV. Clone this file's shape for
your own rules.

Setup:
  * an "inside bar" = today's high < yesterday's high AND today's low > yesterday's
    low (a compression bar),
  * go long when the NEXT bar breaks above the inside bar's high on above-average
    volume,
  * exit when price closes back below that inside bar's low.
"""
from __future__ import annotations

import pandas as pd

from backtester.strategies import Param, Strategy, hold, register


def signal(df: pd.DataFrame, vol_mult: float = 1.2) -> pd.Series:
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]

    inside = (high < high.shift(1)) & (low > low.shift(1))
    # Reference levels come from the inside bar; only valid on the bar after it.
    ref_high = high.shift(1).where(inside.shift(1, fill_value=False))
    ref_low = low.shift(1).where(inside.shift(1, fill_value=False))
    ref_high = ref_high.ffill()
    ref_low = ref_low.ffill()

    vol_ok = vol >= vol_mult * vol.rolling(20).mean()
    entries = (close > ref_high) & vol_ok
    exits = close < ref_low
    return hold(entries, exits)


def overlay(df: pd.DataFrame, vol_mult: float = 1.2) -> dict:
    inside = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
    return {"Inside-bar high": df["High"].shift(1).where(inside.shift(1, fill_value=False)).ffill()}


register(Strategy(
    key="inside_bar_breakout",
    name="Inside-Bar Breakout (example)",
    description="Long on a volume breakout above an inside bar; exit below its low.",
    params=[
        Param("vol_mult", "Volume x 20d avg", "float", 1.2, 1.0, 5.0, 0.1),
    ],
    signal_fn=signal,
    overlay_fn=overlay,
))
