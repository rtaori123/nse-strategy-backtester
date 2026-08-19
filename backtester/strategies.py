"""Parameterized strategy registry.

Each strategy produces a **position** series (1 = long, 0 = flat; long-only for
v1) from an OHLCV frame plus tunable params. Strategies also optionally expose
price-panel overlays and a lower indicator panel so the charts can be drawn
"on the basis of the strategy".

Add a new strategy by appending one ``Strategy`` to ``_REGISTRY``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Param:
    name: str
    label: str
    kind: str  # "int" | "float" | "choice"
    default: object
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None


@dataclass
class Strategy:
    key: str
    name: str
    description: str
    params: list[Param]
    signal_fn: Callable[..., pd.Series]
    overlay_fn: Callable[..., dict] | None = None  # series drawn on the price panel
    subplot_fn: Callable[..., dict] | None = None  # (title, {series}) lower panel

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def signals(self, df: pd.DataFrame, **params) -> pd.Series:
        merged = {**self.defaults(), **params}
        pos = self.signal_fn(df, **merged)
        return pos.reindex(df.index).fillna(0.0).clip(0, 1)

    def overlays(self, df: pd.DataFrame, **params) -> dict:
        if self.overlay_fn is None:
            return {}
        return self.overlay_fn(df, **{**self.defaults(), **params})

    def subplot(self, df: pd.DataFrame, **params):
        if self.subplot_fn is None:
            return None
        return self.subplot_fn(df, **{**self.defaults(), **params})


def _hold(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """Build a held 0/1 position: go long on an entry, stay long until an exit."""
    pos = pd.Series(np.nan, index=entries.index)
    pos[entries] = 1.0
    pos[exits & ~entries] = 0.0
    pos = pos.ffill().fillna(0.0)
    return pos


# --------------------------------------------------------------------------- #
# SMA / EMA crossover
# --------------------------------------------------------------------------- #
def _ma_cross(df, fast, slow, kind):
    ma = ind.sma if kind == "sma" else ind.ema
    f, s = ma(df["Close"], int(fast)), ma(df["Close"], int(slow))
    return (f > s).astype(float)


def _ma_overlay(df, fast, slow, kind):
    ma = ind.sma if kind == "sma" else ind.ema
    label = kind.upper()
    return {
        f"{label} {int(fast)}": ma(df["Close"], int(fast)),
        f"{label} {int(slow)}": ma(df["Close"], int(slow)),
    }


# --------------------------------------------------------------------------- #
# RSI mean-reversion
# --------------------------------------------------------------------------- #
def _rsi_signal(df, period, buy_below, sell_above):
    r = ind.rsi(df["Close"], int(period))
    return _hold(r < buy_below, r > sell_above)


def _rsi_subplot(df, period, buy_below, sell_above):
    r = ind.rsi(df["Close"], int(period))
    return ("RSI", {"RSI": r, "_hlines": [buy_below, sell_above]})


# --------------------------------------------------------------------------- #
# MACD crossover
# --------------------------------------------------------------------------- #
def _macd_signal(df, fast, slow, signal):
    m = ind.macd(df["Close"], int(fast), int(slow), int(signal))
    return (m["macd"] > m["signal"]).astype(float)


def _macd_subplot(df, fast, slow, signal):
    m = ind.macd(df["Close"], int(fast), int(slow), int(signal))
    return ("MACD", {"MACD": m["macd"], "Signal": m["signal"], "_hist": m["hist"]})


# --------------------------------------------------------------------------- #
# Bollinger Bands
# --------------------------------------------------------------------------- #
def _bollinger_signal(df, period, num_std, mode):
    b = ind.bollinger(df["Close"], int(period), float(num_std))
    close = df["Close"]
    if mode == "reversion":
        # Buy when price closes below lower band, exit at the mid line.
        return _hold(close < b["lower"], close >= b["mid"])
    # breakout: long while price is above the upper band.
    return (close > b["upper"]).astype(float)


def _bollinger_overlay(df, period, num_std, mode):
    b = ind.bollinger(df["Close"], int(period), float(num_std))
    return {
        f"BB mid ({int(period)})": b["mid"],
        "BB upper": b["upper"],
        "BB lower": b["lower"],
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Strategy] = {
    "sma_crossover": Strategy(
        key="sma_crossover",
        name="SMA Crossover",
        description="Long while the fast SMA is above the slow SMA.",
        params=[
            Param("fast", "Fast SMA", "int", 20, 2, 200, 1),
            Param("slow", "Slow SMA", "int", 50, 5, 400, 1),
        ],
        signal_fn=lambda df, fast, slow: _ma_cross(df, fast, slow, "sma"),
        overlay_fn=lambda df, fast, slow: _ma_overlay(df, fast, slow, "sma"),
    ),
    "ema_crossover": Strategy(
        key="ema_crossover",
        name="EMA Crossover",
        description="Long while the fast EMA is above the slow EMA.",
        params=[
            Param("fast", "Fast EMA", "int", 12, 2, 200, 1),
            Param("slow", "Slow EMA", "int", 26, 5, 400, 1),
        ],
        signal_fn=lambda df, fast, slow: _ma_cross(df, fast, slow, "ema"),
        overlay_fn=lambda df, fast, slow: _ma_overlay(df, fast, slow, "ema"),
    ),
    "rsi_meanrev": Strategy(
        key="rsi_meanrev",
        name="RSI Mean-Reversion",
        description="Buy when RSI is oversold, sell when it is overbought.",
        params=[
            Param("period", "RSI period", "int", 14, 2, 100, 1),
            Param("buy_below", "Buy below", "int", 30, 1, 50, 1),
            Param("sell_above", "Sell above", "int", 70, 50, 99, 1),
        ],
        signal_fn=_rsi_signal,
        subplot_fn=_rsi_subplot,
    ),
    "macd_crossover": Strategy(
        key="macd_crossover",
        name="MACD Crossover",
        description="Long while the MACD line is above its signal line.",
        params=[
            Param("fast", "Fast EMA", "int", 12, 2, 100, 1),
            Param("slow", "Slow EMA", "int", 26, 5, 200, 1),
            Param("signal", "Signal EMA", "int", 9, 2, 100, 1),
        ],
        signal_fn=_macd_signal,
        subplot_fn=_macd_subplot,
    ),
    "bollinger": Strategy(
        key="bollinger",
        name="Bollinger Bands",
        description="Breakout (long above upper band) or reversion (buy the lower band).",
        params=[
            Param("period", "Period", "int", 20, 5, 200, 1),
            Param("num_std", "Std devs", "float", 2.0, 0.5, 4.0, 0.1),
            Param("mode", "Mode", "choice", "breakout", choices=["breakout", "reversion"]),
        ],
        signal_fn=_bollinger_signal,
        overlay_fn=_bollinger_overlay,
    ),
}


def list_strategies() -> list[Strategy]:
    return list(_REGISTRY.values())


def get_strategy(key: str) -> Strategy:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{key}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[key]
