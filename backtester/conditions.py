"""Condition layer: gate any strategy signal by sector / time / regime, and
model a swing cadence (decision frequency + time-based exit + cooldown).

A base strategy produces a raw long/flat signal. ``Conditions.apply`` turns that
into the position actually held, by:
  1. forcing flat outside the allowed time window and market regime,
  2. only letting the position change on "decision" bars (e.g. weekly),
  3. closing any trade held longer than ``max_hold_bars`` (time exit),
  4. blocking re-entry for ``min_gap_bars`` after an exit (cooldown).

The engine still shifts the result by one bar, so there is no lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class RegimeSpec:
    """A market/stock regime gate. Trading is allowed only while it holds."""

    kind: str = "market_trend"          # "market_trend" | "stock_trend" | "vol"
    ma_window: int = 50
    direction: str = "above"            # "above" | "below" (price vs its MA)
    source: str = "market"              # "market" (benchmark) | "stock"
    vol_window: int = 14                # for kind == "vol"
    vol_max: float | None = None        # allow only when ATR% <= vol_max (e.g. 0.03)

    def mask(self, df: pd.DataFrame, benchmark_df: pd.DataFrame | None) -> pd.Series:
        idx = df.index
        if self.kind == "vol":
            atr = ind.atr(df, self.vol_window)
            atr_pct = atr / df["Close"]
            if self.vol_max is None:
                return pd.Series(True, index=idx)
            return (atr_pct <= self.vol_max).reindex(idx).fillna(False)

        # trend gates: price of the chosen source vs its moving average
        if self.source == "market" and benchmark_df is not None and not benchmark_df.empty:
            close = benchmark_df["Close"].reindex(idx).ffill()
        else:
            close = df["Close"]
        ma = ind.sma(close, self.ma_window)
        gate = close > ma if self.direction == "above" else close < ma
        return gate.reindex(idx).fillna(False)


@dataclass
class Conditions:
    """Context filters + cadence applied on top of a raw strategy signal."""

    sectors: list[str] | None = None          # universe filter (scan level)
    symbols: list[str] | None = None          # explicit ticker override (scan level)
    months: list[int] | None = None           # allowed calendar months, 1-12
    weekdays: list[int] | None = None          # decision weekdays, 0=Mon .. 4=Fri
    date_start: date | None = None
    date_end: date | None = None
    regime: RegimeSpec | None = None
    decision_every_n_bars: int = 1             # act only every N bars
    max_hold_bars: int | None = None           # time-based exit
    min_gap_bars: int = 0                       # cooldown between trades

    # ---- masks -------------------------------------------------------------
    def time_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        mask = pd.Series(True, index=index)
        if self.months:
            mask &= index.month.isin(self.months)
        if self.date_start is not None:
            mask &= index >= pd.Timestamp(self.date_start)
        if self.date_end is not None:
            mask &= index <= pd.Timestamp(self.date_end)
        return mask

    def regime_mask(self, df, benchmark_df) -> pd.Series:
        if self.regime is None:
            return pd.Series(True, index=df.index)
        return self.regime.mask(df, benchmark_df).astype(bool)

    def gate_series(self, df, benchmark_df=None) -> pd.Series:
        """Combined allow/deny mask (time ∧ regime), for gating and chart shading."""
        return self.time_mask(df.index) & self.regime_mask(df, benchmark_df)

    def decision_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """Bars on which the held position is allowed to change."""
        if self.weekdays:
            return pd.Series(index.weekday.isin(self.weekdays), index=index)
        n = max(1, int(self.decision_every_n_bars))
        if n == 1:
            return pd.Series(True, index=index)
        flags = np.zeros(len(index), dtype=bool)
        flags[::n] = True
        return pd.Series(flags, index=index)

    # ---- main --------------------------------------------------------------
    def apply(self, raw_signal: pd.Series, df, benchmark_df=None) -> pd.Series:
        idx = df.index
        raw = raw_signal.reindex(idx).fillna(0.0).clip(0, 1)

        # 1. gate: forced flat outside the allowed window/regime
        gate = self.gate_series(df, benchmark_df)
        desired = raw.where(gate, 0.0)

        # 2. position may only change on decision bars; carry between them
        dmask = self.decision_mask(idx).values
        des = desired.values
        held = np.zeros(len(des))
        cur = 0.0
        for i in range(len(des)):
            if dmask[i]:
                cur = des[i]
            held[i] = cur
        pos = held

        # 3. time-based exit + 4. cooldown (single stateful pass)
        max_hold = self.max_hold_bars
        gap = max(0, int(self.min_gap_bars))
        if max_hold or gap:
            out = np.zeros(len(pos))
            in_pos = False
            bars_held = 0
            cooldown = 0
            for i in range(len(pos)):
                want = pos[i] > 0
                if in_pos:
                    bars_held += 1
                    if max_hold and bars_held >= max_hold:
                        out[i] = 1.0        # last bar of the capped trade
                        in_pos = False
                        # force >=1 flat bar so the trade actually closes
                        # (otherwise an immediate re-entry merges the two runs)
                        cooldown = max(gap, 1)
                        continue
                    if not want:            # normal exit from gate/signal
                        in_pos = False
                        cooldown = gap
                        continue
                    out[i] = 1.0
                else:
                    if cooldown > 0:
                        cooldown -= 1
                        continue
                    if want:
                        in_pos = True
                        bars_held = 1
                        out[i] = 1.0
            pos = out

        return pd.Series(pos, index=idx)


def no_conditions() -> Conditions:
    """An identity condition set (no gating, daily decisions)."""
    return Conditions()
