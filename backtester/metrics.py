"""Performance metrics computed from a per-bar return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def _annual_factor(index: pd.DatetimeIndex) -> float:
    """Estimate periods-per-year from the index spacing (daily -> ~252)."""
    if len(index) < 3:
        return _TRADING_DAYS
    days = np.median(np.diff(index.values).astype("timedelta64[D]").astype(float))
    if days <= 0:
        return _TRADING_DAYS
    return 365.25 / days


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def compute(returns: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    """Summarize a strategy from its per-bar returns (and optional trade log)."""
    returns = returns.fillna(0.0)
    if returns.empty:
        return {
            "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_drawdown": 0.0, "volatility": 0.0, "exposure": 0.0,
            "num_trades": 0, "win_rate": float("nan"),
        }

    equity = (1.0 + returns).cumprod()
    ann = _annual_factor(returns.index)

    total_return = float(equity.iloc[-1] - 1.0)
    years = len(returns) / ann
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0

    mean, std = returns.mean(), returns.std(ddof=0)
    sharpe = float(mean / std * np.sqrt(ann)) if std > 0 else 0.0
    downside = returns[returns < 0].std(ddof=0)
    sortino = float(mean / downside * np.sqrt(ann)) if downside and downside > 0 else 0.0
    volatility = float(std * np.sqrt(ann))

    result = {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(equity),
        "volatility": volatility,
    }

    if trades is not None and not trades.empty:
        wins = (trades["return_pct"] > 0).sum()
        result["num_trades"] = int(len(trades))
        result["win_rate"] = float(wins / len(trades))
        result["exposure"] = float(trades["bars_held"].sum() / len(returns))
    else:
        result["num_trades"] = 0
        result["win_rate"] = float("nan")
        result["exposure"] = 0.0
    return result


def buy_and_hold(close: pd.Series) -> dict:
    """Benchmark metrics for a passive long position."""
    returns = close.pct_change().fillna(0.0)
    m = compute(returns, trades=None)
    m["num_trades"] = 1
    m["win_rate"] = 1.0 if m["total_return"] > 0 else 0.0
    m["exposure"] = 1.0
    return m
