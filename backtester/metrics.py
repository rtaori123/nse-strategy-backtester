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
        result.update(trade_stats(trades))
        result["exposure"] = float(trades["bars_held"].sum() / len(returns))
    else:
        result["num_trades"] = 0
        result["win_rate"] = float("nan")
        result["exposure"] = 0.0
        for k in ("avg_win", "avg_loss", "expectancy", "profit_factor",
                  "payoff_ratio", "avg_hold_bars"):
            result[k] = float("nan")
    return result


def trade_stats(trades: pd.DataFrame) -> dict:
    """Win/loss + expectancy stats from a trade log (``return_pct`` per trade)."""
    n = len(trades)
    if n == 0:
        return {"num_trades": 0, "win_rate": float("nan")}
    r = trades["return_pct"].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # <= 0
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "num_trades": int(n),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        # expected % return per trade — the number that actually matters
        "expectancy": float(win_rate * avg_win + (1 - win_rate) * avg_loss),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0
        else float("inf") if gross_win > 0 else 0.0,
        "payoff_ratio": float(avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf"),
        "avg_hold_bars": float(trades["bars_held"].mean()) if "bars_held" in trades else float("nan"),
    }


def pool_trades(trade_frames: list[pd.DataFrame]) -> dict:
    """Aggregate stats across many symbols' trade logs (the scorecard math)."""
    frames = [t for t in trade_frames if t is not None and not t.empty]
    if not frames:
        return {"num_trades": 0, "win_rate": float("nan"), "expectancy": float("nan"),
                "profit_factor": float("nan"), "avg_win": float("nan"),
                "avg_loss": float("nan"), "payoff_ratio": float("nan"),
                "avg_hold_bars": float("nan")}
    allt = pd.concat(frames, ignore_index=True)
    return trade_stats(allt)


def buy_and_hold(close: pd.Series) -> dict:
    """Benchmark metrics for a passive long position."""
    returns = close.pct_change().fillna(0.0)
    m = compute(returns, trades=None)
    m["num_trades"] = 1
    m["win_rate"] = 1.0 if m["total_return"] > 0 else 0.0
    m["exposure"] = 1.0
    return m
