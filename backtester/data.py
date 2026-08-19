"""Fetch and cache OHLCV data via yfinance.

Each symbol is cached to ``data/{symbol}_{interval}.parquet``. Cached files are
reused unless ``refresh=True`` or the cache is older than ``max_age_days``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return _DATA_DIR / f"{safe}_{interval}.parquet"


def _cache_fresh(path: Path, max_age_days: float) -> bool:
    if not path.exists():
        return False
    if max_age_days is None:
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days <= max_age_days


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return a clean single-symbol OHLCV frame with a DatetimeIndex."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)

    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the level that carries OHLCV field names.
        lvl0 = set(df.columns.get_level_values(0))
        if {"Open", "Close"} <= lvl0:
            df = df.xs(key=df.columns.get_level_values(1)[0], axis=1, level=1)
        else:
            df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={c: c.title() for c in df.columns})
    keep = [c for c in _OHLCV_COLS if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=[c for c in ["Close"] if c in df.columns])
    return df


def get_ohlcv(
    symbol: str,
    period: str = "5y",
    interval: str = "1d",
    refresh: bool = False,
    max_age_days: float = 1.0,
) -> pd.DataFrame:
    """Return OHLCV for ``symbol`` (e.g. ``RELIANCE.NS``), using the disk cache.

    An empty DataFrame is returned for delisted/unknown symbols rather than
    raising, so batch runs can skip and continue.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, interval)

    if not refresh and _cache_fresh(path, max_age_days):
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - corrupt cache -> refetch
            logger.warning("Corrupt cache for %s, refetching", symbol)

    import yfinance as yf

    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=_OHLCV_COLS)

    df = _normalize(raw, symbol)
    if not df.empty:
        try:
            df.to_parquet(path)
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            logger.warning("Could not cache %s: %s", symbol, exc)
    return df


def get_benchmark(
    period: str = "5y", interval: str = "1d", refresh: bool = False
) -> pd.DataFrame:
    """OHLCV for the market benchmark used by regime gating.

    Tries the Nifty 50 index (``^NSEI``); falls back to Nifty 500 (``^CRSLDX``).
    Returns an empty frame if both fail (regime gate then allows everything).
    """
    for sym in ("^NSEI", "^CRSLDX"):
        df = get_ohlcv(sym, period=period, interval=interval, refresh=refresh)
        if not df.empty:
            return df
    return pd.DataFrame(columns=_OHLCV_COLS)


def get_many(
    symbols: list[str],
    period: str = "5y",
    interval: str = "1d",
    refresh: bool = False,
    max_age_days: float = 1.0,
    batch_size: int = 50,
    progress_cb=None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch many symbols, using cache where possible.

    Returns ``(data, failed)`` where ``data`` maps symbol -> OHLCV frame (only
    non-empty frames) and ``failed`` lists symbols with no usable data.

    ``progress_cb(done, total, symbol)`` is called after each symbol if given.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    # First pass: serve everything we can from a fresh cache.
    to_fetch: list[str] = []
    for sym in symbols:
        path = _cache_path(sym, interval)
        if not refresh and _cache_fresh(path, max_age_days):
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    data[sym] = df
                    continue
            except Exception:  # noqa: BLE001
                pass
        to_fetch.append(sym)

    total = len(symbols)
    done = len(data)
    if progress_cb:
        for sym in data:
            progress_cb(done, total, sym)

    # Second pass: batch-download the rest.
    import yfinance as yf

    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch download failed (%s...): %s", batch[0], exc)
            raw = None

        for sym in batch:
            df = pd.DataFrame()
            if raw is not None and isinstance(raw.columns, pd.MultiIndex):
                if sym in raw.columns.get_level_values(0):
                    df = _normalize(raw[sym], sym)
            elif raw is not None and len(batch) == 1:
                df = _normalize(raw, sym)

            # Fallback to a per-symbol fetch if the batch missed it.
            if df.empty:
                df = get_ohlcv(
                    sym, period=period, interval=interval,
                    refresh=refresh, max_age_days=max_age_days,
                )

            if df.empty:
                failed.append(sym)
            else:
                data[sym] = df
                try:
                    df.to_parquet(_cache_path(sym, interval))
                except Exception:  # noqa: BLE001
                    pass

            done += 1
            if progress_cb:
                progress_cb(done, total, sym)

    return data, failed
