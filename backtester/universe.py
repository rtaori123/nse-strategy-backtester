"""Load the Nifty 500 ticker universe.

Ships a bundled snapshot (universe/nifty500.csv) so the app always works
offline. `refresh_universe()` pulls the current official list from NSE and
overwrites the snapshot; on any failure it leaves the snapshot untouched.
"""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_UNIVERSE_DIR = Path(__file__).resolve().parent.parent / "universe"
_SNAPSHOT = _UNIVERSE_DIR / "nifty500.csv"
_NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}


def load_rows() -> list[dict[str, str]]:
    """Return the bundled snapshot as a list of {Symbol, Company Name, Industry}."""
    if not _SNAPSHOT.exists():
        raise FileNotFoundError(
            f"Universe snapshot missing at {_SNAPSHOT}. Run refresh_universe()."
        )
    with _SNAPSHOT.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _field(row: dict, key: str) -> str:
    """Safe cell read: csv.DictReader yields None for missing cells."""
    return (row.get(key) or "").strip()


def load_symbols(suffix: str = ".NS") -> list[str]:
    """Return tickers ready for yfinance (NSE symbols get a ``.NS`` suffix)."""
    return [_field(r, "Symbol") + suffix for r in load_rows() if _field(r, "Symbol")]


def load_tickers() -> list[str]:
    """Alias for :func:`load_symbols` (yfinance-ready ``*.NS`` tickers)."""
    return load_symbols()


def symbol_to_name() -> dict[str, str]:
    """Map yfinance ticker (e.g. ``RELIANCE.NS``) -> company name."""
    return {_field(r, "Symbol") + ".NS": _field(r, "Company Name")
            for r in load_rows() if _field(r, "Symbol")}


def load_sectors() -> list[str]:
    """Sorted list of distinct sectors (the NSE macro-industry column)."""
    return sorted({_field(r, "Industry") for r in load_rows() if _field(r, "Industry")})


def symbol_to_sector(suffix: str = ".NS") -> dict[str, str]:
    """Map yfinance ticker -> sector name."""
    return {_field(r, "Symbol") + suffix: _field(r, "Industry")
            for r in load_rows() if _field(r, "Symbol")}


def symbols_for_sectors(sectors: list[str] | None, suffix: str = ".NS") -> list[str]:
    """Tickers belonging to any of ``sectors`` (all tickers if empty/None)."""
    if not sectors:
        return load_symbols(suffix)
    wanted = {s.strip() for s in sectors}
    return [
        _field(r, "Symbol") + suffix
        for r in load_rows()
        if _field(r, "Symbol") and _field(r, "Industry") in wanted
    ]


def refresh_universe(timeout: int = 20) -> int:
    """Fetch the current Nifty 500 list from NSE and overwrite the snapshot.

    Returns the number of symbols written. On failure the existing snapshot is
    kept and the exception is re-raised so callers can surface it.
    """
    import requests

    session = requests.Session()
    session.headers.update(_HEADERS)
    # Prime cookies via the homepage; NSE rejects cold archive requests.
    try:
        session.get("https://www.nseindia.com", timeout=timeout)
    except Exception:  # noqa: BLE001 - cookie priming is best-effort
        pass

    resp = session.get(_NSE_URL, timeout=timeout)
    resp.raise_for_status()

    rows = list(csv.DictReader(io.StringIO(resp.text)))
    if not rows:
        raise ValueError("NSE returned an empty constituent list")

    _UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    with _SNAPSHOT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Symbol", "Company Name", "Industry"])
        for r in rows:
            writer.writerow(
                [r["Symbol"].strip(), r["Company Name"].strip(), r.get("Industry", "").strip()]
            )
    logger.info("Refreshed Nifty 500 snapshot: %d symbols", len(rows))
    return len(rows)
