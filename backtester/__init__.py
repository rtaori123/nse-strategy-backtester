"""Indian-market stock backtesting toolkit.

Modules:
    universe    - load Nifty 500 tickers (bundled snapshot + NSE refresh)
    data        - fetch + cache OHLCV via yfinance
    indicators  - technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR)
    strategies  - parameterized strategy registry
    engine      - vectorized backtest engine
    metrics     - performance metrics
    charts      - Plotly figures for the Streamlit UI
"""

__version__ = "0.1.0"

# Configure TLS trust (corporate-proxy safe) before any HTTP client is imported.
from . import _ssl as _ssl  # noqa: E402,F401
