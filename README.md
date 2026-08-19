# NSE Strategy Backtester

Define a trading strategy, pull Indian-market (NSE) stock data, backtest it
across the **Nifty 500**, and chart the results — price with buy/sell markers,
equity curve vs buy & hold, and drawdown.

## Setup

```bash
cd stock-backtester
python -m pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Opens a local dashboard:

- **Single stock** — pick a Nifty 500 name, tune the strategy in the sidebar,
  and see the strategy chart, equity curve, drawdown, and metrics vs buy & hold.
- **Universe scan** — run the chosen strategy across the Nifty 500, get a ranked
  table (by Sharpe / return) and a downloadable CSV.

The **first** data pull downloads OHLCV from Yahoo Finance and caches it to
`data/*.parquet`; later runs read the cache and are fast. Tick "Force refresh
data" (or use `--refresh`) to re-download.

## Headless / batch

```bash
python run_batch.py --strategy sma_crossover --fast 20 --slow 50 --period 5y --out results.csv
python run_batch.py --strategy rsi_meanrev --period 3y --limit 100
```

## Strategies (built-in, parameterized)

| Key              | What it does                                         |
|------------------|------------------------------------------------------|
| `sma_crossover`  | Long while fast SMA > slow SMA                       |
| `ema_crossover`  | Long while fast EMA > slow EMA                       |
| `rsi_meanrev`    | Buy oversold RSI, sell overbought                    |
| `macd_crossover` | Long while MACD line > signal line                   |
| `bollinger`      | Breakout (above upper band) or reversion (lower band)|

**Add a built-in strategy:** append one `Strategy` entry to `_REGISTRY` in
`backtester/strategies.py`.

**Add your own (custom) strategy:** drop a file in `backtester/custom/` — copy
`_template.py`, write your entry/exit rule with full OHLCV (incl. Volume), and
call `register(Strategy(...))`. It auto-appears in the app + CLI. See
`custom/example_inside_bar.py` for a working candle/volume example.

## Conditions — sector / time / regime / cadence

Any strategy can be **gated** so it only trades in the right context (set these
in the sidebar "Conditions", or via CLI flags):

- **Sector / symbols** — restrict the universe (e.g. only *Healthcare*, or a
  manual watchlist).
- **Time window** — only certain months (seasonality) or a date range.
- **Market regime** — trade only when the Nifty is above/below its N-day MA.
- **Swing cadence** — review/act only every N bars (or chosen weekdays),
  **time-exit** after H bars, and a cooldown between trades.

The universe scan then shows a **Strategy Scorecard** — pooled win rate,
**expectancy per trade**, profit factor, % of stocks profitable — with a
**sample-size warning** when there are too few trades to trust.

> ⚠️ These are **in-sample** results and *overstate* live odds, especially with
> few trades. Out-of-sample / walk-forward validation is the honest next step.

CLI example:
```bash
python run_batch.py --strategy inside_bar_breakout --sectors Healthcare \
    --months 3 --max-hold 5 --every 2 --regime-ma 50 --period 5y
```

## How it works (no lookahead)

Signals are generated on bar *t* and executed on bar *t+1*
(`position = signal.shift(1)`), returns are position × close-to-close change
minus a transaction cost on every position change. Conditions gate the raw
signal *before* that shift, so the no-lookahead guarantee holds. Long/flat only.

## Data notes

- Universe = bundled snapshot `universe/nifty500.csv`; refresh from NSE with
  `python -c "from backtester.universe import refresh_universe; refresh_universe()"`.
- Data source = `yfinance` (`.NS` symbols). Free, daily, no API key.
- Backtests are historical simulations, **not** investment advice.
```
