"""Streamlit UI for the Indian-market stock backtester.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backtester import charts, data, engine, metrics, strategies, universe
from backtester.conditions import Conditions, RegimeSpec

st.set_page_config(page_title="NSE Strategy Backtester", page_icon="📈", layout="wide")

PERIODS = ["1y", "2y", "3y", "5y", "10y", "max"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


@st.cache_data(show_spinner=False)
def load_benchmark(period: str):
    return data.get_benchmark(period)


# --------------------------------------------------------------------------- #
# Sidebar: conditions (sector / time / regime / cadence)
# --------------------------------------------------------------------------- #
def render_conditions() -> Conditions:
    st.sidebar.header("Conditions (optional)")
    st.sidebar.caption("Gate the strategy by sector, time and market regime — "
                       "and set the swing cadence.")

    with st.sidebar.expander("Sector / stock filter"):
        sectors = st.multiselect("Sectors", universe.load_sectors(),
                                 help="Applies to the universe scan.")
        manual = st.text_input("Or specific symbols (comma-separated)",
                               help="e.g. SUNPHARMA, CIPLA — overrides sectors.")
        symbols = [s.strip().upper() for s in manual.split(",") if s.strip()] or None

    with st.sidebar.expander("Time window"):
        mth_names = st.multiselect("Only these months", MONTHS)
        months = [MONTHS.index(m) + 1 for m in mth_names] or None
        use_dates = st.checkbox("Limit to a date range")
        date_start = date_end = None
        if use_dates:
            date_start = st.date_input("From", value=None)
            date_end = st.date_input("To", value=None)

    with st.sidebar.expander("Market regime"):
        use_regime = st.checkbox("Trade only when the market trend agrees")
        regime = None
        if use_regime:
            ma = st.slider("Nifty MA window (bars)", 10, 200, 50, 5)
            direction = st.radio("Trade when Nifty is", ["above", "below"],
                                 horizontal=True)
            regime = RegimeSpec(kind="market_trend", ma_window=ma,
                                direction=direction, source="market")

    with st.sidebar.expander("Cadence (swing)"):
        wd_names = st.multiselect("Act only on weekdays", WEEKDAYS,
                                  help="Leave empty to use 'every N bars' below.")
        weekdays = [WEEKDAYS.index(w) for w in wd_names] or None
        every = st.number_input("…or review every N bars", 1, 20, 1)
        max_hold = st.number_input("Time exit: close after H bars (0 = off)", 0, 60, 0)
        gap = st.number_input("Cooldown after a trade (bars)", 0, 30, 0)

    return Conditions(
        sectors=sectors or None, symbols=symbols, months=months,
        weekdays=weekdays, date_start=date_start, date_end=date_end,
        regime=regime, decision_every_n_bars=int(every),
        max_hold_bars=int(max_hold) or None, min_gap_bars=int(gap),
    )


def conditions_active(c: Conditions) -> bool:
    return any([c.sectors, c.symbols, c.months, c.weekdays, c.date_start, c.date_end,
                c.regime, c.decision_every_n_bars > 1, c.max_hold_bars, c.min_gap_bars])


# --------------------------------------------------------------------------- #
# Sidebar: strategy + params (auto-rendered from the param spec)
# --------------------------------------------------------------------------- #
def render_params(strategy) -> dict:
    values = {}
    for p in strategy.params:
        key = f"param_{strategy.key}_{p.name}"
        if p.kind == "int":
            values[p.name] = st.sidebar.slider(
                p.label, int(p.min), int(p.max), int(p.default),
                step=int(p.step or 1), key=key,
            )
        elif p.kind == "float":
            values[p.name] = st.sidebar.slider(
                p.label, float(p.min), float(p.max), float(p.default),
                step=float(p.step or 0.1), key=key,
            )
        elif p.kind == "choice":
            values[p.name] = st.sidebar.selectbox(
                p.label, p.choices, index=p.choices.index(p.default), key=key,
            )
    return values


@st.cache_data(show_spinner=False)
def load_universe_rows():
    return universe.load_rows()


def metrics_table(result) -> pd.DataFrame:
    m, b = result.metrics, result.benchmark_metrics
    def pct(x):
        return f"{x*100:.1f}%" if pd.notna(x) else "—"
    rows = {
        "Total return": [pct(m["total_return"]), pct(b["total_return"])],
        "CAGR": [pct(m["cagr"]), pct(b["cagr"])],
        "Sharpe": [f"{m['sharpe']:.2f}", f"{b['sharpe']:.2f}"],
        "Sortino": [f"{m['sortino']:.2f}", f"{b['sortino']:.2f}"],
        "Max drawdown": [pct(m["max_drawdown"]), pct(b["max_drawdown"])],
        "Volatility": [pct(m["volatility"]), pct(b["volatility"])],
        "Win rate": [pct(m["win_rate"]), "—"],
        "Expectancy / trade": [pct(m.get("expectancy", float("nan"))), "—"],
        "Profit factor": [f"{m.get('profit_factor', float('nan')):.2f}", "—"],
        "Avg win / Avg loss": [
            f"{pct(m.get('avg_win', float('nan')))} / {pct(m.get('avg_loss', float('nan')))}", "—"],
        "# Trades": [m["num_trades"], "—"],
        "Exposure": [pct(m["exposure"]), pct(b["exposure"])],
    }
    return pd.DataFrame(rows, index=["Strategy", "Buy & Hold"]).T


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def single_stock_view(strategy, params, period, cost_bps, refresh, conditions, benchmark_df):
    rows = load_universe_rows()
    names = {r["Symbol"].strip(): r["Company Name"].strip() for r in rows}
    options = [r["Symbol"].strip() for r in rows]

    default_ix = options.index("RELIANCE") if "RELIANCE" in options else 0
    sym = st.selectbox(
        "Stock", options, index=default_ix,
        format_func=lambda s: f"{s} — {names.get(s, '')}",
    )
    show_candles = st.checkbox("Candlesticks", value=True)

    ticker = sym + ".NS"
    with st.spinner(f"Loading {ticker}…"):
        df = data.get_ohlcv(ticker, period=period, refresh=refresh)

    if df.empty or len(df) < 30:
        st.error(f"No usable data for {ticker}.")
        return

    result = engine.run(df, strategy, params, cost_bps=cost_bps, symbol=ticker,
                        conditions=conditions, benchmark_df=benchmark_df)

    m = result.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{m['total_return']*100:.1f}%",
              f"{(m['total_return']-result.benchmark_metrics['total_return'])*100:.1f}% vs B&H")
    c2.metric("Expectancy / trade",
              f"{m.get('expectancy', float('nan'))*100:.2f}%" if pd.notna(m.get("expectancy")) else "—")
    c3.metric("Sharpe", f"{m['sharpe']:.2f}")
    c4.metric("Max drawdown", f"{m['max_drawdown']*100:.1f}%")

    if conditions is not None and result.gate is not None:
        st.caption("Grey bands = strategy was **not allowed to trade** "
                   "(outside its time window / regime).")
    st.plotly_chart(charts.price_chart(result, strategy, show_candles, gate=result.gate),
                    use_container_width=True)
    a, b = st.columns([3, 2])
    a.plotly_chart(charts.equity_chart(result), use_container_width=True)
    b.plotly_chart(charts.drawdown_chart(result), use_container_width=True)

    st.subheader("Performance vs Buy & Hold")
    st.dataframe(metrics_table(result), use_container_width=True)

    if not result.trades.empty:
        tl = result.trades.copy()
        n = len(tl)
        wins = int((tl["return_pct"] > 0).sum())
        show = pd.DataFrame({
            "Result": tl["return_pct"].apply(lambda x: "✅ Win" if x > 0 else "❌ Loss"),
            "Entry date": pd.to_datetime(tl["entry_date"]).dt.date,
            "Entry ₹": tl["entry_price"].round(1),
            "Exit date": pd.to_datetime(tl["exit_date"]).dt.date,
            "Exit ₹": tl["exit_price"].round(1),
            "Return %": (tl["return_pct"] * 100).round(1),
            "Held (bars)": tl["bars_held"],
            "Open?": tl["open"] if "open" in tl.columns else False,
        })

        def _highlight(row):
            color = "rgba(22,163,74,0.12)" if row["Return %"] > 0 else "rgba(220,38,38,0.12)"
            return [f"background-color: {color}"] * len(row)

        with st.expander(
            f"Trade log — {wins}/{n} winners ({wins/n*100:.0f}% win rate)", expanded=True
        ):
            st.dataframe(
                show.style.apply(_highlight, axis=1),
                use_container_width=True, hide_index=True,
            )


def _scorecard(pool: dict, n_symbols: int, pct_profitable: float):
    """The honest 'chances of winning' panel — pooled across all trades."""
    st.subheader("Strategy scorecard (pooled across all trades)")
    n = pool["num_trades"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total trades", f"{n}")
    c2.metric("Win rate", f"{pool['win_rate']*100:.0f}%" if pd.notna(pool["win_rate"]) else "—")
    c3.metric("Expectancy / trade",
              f"{pool['expectancy']*100:.2f}%" if pd.notna(pool["expectancy"]) else "—")
    pf = pool["profit_factor"]
    c4.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}" if pd.notna(pf) else "—")
    c5.metric("Stocks profitable", f"{pct_profitable*100:.0f}%")

    if n < 30:
        st.warning(f"⚠️ Only **{n} trades** — too few to trust. With small samples a "
                   "good-looking result is very likely luck. Widen the universe/window or "
                   "loosen the conditions before drawing conclusions.")
    exp = pool["expectancy"]
    if pd.notna(exp):
        if exp > 0:
            st.info(f"Positive expectancy (**+{exp*100:.2f}% per trade** before slippage). "
                    "Encouraging — but in-sample; see the caveat below.")
        else:
            st.info(f"**Negative expectancy** ({exp*100:.2f}% per trade). As backtested here, "
                    "this conditional rule loses money after costs.")


def universe_view(strategy, params, period, cost_bps, refresh, conditions, benchmark_df):
    rows = load_universe_rows()
    names = {r["Symbol"].strip() + ".NS": r["Company Name"].strip() for r in rows}

    # Universe = manual symbols > sector filter > everything.
    if conditions.symbols:
        pool_syms = [s if s.endswith(".NS") else s + ".NS" for s in conditions.symbols]
    else:
        pool_syms = universe.symbols_for_sectors(conditions.sectors)
    st.caption(f"Universe: **{len(pool_syms)}** stocks"
               + (f" in {', '.join(conditions.sectors)}" if conditions.sectors else "")
               + ". First scan downloads data (slow); later scans read the cache.")

    limit = st.slider("How many stocks to scan", 10, max(10, len(pool_syms)),
                      min(len(pool_syms), 100), step=10)

    if not st.button("Run scan", type="primary"):
        st.info("Set your strategy + conditions in the sidebar, then run the scan.")
        return

    symbols = pool_syms[:limit]
    prog = st.progress(0.0, text="Fetching data…")

    def cb(done, total, sym):
        prog.progress(min(done / total, 1.0), text=f"Fetching {done}/{total}: {sym}")

    ohlcv, failed = data.get_many(symbols, period=period, refresh=refresh, progress_cb=cb)
    prog.progress(1.0, text="Backtesting…")

    records, trade_frames, n_profitable = [], [], 0
    for sym, df in ohlcv.items():
        if len(df) < 30:
            continue
        res = engine.run(df, strategy, params, cost_bps=cost_bps, symbol=sym,
                         conditions=conditions, benchmark_df=benchmark_df)
        m = res.metrics
        trade_frames.append(res.trades)
        if m["total_return"] > 0:
            n_profitable += 1
        pf = m.get("profit_factor", float("nan"))
        records.append({
            "Symbol": sym.replace(".NS", ""),
            "Company": names.get(sym, ""),
            "Return %": round(m["total_return"] * 100, 1),
            "vs B&H %": round((m["total_return"] - res.benchmark_metrics["total_return"]) * 100, 1),
            "Expectancy %": round(m.get("expectancy", float("nan")) * 100, 2),
            "Profit factor": None if pf in (float("inf"), float("nan")) else round(pf, 2),
            "Sharpe": round(m["sharpe"], 2),
            "MaxDD %": round(m["max_drawdown"] * 100, 1),
            "Win %": round(m["win_rate"] * 100, 1) if pd.notna(m["win_rate"]) else None,
            "Trades": m["num_trades"],
        })
    prog.empty()

    if not records:
        st.error("No results — data failed to download, or no stock met the conditions.")
        return

    pool = metrics.pool_trades(trade_frames)
    _scorecard(pool, len(records), n_profitable / len(records))

    st.markdown(
        "> ⚠️ **In-sample results.** These show how the rule *would have* done on history it "
        "was selected against — they **overstate** live odds, especially with few trades. A "
        "walk-forward / out-of-sample test (next phase) is what separates a real edge from an "
        "overfit one."
    )

    table = pd.DataFrame(records).sort_values("Expectancy %", ascending=False).reset_index(drop=True)
    st.success(f"Scanned {len(table)} stocks"
               + (f" — {len(failed)} failed to load." if failed else "."))
    st.dataframe(table, use_container_width=True, height=460)
    st.download_button(
        "Download results CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{strategy.key}_scan.csv", mime="text/csv",
    )
    if failed:
        with st.expander(f"{len(failed)} symbols with no data"):
            st.write(", ".join(s.replace(".NS", "") for s in failed))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("📈 NSE Strategy Backtester")
    st.caption("Define a strategy, backtest it across the Nifty 500, and chart the results.")

    st.sidebar.header("Strategy")
    strat_list = strategies.list_strategies()
    labels = {s.name: s for s in strat_list}
    chosen = st.sidebar.selectbox("Strategy", list(labels))
    strategy = labels[chosen]
    st.sidebar.caption(strategy.description)

    params = render_params(strategy)

    st.sidebar.header("Backtest settings")
    period = st.sidebar.selectbox("History window", PERIODS, index=PERIODS.index("5y"))
    cost_bps = st.sidebar.slider("Transaction cost (bps per side)", 0, 50, 5, 1)
    refresh = st.sidebar.checkbox("Force refresh data", value=False)

    conditions = render_conditions()
    benchmark_df = load_benchmark(period) if conditions.regime is not None else None

    if not conditions_active(conditions):
        conditions = None  # plain backtest when nothing is set

    tab1, tab2 = st.tabs(["Single stock", "Universe scan"])
    with tab1:
        single_stock_view(strategy, params, period, cost_bps, refresh, conditions, benchmark_df)
    with tab2:
        if conditions is None:
            st.info("No conditions set — scanning the full Nifty 500. Add sector/time/regime "
                    "filters in the sidebar to target the strategy.")
            universe_view(strategy, params, period, cost_bps, refresh,
                          Conditions(), benchmark_df)
        else:
            universe_view(strategy, params, period, cost_bps, refresh, conditions, benchmark_df)


if __name__ == "__main__":
    main()
