"""Streamlit UI for the Indian-market stock backtester.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backtester import charts, data, engine, metrics, strategies, universe

st.set_page_config(page_title="NSE Strategy Backtester", page_icon="📈", layout="wide")

PERIODS = ["1y", "2y", "3y", "5y", "10y", "max"]


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
        "# Trades": [m["num_trades"], "—"],
        "Exposure": [pct(m["exposure"]), pct(b["exposure"])],
    }
    return pd.DataFrame(rows, index=["Strategy", "Buy & Hold"]).T


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def single_stock_view(strategy, params, period, cost_bps, refresh):
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

    result = engine.run(df, strategy, params, cost_bps=cost_bps, symbol=ticker)

    m = result.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{m['total_return']*100:.1f}%",
              f"{(m['total_return']-result.benchmark_metrics['total_return'])*100:.1f}% vs B&H")
    c2.metric("CAGR", f"{m['cagr']*100:.1f}%")
    c3.metric("Sharpe", f"{m['sharpe']:.2f}")
    c4.metric("Max drawdown", f"{m['max_drawdown']*100:.1f}%")

    st.plotly_chart(charts.price_chart(result, strategy, show_candles),
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


def universe_view(strategy, params, period, cost_bps, refresh):
    rows = load_universe_rows()
    names = {r["Symbol"].strip() + ".NS": r["Company Name"].strip() for r in rows}
    all_syms = [r["Symbol"].strip() + ".NS" for r in rows]

    limit = st.slider("How many stocks to scan", 10, len(all_syms),
                      min(100, len(all_syms)), step=10)
    st.caption("First scan downloads data (slow); later scans read the cache.")

    if not st.button("Run scan", type="primary"):
        st.info("Pick your strategy + params in the sidebar, then run the scan.")
        return

    symbols = all_syms[:limit]
    prog = st.progress(0.0, text="Fetching data…")

    def cb(done, total, sym):
        prog.progress(min(done / total, 1.0), text=f"Fetching {done}/{total}: {sym}")

    ohlcv, failed = data.get_many(symbols, period=period, refresh=refresh, progress_cb=cb)
    prog.progress(1.0, text="Backtesting…")

    records = []
    for sym, df in ohlcv.items():
        if len(df) < 30:
            continue
        res = engine.run(df, strategy, params, cost_bps=cost_bps, symbol=sym)
        m = res.metrics
        records.append({
            "Symbol": sym.replace(".NS", ""),
            "Company": names.get(sym, ""),
            "Return %": round(m["total_return"] * 100, 1),
            "vs B&H %": round((m["total_return"] - res.benchmark_metrics["total_return"]) * 100, 1),
            "CAGR %": round(m["cagr"] * 100, 1),
            "Sharpe": round(m["sharpe"], 2),
            "MaxDD %": round(m["max_drawdown"] * 100, 1),
            "Win %": round(m["win_rate"] * 100, 1) if pd.notna(m["win_rate"]) else None,
            "Trades": m["num_trades"],
        })
    prog.empty()

    if not records:
        st.error("No results — data may have failed to download.")
        return

    table = pd.DataFrame(records).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    st.success(f"Scanned {len(table)} stocks"
               + (f" — {len(failed)} failed to load." if failed else "."))
    st.dataframe(table, use_container_width=True, height=520)
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

    tab1, tab2 = st.tabs(["Single stock", "Universe scan (Nifty 500)"])
    with tab1:
        single_stock_view(strategy, params, period, cost_bps, refresh)
    with tab2:
        universe_view(strategy, params, period, cost_bps, refresh)


if __name__ == "__main__":
    main()
