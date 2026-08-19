"""Plotly figures for the Streamlit UI, drawn on the basis of the strategy."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import metrics as _metrics

_UP = "#16a34a"
_DOWN = "#dc2626"
_BENCH = "#64748b"
_STRAT = "#2563eb"


def price_chart(result, strategy, show_candles: bool = True) -> go.Figure:
    """Price + strategy overlays + buy/sell markers, with an optional lower panel."""
    df = result.ohlcv
    sub = strategy.subplot(df, **result.params)
    rows = 2 if sub else 1
    heights = [0.72, 0.28] if sub else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=heights,
    )

    if show_candles:
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price",
                increasing_line_color=_UP, decreasing_line_color=_DOWN,
                showlegend=False,
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], name="Close",
                       line=dict(color="#0f172a", width=1)),
            row=1, col=1,
        )

    # Strategy overlays on the price panel (MAs / Bollinger bands).
    for label, series in strategy.overlays(df, **result.params).items():
        fig.add_trace(
            go.Scatter(x=series.index, y=series, name=label, line=dict(width=1.2)),
            row=1, col=1,
        )

    # Trade markers, colored by outcome: winners green, losers red.
    trades = result.trades
    if not trades.empty:
        wins = trades[trades["return_pct"] > 0]
        losses = trades[trades["return_pct"] <= 0]

        # Connecting segment from entry -> exit for each trade (color = outcome).
        def _segments(tr):
            xs, ys = [], []
            for _, t in tr.iterrows():
                xs += [t["entry_date"], t["exit_date"], None]
                ys += [t["entry_price"], t["exit_price"], None]
            return xs, ys

        for grp, color, name in ((wins, _UP, "Winning trade"), (losses, _DOWN, "Losing trade")):
            if grp.empty:
                continue
            xs, ys = _segments(grp)
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", name=name, opacity=0.55,
                           line=dict(color=color, width=1.4), hoverinfo="skip"),
                row=1, col=1,
            )

        # Entry markers (neutral blue) — where the strategy bought.
        fig.add_trace(
            go.Scatter(
                x=trades["entry_date"], y=trades["entry_price"], mode="markers",
                name="Entry", customdata=trades["return_pct"] * 100,
                marker=dict(symbol="triangle-up", size=10, color="#2563eb",
                            line=dict(width=0.6, color="white")),
                hovertemplate="Entry %{x|%d %b %Y}<br>₹%{y:.1f}"
                              "<br>trade P/L %{customdata:.1f}%<extra></extra>",
            ),
            row=1, col=1,
        )

        # Exit markers, split by win/loss.
        for grp, color, name in ((wins, _UP, "Exit — win"), (losses, _DOWN, "Exit — loss")):
            if grp.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=grp["exit_date"], y=grp["exit_price"], mode="markers",
                    name=name, customdata=grp["return_pct"] * 100,
                    marker=dict(symbol="triangle-down", size=12, color=color,
                                line=dict(width=0.6, color="white")),
                    hovertemplate=name + " %{x|%d %b %Y}<br>₹%{y:.1f}"
                                  "<br>P/L %{customdata:.1f}%<extra></extra>",
                ),
                row=1, col=1,
            )

    # Lower indicator panel (RSI / MACD).
    if sub:
        title, series_map = sub
        for label, series in series_map.items():
            if label == "_hlines":
                for y in series:
                    fig.add_hline(y=y, line=dict(color="#94a3b8", dash="dash", width=1),
                                  row=2, col=1)
            elif label == "_hist":
                colors = [_UP if v >= 0 else _DOWN for v in series.fillna(0)]
                fig.add_trace(
                    go.Bar(x=series.index, y=series, name="Hist", marker_color=colors,
                           opacity=0.5),
                    row=2, col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(x=series.index, y=series, name=label, line=dict(width=1.2)),
                    row=2, col=1,
                )
        fig.update_yaxes(title_text=title, row=2, col=1)

    fig.update_layout(
        height=640 if sub else 520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def equity_chart(result) -> go.Figure:
    """Strategy equity vs buy & hold."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=result.equity.index, y=result.equity, name="Strategy",
                   line=dict(color=_STRAT, width=2))
    )
    fig.add_trace(
        go.Scatter(x=result.benchmark_equity.index, y=result.benchmark_equity,
                   name="Buy & Hold", line=dict(color=_BENCH, width=1.5, dash="dot"))
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        title="Equity curve (growth of 1)", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hovermode="x unified",
    )
    return fig


def drawdown_chart(result) -> go.Figure:
    dd = _metrics.drawdown_series(result.equity)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd * 100, name="Drawdown", fill="tozeroy",
                   line=dict(color=_DOWN, width=1))
    )
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=30, b=10),
        title="Drawdown (%)", template="plotly_white",
        yaxis=dict(ticksuffix="%"), hovermode="x unified",
    )
    return fig
