"""Plotly charts, tables and small UI pieces for the dashboard (dark theme).

Colors: the validated categorical dark-mode steps (fixed order, never cycled), with green/red
reserved for profit/loss and best/off-market status, which always carry a text label too.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SURFACE = "#1a1a19"
PAPER = "#121211"
TEXT = "#f0efec"
TEXT_2 = "#c3c2b7"
GRID = "#2c2c2a"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
GOOD, BAD, WARN = "#0ca30c", "#e66767", "#c98500"
FONT = "Inter, Segoe UI, system-ui, sans-serif"

CSS = f"""
<style>
/* clear Streamlit's fixed header so it never clips the tabs (mobile + desktop) */
.block-container {{ padding-top: 3.5rem !important; padding-bottom: 2rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
[data-testid="stMetricLabel"] p {{ color: {TEXT_2}; }}
.small-note {{ color: {TEXT_2}; font-size: 0.85rem; }}
.pill {{ display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.78rem;
        border: 1px solid {GRID}; color: {TEXT_2}; margin-right: 4px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def series_color(i: int) -> str:
    return SERIES[i % len(SERIES)] if i < len(SERIES) else TEXT_2  # > 8 series fold to neutral


def base_layout(fig: go.Figure, title: str | None = None, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PAPER, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, color=TEXT, size=13), title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        margin=dict(l=56, r=24, t=48 if title else 16, b=48), hoverlabel=dict(bgcolor="#262624", font_color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def fmt_american(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "—"
    v = int(round(float(v)))
    return f"+{v}" if v > 0 else str(v)


def kpi_row(items: Sequence[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        col.metric(label, value, help=help_)


# ======================================================================================
# Weather badge
# ======================================================================================
HIGH_WIND_MPH = 15.0


def weather_label(weather: dict | None) -> tuple[str, str, bool]:
    """(text, badge color, high-wind alert) for a ``weather_client`` dict (None = not fetched)."""
    if weather is None:
        return "🌤️ Outdoor: forecast not loaded — click Re-run Match Simulations", "gray", False
    if weather.get("is_dome"):
        return "🏟️ Dome (Controlled)", "blue", False
    if weather.get("available") is False:
        return f"☁️ {weather.get('summary', 'No forecast')} · no weather adjustment", "gray", False
    wind = float(weather.get("wind_mph") or 0.0)
    temp = weather.get("temp_f")
    text = (f"🌬️ Outdoor: {temp:.0f}°F | Wind: {wind:.0f} mph | Precip: {int(weather.get('precip_prob') or 0)}%"
            if temp is not None else f"🌬️ {weather.get('summary', '')}")
    high = wind > HIGH_WIND_MPH
    return (text + " · high wind" if high else text), ("red" if high else "green"), high


def weather_badge(weather: dict | None, where=None) -> None:
    """Compact forecast badge; wind above 15 mph is flagged red with a warning icon."""
    where = where or st
    text, color, high = weather_label(weather)
    where.badge(text, icon=":material/warning:" if high else None, color=color)


# ======================================================================================
# Scanner
# ======================================================================================
SCANNER_COLUMN_CONFIG = {
    "Line": st.column_config.NumberColumn(format="%.1f"),
    "Best Odds": st.column_config.NumberColumn(format="%+d"),
    "Sharp No-Vig Fair": st.column_config.NumberColumn(format="%+d", help="Devigged fair odds (sharp book first, else consensus)"),
    "Model Fair": st.column_config.NumberColumn(format="%+d", help="Fair odds implied by the 10k-trial simulation / TD model"),
    "Edge %": st.column_config.ProgressColumn(format="%.1f", min_value=0.0, max_value=15.0,
                                              help="Model probability minus devigged fair probability (pp)"),
    "EV %": st.column_config.NumberColumn(format="%.1f"),
    "Kelly Stake $": st.column_config.NumberColumn(format="$%.0f"),
    "Hist. Games": st.column_config.NumberColumn(format="%d", help="Prior games with >20% snap share (player props). "
                                                 "Below 3, usage is blended with a positional role prior."),
    "Books": st.column_config.NumberColumn(format="%d"),
}


def edge_table(df: pd.DataFrame, key: str) -> None:
    st.dataframe(df, width="stretch", hide_index=True, column_config=SCANNER_COLUMN_CONFIG, key=key,
                 height=min(38 + 35 * max(len(df), 1), 640))


# ======================================================================================
# Player sim histogram
# ======================================================================================
# Horizontal legend centred under the plot + roomy margins, so on a narrow (portrait phone)
# screen the legend never overlaps the title or the "mean" annotation.
PLAYER_SIM_LEGEND = dict(
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=56, r=24, t=60, b=90))


def sim_histogram(x: np.ndarray, lines: pd.DataFrame, stat_label: str, player: str,
                  highlight_line: float | None = None) -> go.Figure:
    """Histogram of simulated outcomes with a vertical line per (book, line)."""
    x = np.asarray(x, float)
    discrete = np.allclose(x, np.round(x)) and (x.max() - x.min()) <= 60
    fig = go.Figure()
    if discrete:
        vals, counts = np.unique(x, return_counts=True)
        fig.add_bar(x=vals, y=counts / len(x), marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
                    name="Simulated", hovertemplate=f"{stat_label} = %{{x:.0f}}<br>P = %{{y:.1%}}<extra></extra>")
    else:
        lo, hi = np.percentile(x, [0.2, 99.8])
        fig.add_histogram(x=np.clip(x, lo, hi), histnorm="probability", nbinsx=60, name="Simulated",
                          marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=1)),
                          hovertemplate=f"{stat_label} %{{x}}<br>P = %{{y:.1%}}<extra></extra>")
    mean = float(x.mean())
    fig.add_vline(x=mean, line=dict(color=TEXT_2, width=2, dash="dot"),
                  annotation=dict(text=f"mean {mean:.1f}", font_color=TEXT_2), annotation_position="top")
    if lines is not None and not lines.empty:
        books = list(dict.fromkeys(lines["bookmaker_key"]))
        for _, r in lines.iterrows():
            c = series_color(1 + books.index(r["bookmaker_key"]) % 7)
            dash = "dash" if r["is_alternate"] else "solid"
            width = 3 if highlight_line is not None and float(r["line"]) == float(highlight_line) else 2
            fig.add_vline(x=float(r["line"]), line=dict(color=c, width=width, dash=dash), opacity=0.9)
        # legend entries (one per book) via invisible traces so identity is never color alone
        for i, b in enumerate(books):
            fig.add_scatter(x=[None], y=[None], mode="lines", name=b,
                            line=dict(color=series_color(1 + i % 7), width=2))
        fig.add_scatter(x=[None], y=[None], mode="lines", name="alternate line", line=dict(color=TEXT_2, dash="dash"))
        # hover markers carrying the prices at each line
        agg = lines.groupby("line").apply(lambda g: "<br>".join(
            f"{b}: O {fmt_american(o)} / U {fmt_american(u)}" for b, o, u in zip(g["bookmaker_key"], g["Over"], g["Under"])),
            include_groups=False)
        fig.add_scatter(x=agg.index, y=[0] * len(agg), mode="markers", marker=dict(size=12, color="rgba(0,0,0,0)"),
                        hovertext=agg.values, hovertemplate="line %{x}<br>%{hovertext}<extra></extra>", showlegend=False)
    base_layout(fig, f"{player} — {stat_label}: {len(x):,} simulated games", height=460, bargap=0.05)
    fig.update_layout(**PLAYER_SIM_LEGEND)
    fig.update_xaxes(title=stat_label)
    fig.update_yaxes(title="Probability", tickformat=".0%")
    return fig


# ======================================================================================
# Line shopper
# ======================================================================================
def line_shop_styler(price: pd.DataFrame, flags: pd.DataFrame):
    def cell_style(flag: str) -> str:
        if not isinstance(flag, str) or not flag:
            return ""
        if flag == "fair":
            return f"color: {TEXT_2}; font-style: italic;"
        styles = []
        if "best" in flag:
            styles.append(f"background-color: rgba(12,163,12,0.28); font-weight: 600;")
        if "off" in flag:
            styles.append(f"outline: 2px solid {BAD}; outline-offset: -2px;")
        if "stale" in flag:
            styles.append(f"background-color: rgba(201,133,0,0.30);")
        return " ".join(styles)

    styled = price.style.format(fmt_american).apply(lambda _: flags.map(cell_style), axis=None)
    return styled


def flag_legend() -> None:
    st.markdown(
        f"<span class='pill' style='background:rgba(12,163,12,0.28)'>best price</span>"
        f"<span class='pill' style='outline:2px solid {BAD}'>off-market (no-vig ≥ threshold from fair)</span>"
        f"<span class='pill' style='background:rgba(201,133,0,0.30)'>stale (updated ≥ N min before freshest book)</span>",
        unsafe_allow_html=True)


def off_market_fig(detail: pd.DataFrame, threshold: float) -> go.Figure:
    d = detail.dropna(subset=["off_pp"])
    fig = go.Figure()
    books = list(dict.fromkeys(d["bookmaker_key"]))
    for i, b in enumerate(books):
        g = d[d["bookmaker_key"] == b]
        fig.add_scatter(x=g["off_pp"], y=g["label"], mode="markers", name=b,
                        marker=dict(size=10, color=series_color(i), line=dict(color=SURFACE, width=2)),
                        customdata=np.stack([g["price_american"], g["lag_min"]], axis=1),
                        hovertemplate=f"{b}<br>%{{y}}<br>no-vig − fair: %{{x:+.2f}} pp"
                                      "<br>price %{customdata[0]:+.0f}<br>updated %{customdata[1]:.0f} min before freshest"
                                      "<extra></extra>")
    for t in (-threshold, threshold):
        fig.add_vline(x=t, line=dict(color=BAD, width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=TEXT_2, width=1))
    base_layout(fig, "Book no-vig probability vs fair (pp)", height=max(260, 34 * d["label"].nunique() + 120))
    fig.update_xaxes(title="percentage points (book − fair)")
    return fig


# ======================================================================================
# Backtest
# ======================================================================================
def equity_curve_fig(curve: pd.DataFrame, bankroll: float) -> go.Figure:
    fig = go.Figure()
    series = ["All markets"] + sorted(s for s in curve["series"].unique() if s != "All markets")
    for i, s in enumerate(series):
        g = curve[curve["series"] == s]
        fig.add_scatter(x=g["game_date"], y=g["bankroll"], mode="lines", name=s,
                        line=dict(color=series_color(i), width=3 if i == 0 else 2),
                        customdata=g["n_bets"], hovertemplate=f"{s}<br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}"
                                                              "<br>%{customdata} bets<extra></extra>")
    fig.add_hline(y=bankroll, line=dict(color=TEXT_2, width=1, dash="dot"),
                  annotation=dict(text=f"start ${bankroll:,.0f}", font_color=TEXT_2))
    base_layout(fig, "Bankroll equity (fractional Kelly on a fixed starting bankroll)", height=440,
                hovermode="x unified")
    fig.update_yaxes(title="Bankroll $", tickprefix="$", tickformat=",.0f")
    return fig


def roi_bar_fig(roi: pd.DataFrame) -> go.Figure:
    d = roi.sort_values("roi_pct")
    colors = [GOOD if v >= 0 else BAD for v in d["roi_pct"]]
    fig = go.Figure(go.Bar(
        x=d["roi_pct"], y=d["market_category"], orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{v:+.1f}%  ({n} bets)" for v, n in zip(d["roi_pct"], d["n_bets"])], textposition="outside",
        textfont=dict(color=TEXT), customdata=np.stack([d["flat_roi_pct"], d["hit_rate"]], axis=1),
        hovertemplate="%{y}<br>Kelly ROI %{x:+.2f}%<br>flat ROI %{customdata[0]:+.2f}%"
                      "<br>hit rate %{customdata[1]:.1%}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=TEXT_2, width=1))
    span = max(5.0, float(d["roi_pct"].abs().max()) * 1.5) if len(d) else 5.0
    base_layout(fig, "ROI by market (Kelly-staked)", height=max(260, 48 * len(d) + 100))
    fig.update_xaxes(title="ROI %", range=[-span, span], ticksuffix="%")
    return fig


def calibration_fig(cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(color=TEXT_2, dash="dot", width=1))
    for i, (src, g) in enumerate(cal.groupby("source", sort=False)):
        fig.add_scatter(x=g["mean_pred"], y=g["observed"], mode="lines+markers", name=src,
                        line=dict(color=series_color(i), width=2),
                        marker=dict(size=np.clip(np.sqrt(g["n"]) / 2, 8, 22), color=series_color(i),
                                    line=dict(color=SURFACE, width=2)),
                        customdata=g["n"], hovertemplate=f"{src}<br>predicted %{{x:.1%}}<br>observed %{{y:.1%}}"
                                                         "<br>n = %{customdata:,}<extra></extra>")
    base_layout(fig, "Calibration (reliability) curve", height=440)
    fig.update_xaxes(title="Predicted probability", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="Observed win rate", range=[0, 1], tickformat=".0%")
    return fig


def clv_hist_fig(bets: pd.DataFrame) -> go.Figure:
    b = bets[bets["is_bet"].astype(bool) & bets["clv_pct"].notna()]
    fig = go.Figure(go.Histogram(x=b["clv_pct"], nbinsx=40, marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=1)),
                                 hovertemplate="CLV %{x}%<br>%{y} bets<extra></extra>", name="CLV"))
    fig.add_vline(x=0, line=dict(color=TEXT_2, width=1))
    if len(b):
        m = float(b["clv_pct"].mean())
        fig.add_vline(x=m, line=dict(color=GOOD if m > 0 else BAD, width=2, dash="dash"),
                      annotation=dict(text=f"mean {m:+.2f}%", font_color=TEXT))
    base_layout(fig, "Closing line value of placed bets", height=360, showlegend=False)
    fig.update_xaxes(title="CLV % (EV of bet price at devigged close)", ticksuffix="%")
    fig.update_yaxes(title="Bets")
    return fig


def empty_fig(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=TEXT_2, size=14), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    base_layout(fig, height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
