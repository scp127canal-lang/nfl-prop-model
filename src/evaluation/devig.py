"""Devigging, fair-price selection, EV / edge and fractional Kelly sizing.

Rule 5 (CLAUDE.md): model probabilities are never compared to raw implied odds. Every market is
devigged first, using a sharp reference book when one quoted it and a cross-book consensus
otherwise.

Methods
-------
* **multiplicative**: ``p_i = q_i / sum(q)`` with ``q_i = 1 / decimal_i``.
* **shin**: Shin (1993) insider-trading model. Solve for ``z`` such that
  ``p_i = (sqrt(z^2 + 4 (1 - z) q_i^2 / S) - z) / (2 (1 - z))`` sums to 1 (``S = sum(q)``).
  It takes more margin off longshots than favourites (favourite-longshot bias).

Both work for two-way and multi-way (mutually exclusive) markets and always return
probabilities summing to 1. One-sided quotes (e.g. "Yes" only on anytime TD) cannot be devigged
by pairing; ``one_way_fair_prob`` applies an assumed margin and labels the result.

Market grouping for raw_odds rows (one devig group = one set of mutually exclusive outcomes):
* spreads: team + canonical point (home -3.5 pairs with away +3.5)
* totals / team totals / player O/U: description + point, Over vs Under
* first/last TD scorer: every player's "Yes" in one multi-way group
* anytime TD: Yes vs No when both are quoted, otherwise one-way
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

SHARP_BOOKS: tuple[str, ...] = ("pinnacle", "circasports", "circa", "bookmaker", "bookmakereu")
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_BANKROLL = 10_000.0
DEFAULT_MAX_STAKE_PCT = 0.05
DEFAULT_ONE_WAY_MARGIN = 0.08  # assumed overround when only one side is quoted (e.g. anytime TD "Yes")
MULTIWAY_EXCLUSIVE = {"player_1st_td", "player_last_td"}
SPREAD_MARKETS = {"spreads", "alternate_spreads"}


# ======================================================================================
# Odds conversion
# ======================================================================================
def american_to_decimal(american) -> np.ndarray:
    a = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / np.abs(a))
    return np.where(a == 0, np.nan, out)


def decimal_to_american(decimal) -> np.ndarray:
    d = np.asarray(decimal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d >= 2.0, (d - 1) * 100.0, -100.0 / (d - 1))


def implied_prob(decimal) -> np.ndarray:
    """Raw (vigged) implied probability ``1 / decimal``. Never compare a model probability to this."""
    return 1.0 / np.asarray(decimal, dtype=float)


def overround(decimals: Sequence[float]) -> float:
    return float(implied_prob(decimals).sum())


# ======================================================================================
# Devig
# ======================================================================================
def _validate(decimals: Sequence[float]) -> np.ndarray:
    d = np.asarray(decimals, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("devig needs at least two mutually exclusive outcomes")
    if not np.all(np.isfinite(d)) or np.any(d <= 1.0):
        raise ValueError(f"decimal odds must be finite and > 1.0, got {d}")
    return d


def devig_multiplicative(decimals: Sequence[float]) -> np.ndarray:
    q = implied_prob(_validate(decimals))
    return q / q.sum()


def shin_z(decimals: Sequence[float]) -> float:
    """Shin's insider proportion z for a market (0 when there is no overround)."""
    q = implied_prob(_validate(decimals))
    s = q.sum()
    if s <= 1.0 + 1e-12:
        return 0.0

    def excess(z: float) -> float:
        return float(np.sum((np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s) - z) / (2 * (1 - z))) - 1.0)

    return float(brentq(excess, 0.0, 0.999, xtol=1e-14))


def devig_shin(decimals: Sequence[float]) -> np.ndarray:
    d = _validate(decimals)
    q = implied_prob(d)
    s = q.sum()
    if s <= 1.0 + 1e-12:  # no vig (or an arb): nothing to remove beyond normalisation
        return q / s
    z = shin_z(d)
    p = (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s) - z) / (2 * (1 - z))
    return p / p.sum()  # exact normalisation against root-finding tolerance


def devig(decimals: Sequence[float], method: str = "shin") -> np.ndarray:
    if method == "multiplicative":
        return devig_multiplicative(decimals)
    if method == "shin":
        return devig_shin(decimals)
    raise ValueError(f"unknown devig method {method!r}")


def devig_american(americans: Sequence[float], method: str = "shin") -> np.ndarray:
    return devig(american_to_decimal(americans), method)


def one_way_fair_prob(decimal: float, margin: float = DEFAULT_ONE_WAY_MARGIN) -> float:
    """Fallback for one-sided quotes: implied probability shrunk by an assumed overround."""
    return float(implied_prob(decimal) / (1.0 + margin))


# ======================================================================================
# EV, edge, Kelly
# ======================================================================================
def expected_value(p_win, decimal, p_push=0.0) -> np.ndarray:
    """EV per unit staked: p_win * (d - 1) - p_lose, pushes refund."""
    p_win, d, p_push = (np.asarray(x, dtype=float) for x in (p_win, decimal, p_push))
    return p_win * (d - 1.0) - (1.0 - p_win - p_push)


def ev_pct(p_win, decimal, p_push=0.0) -> np.ndarray:
    return 100.0 * expected_value(p_win, decimal, p_push)


def edge_pct(p_model, p_fair) -> np.ndarray:
    """Model probability minus devigged fair probability, in percentage points."""
    return 100.0 * (np.asarray(p_model, dtype=float) - np.asarray(p_fair, dtype=float))


def kelly_fraction(p_win, decimal, p_push=0.0) -> np.ndarray:
    """Full-Kelly fraction of bankroll, floored at 0 (no bet when EV <= 0).

    With pushes the bet is decided on the non-push outcomes: f* = (b p' - q') / b where
    p' = p_win / (1 - p_push).
    """
    p_win, d, p_push = (np.asarray(x, dtype=float) for x in (p_win, decimal, p_push))
    b = d - 1.0
    denom = np.clip(1.0 - p_push, 1e-12, None)
    p = np.clip(p_win / denom, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (b * p - (1.0 - p)) / b
    f = np.where((b > 0) & np.isfinite(f), f, 0.0)
    return np.clip(f, 0.0, 1.0)


def kelly_stake(p_win, decimal, bankroll: float = DEFAULT_BANKROLL, fraction: float = DEFAULT_KELLY_FRACTION,
                p_push=0.0, max_stake_pct: float | None = DEFAULT_MAX_STAKE_PCT) -> np.ndarray:
    """Fractional-Kelly stake in currency units (0.0 for negative or zero EV)."""
    if bankroll < 0 or fraction < 0:
        raise ValueError("bankroll and fraction must be non-negative")
    stake = bankroll * fraction * kelly_fraction(p_win, decimal, p_push)
    if max_stake_pct is not None:
        stake = np.minimum(stake, bankroll * max_stake_pct)
    return np.where(stake > 0, stake, 0.0)


# ======================================================================================
# raw_odds -> fair probabilities (sharp-first, consensus fallback)
# ======================================================================================
def _canonical_point(df: pd.DataFrame) -> pd.Series:
    """Grouping point. Spreads are keyed from the alphabetically-first team's side so that
    home -3.5 and away +3.5 fall in the same group."""
    point = pd.to_numeric(df["outcome_point"], errors="coerce").astype(float)
    if "home_team" not in df:
        return point.fillna(np.inf)
    first = np.where(df["home_team"].fillna("") <= df["away_team"].fillna(""), df["home_team"], df["away_team"])
    is_spread = df["market_key"].isin(SPREAD_MARKETS)
    canon = np.where(is_spread & (df["outcome_name"] != first), -point, point)
    return pd.Series(canon, index=df.index).fillna(np.inf)


def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``group_key`` (mutually exclusive set) and ``selection_key`` (one outcome)."""
    df = df.copy()

    def text(s: pd.Series) -> pd.Series:  # missing -> "" (pandas 3 keeps None missing under astype(str))
        return s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))

    def num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):g}")

    ev, mk = text(df["event_id"]), text(df["market_key"])
    desc = text(df["outcome_description"]) if "outcome_description" in df else pd.Series("", index=df.index)
    canon = _canonical_point(df).map(lambda v: "" if not np.isfinite(v) else f"{float(v):g}")
    multi = df["market_key"].isin(MULTIWAY_EXCLUSIVE)
    df["group_key"] = np.where(multi, ev + "|" + mk, ev + "|" + mk + "|" + desc + "|" + canon)
    df["selection_key"] = ev + "|" + mk + "|" + desc + "|" + text(df["outcome_name"]) + "|" + num(df["outcome_point"])
    return df


def devig_book_groups(df: pd.DataFrame, method: str = "shin",
                      one_way_margin: float = DEFAULT_ONE_WAY_MARGIN) -> pd.DataFrame:
    """Devig every (bookmaker, group) separately. Adds ``fair_prob_book`` and ``devig_method``."""
    df = add_group_keys(df)
    if "price_decimal" not in df:
        df["price_decimal"] = american_to_decimal(df["price_american"])
    fair = np.full(len(df), np.nan)
    how = np.empty(len(df), dtype=object)
    pos = {idx: i for i, idx in enumerate(df.index)}
    for _, g in df.groupby(["bookmaker_key", "group_key"], sort=False):
        g = g.drop_duplicates("selection_key", keep="last")
        rows = [pos[i] for i in g.index]
        if len(g) >= 2:
            fair[rows] = devig(g["price_decimal"].to_numpy(), method)
            how[rows] = method
        else:
            fair[rows] = one_way_fair_prob(float(g["price_decimal"].iloc[0]), one_way_margin)
            how[rows] = "one_way_assumed_margin"
    df["fair_prob_book"] = fair
    df["devig_method"] = how
    return df


def fair_probabilities(df: pd.DataFrame, method: str = "shin", sharp_books: Sequence[str] = SHARP_BOOKS,
                       one_way_margin: float = DEFAULT_ONE_WAY_MARGIN) -> pd.DataFrame:
    """One row per selection: fair probability, its source and the best available price.

    For every group, the highest-priority sharp book quoting the *complete* group supplies the
    fair probabilities. Otherwise the per-book devigged probabilities are averaged across books
    (consensus) and renormalised within the group so it still sums to 1.
    """
    if df.empty:
        return pd.DataFrame()
    d = devig_book_groups(df, method, one_way_margin)
    d = d[d["fair_prob_book"].notna()]
    priority = {b: i for i, b in enumerate(sharp_books)}
    group_size = d.groupby("group_key")["selection_key"].nunique()
    out = []
    for gk, g in d.groupby("group_key", sort=False):
        n_sel = group_size[gk]
        complete = g.groupby("bookmaker_key")["selection_key"].nunique() == n_sel
        sharp = sorted((b for b in complete[complete].index if b in priority), key=priority.get)
        if sharp:
            src = g[g["bookmaker_key"] == sharp[0]].drop_duplicates("selection_key", keep="last")
            fair = src.set_index("selection_key")["fair_prob_book"]
            source = sharp[0]
        else:
            fair = g.groupby("selection_key")["fair_prob_book"].mean()
            if n_sel >= 2:
                fair = fair / fair.sum()
            source = "consensus"
        best = g.sort_values("price_decimal", ascending=False).drop_duplicates("selection_key")
        best = best.set_index("selection_key")
        nb = g.groupby("selection_key")["bookmaker_key"].nunique()
        for sel, p in fair.items():
            b = best.loc[sel]
            out.append({**{c: b[c] for c in ("event_id", "market_key", "outcome_name", "outcome_description",
                                            "outcome_point") if c in b.index},
                        "group_key": gk, "selection_key": sel, "fair_prob": float(p), "fair_source": source,
                        "devig_method": src["devig_method"].iloc[0] if sharp else g["devig_method"].iloc[0],
                        "n_books": int(nb.get(sel, 0)), "best_book": b["bookmaker_key"],
                        "best_price_decimal": float(b["price_decimal"]),
                        "best_price_american": float(decimal_to_american(b["price_decimal"]))})
    return pd.DataFrame(out)


def evaluate_selections(fair: pd.DataFrame, model_prob_col: str = "model_prob", push_prob_col: str | None = None,
                        bankroll: float = DEFAULT_BANKROLL, kelly: float = DEFAULT_KELLY_FRACTION,
                        max_stake_pct: float | None = DEFAULT_MAX_STAKE_PCT) -> pd.DataFrame:
    """Add EV %, edge %, market EV % (fair prob at best price) and fractional-Kelly stake."""
    out = fair.copy()
    p = out[model_prob_col].to_numpy(float)
    push = out[push_prob_col].to_numpy(float) if push_prob_col else 0.0
    d = out["best_price_decimal"].to_numpy(float)
    out["ev_pct"] = ev_pct(p, d, push)
    out["edge_pct"] = edge_pct(p, out["fair_prob"].to_numpy(float))
    out["market_ev_pct"] = ev_pct(out["fair_prob"].to_numpy(float), d)
    out["kelly_full"] = kelly_fraction(p, d, push)
    out["stake"] = kelly_stake(p, d, bankroll, kelly, push, max_stake_pct)
    return out


def clv_pct(bet_decimal, close_fair_prob) -> np.ndarray:
    """Closing line value: EV of the bet price if the devigged closing probability is true."""
    return ev_pct(close_fair_prob, bet_decimal)


def sum_to_one(groups: Iterable[np.ndarray], tol: float = 1e-9) -> bool:
    return all(abs(float(np.sum(g)) - 1.0) <= tol for g in groups)
