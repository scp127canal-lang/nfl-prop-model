"""NFL all-markets & player-prop +EV dashboard.

Run:  streamlit run app.py

Rerun safety (CLAUDE.md rule 2): Streamlit re-executes this file on every interaction. The only
live Odds API path is the ``on_click`` callback of the sidebar "Pull Live Odds" button; every
DuckDB read and simulation goes through ``st.cache_data`` / ``st.cache_resource`` in
``src.app.state``. Importing this module renders nothing (``main()`` runs under Streamlit only).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.app import components as ui
from src.app import state
from src.evaluation import devig as dv
from src.ingestion import odds_client as oc

REGIONS = ["us", "us2", "eu", "uk", "au"]
CATEGORY_ORDER = ["moneyline", "spread", "total", "team_total", "player_passing", "player_rushing",
                  "player_receiving", "player_anytime_td", "player_first_td", "player_other"]


# ======================================================================================
# Sidebar
# ======================================================================================
def sidebar(version: str) -> state.Settings:
    sb = st.sidebar
    sb.title("🏈 NFL +EV Model")
    sb.caption(f"DB: `{state.db_path().name}`")

    sb.subheader("Bankroll & sizing")
    sb.number_input("Bankroll $", min_value=0.0, step=500.0, key="bankroll", format="%.0f")
    sb.slider("Kelly fraction", 0.05, 1.0, step=0.05, key="kelly",
              help="Fraction of full Kelly. 0.25 = quarter Kelly (default). Stakes are capped at 5% of bankroll.")
    sb.slider("Min edge %", 0.0, 15.0, step=0.5, key="min_edge_pct",
              help="Model probability minus devigged fair probability, in percentage points.")

    sb.subheader("Fair-price reference")
    books = state.cached_bookmakers(version)
    options = [state.SHARP_AUTO] + list(dict.fromkeys(list(dv.SHARP_BOOKS) + books))
    if st.session_state["sharp_book"] not in options:
        st.session_state["sharp_book"] = state.SHARP_AUTO
    sb.selectbox("Sharp reference book", options, key="sharp_book",
                 format_func=lambda b: "Auto (Pinnacle → Circa → Bookmaker)" if b == state.SHARP_AUTO else b,
                 help="Book used for no-vig fair odds when it quotes the full market; otherwise the "
                      "remaining sharp books, then a consensus of devigged books.")
    sb.radio("Devig method", ["shin", "multiplicative"], key="devig_method", horizontal=True)

    depth_panel(version)
    live_odds_panel(version)
    return state.settings_from_session()


# ======================================================================================
# Injury & depth chart overrides (global: scanner + player sim)
# ======================================================================================
RERUN_LABEL = "🔄 Re-run Match Simulations"
AUTO_QB = "auto"


def _store_qb_pick(team: str) -> None:
    staged = dict(st.session_state.get("qb_staged", {}))
    pick = st.session_state.get(f"qb_pick_{team}", AUTO_QB)
    if pick == AUTO_QB:
        staged.pop(team, None)
    else:
        staged[team] = pick
    st.session_state["qb_staged"] = staged


def _clear_qb_picks() -> None:
    st.session_state["qb_staged"] = {}
    for k in [k for k in st.session_state if str(k).startswith("qb_pick_")]:
        st.session_state[k] = AUTO_QB


def depth_panel(version: str) -> None:
    """Inactive list + starting-QB replacement. Changes are staged; the Re-run button applies
    them (and clears the simulation caches) for the +EV scanner and the player sim alike."""
    sb = st.sidebar
    sb.subheader("Injury & depth chart overrides")
    players = state.cached_depth_players(version)
    ids = players["gsis_id"].tolist()
    default_out = players.loc[players["default_out"], "gsis_id"].tolist()
    if "depth_inactive" not in st.session_state:
        st.session_state["depth_inactive"] = default_out
    else:   # drop ids that left the slate (new DB version) so the widget value stays valid
        st.session_state["depth_inactive"] = [i for i in st.session_state["depth_inactive"] if i in set(ids)]
    labels = dict(zip(players["gsis_id"], players["label"]))
    sb.multiselect("Inactive / Out players", ids, key="depth_inactive", format_func=lambda i: labels.get(i, i),
                   placeholder="Nobody ruled out",
                   help="Preloaded with players listed Out, Doubtful or IR for this week's slate. Their target, "
                        "carry and snap share go to 0 and their volume is redistributed to active teammates. "
                        "Missing WR1/WR2/TE1 also lowers the QB's yards per attempt (≈13.5% per 0.25 target "
                        "share). Remove a player to re-activate him.")

    qbs = players[players["position"] == "QB"]
    teams = sorted(qbs["team"].unique())
    if teams:
        team = sb.selectbox("Team (starting QB override)", teams, key="qb_team")
        options = [AUTO_QB] + qbs.loc[qbs["team"] == team, "gsis_id"].tolist()
        key = f"qb_pick_{team}"
        current = st.session_state.get("qb_staged", {}).get(team, AUTO_QB)
        if st.session_state.get(key) not in options:
            st.session_state[key] = current if current in options else AUTO_QB
        sb.selectbox("Starting QB / key replacement", options, key=key, on_change=_store_qb_pick, args=(team,),
                     format_func=lambda i: "Auto (depth chart, healthy QB)" if i == AUTO_QB else labels.get(i, i),
                     help="Pick who starts (e.g. a backup when QB1 is out). A QB who isn't the team's "
                          "established QB1 throws with the backup efficiency factor below.")
    sb.slider("Backup QB efficiency (× YPA / CPOE)", 0.75, 1.0, step=0.01, key="qb_eff",
              help="Multiplier on yards per attempt for a non-QB1 starter (catch rate × f^0.4, yards per "
                   "catch × f^0.6). 0.82–0.88 is typical; 1.0 = no penalty.")
    sb.slider("Replacement QB share of pass attempts", 0.5, 1.0, step=0.05, key="qb_att_share",
              help="Fraction of the team's passing credited to an overridden starting QB (e.g. 0.7 if a "
                   "split or an in-game change is expected).")
    staged_qbs = st.session_state.get("qb_staged", {})
    if staged_qbs:
        sb.caption("QB overrides: " + ", ".join(f"{t} → {labels.get(q, q)}" for t, q in sorted(staged_qbs.items())))
        sb.button("Clear QB overrides", key="qb_clear", on_click=_clear_qb_picks, width="stretch")

    staged = state.staged_depth_spec(players)
    if st.session_state.get("depth_applied") is None:      # first render: apply the injury report
        st.session_state["depth_applied"] = staged
    pending = staged != st.session_state["depth_applied"]
    if pending:
        sb.warning("Overrides changed — re-run the simulations to apply them.")
    rerun_button(staged, key="rerun_sidebar", primary=pending)
    applied = st.session_state["depth_applied"]
    sb.caption(f"Applied: {len(applied['inactive'])} out, {len(applied['reactivate'])} re-activated, "
               f"{len(applied['starting_qbs'])} QB override(s).")
    w = st.session_state.get("weather") or {}
    if st.session_state.get("weather_fetched_at"):
        windy = sum(1 for x in w.values() if float(x.get("wind_mph") or 0) > ui.HIGH_WIND_MPH)
        sb.caption(f"Weather (Open-Meteo) for {len(w)} games, {st.session_state['weather_fetched_at'][11:16]} UTC"
                   + (f" · :red[{windy} with wind > 15 mph]" if windy else ""))
    else:
        sb.caption("Weather: domes applied; outdoor forecasts load on Re-run (Open-Meteo, free).")


def rerun_button(spec: dict | None = None, key: str = "rerun", primary: bool = True, where=None) -> None:
    where = where or st.sidebar
    if spec is None:
        spec = state.staged_depth_spec(state.cached_depth_players(state.db_version()))
    where.button(RERUN_LABEL, key=key, type="primary" if primary else "secondary", width="stretch",
                 on_click=state.rerun_simulations_callback, args=(spec,),
                 help="Apply the injury / depth overrides and the sliders, clear the cached simulations and "
                      "re-run every 10,000-trial Monte Carlo (scanner probabilities and Edge % update too).")


def live_odds_panel(version: str) -> None:
    sb = st.sidebar
    sb.subheader("Live odds")
    quota = state.last_known_quota()
    remaining = quota.get("remaining")
    as_of = (quota.get("as_of") or "")[:16].replace("T", " ")
    sb.metric("API credits remaining", "unknown" if remaining is None else f"{remaining:,}",
              help="Last value reported by x-requests-remaining in a cached payload"
                   + (f" ({as_of} UTC)" if as_of else " — no live pull yet") + ". Reading it costs 0 credits.")

    groups = sb.multiselect("Markets", list(state.MARKET_GROUP_LABELS), default=["main", "team_totals", "player_props"],
                            format_func=state.MARKET_GROUP_LABELS.get, key="pull_groups")
    regions = sb.multiselect("Regions", REGIONS, default=["us"], key="pull_regions")
    max_events = int(sb.number_input("Max games", 1, 32, 16, key="pull_max_events",
                                     help="Soonest kickoffs first. The events list itself costs 0 credits."))
    cost = state.estimate_pull_cost(groups, regions or ["us"], max_events)
    n_markets = len(state.selected_markets(groups))
    sb.markdown(f"**Estimated cost: up to {cost:,} credits**  \n"
                f"<span class='small-note'>{max_events} games × {n_markets} markets × {len(regions or ['us'])} "
                f"region(s); markets a book doesn't offer aren't charged. Fresh cached snapshots (<15 min) are "
                f"reused at 0 cost.</span>", unsafe_allow_html=True)
    too_expensive = remaining is not None and cost > remaining
    if too_expensive:
        sb.error(f"Estimate exceeds remaining credits ({remaining:,}).")
    has_key = state.api_key_configured()
    if not has_key:
        sb.warning("ODDS_API_KEY is not set in .env or Streamlit secrets — live pulls disabled.")
    confirm = sb.checkbox(f"I approve spending up to {cost:,} credits", key="pull_confirm")
    sb.button("Pull Live Odds", type="primary", width="stretch",
              disabled=not (has_key and confirm and groups and regions and cost > 0 and not too_expensive),
              on_click=state.pull_live_odds_callback, kwargs=dict(groups=tuple(groups), regions=tuple(regions),
                                                                  max_events=max_events))
    n_cached = quota.get("n_files", 0)
    sb.button(f"Load cached odds (0 credits · {n_cached} files)", width="stretch", disabled=n_cached == 0,
              on_click=state.load_cached_odds_callback)

    if st.session_state.get("pull_error"):
        sb.error(f"Pull failed: {st.session_state['pull_error']}")
    if (lp := st.session_state.get("last_pull")):
        sb.success(f"{lp['at'][11:16]} UTC — {lp['events']} games, {lp['fetched']} fetched, "
                   f"{lp['from_cache']} from cache, {lp['rows']:,} rows, {lp['credits_spent']} credits spent"
                   + (f", {lp['remaining']:,} remaining" if lp.get("remaining") is not None else ""))


# ======================================================================================
# Tab 1: +EV scanner
# ======================================================================================
def scanner_tab(version: str, settings: state.Settings) -> None:
    c1, c2 = st.columns([3, 1])
    c1.subheader("+EV Market Scanner")
    include_started = c2.toggle("Include started games", value=False, key="scan_started")
    depth = state.applied_depth_spec()
    priced, diag = state.cached_price_markets(version, settings.sharp_priority(), settings.devig_method,
                                              settings.n_trials, settings.seed, include_started,
                                              state.depth_json(depth),
                                              state.weather_json(st.session_state.get("weather")))
    if priced.empty:
        st.info("No priced markets yet. `raw_odds` has no pre-kickoff snapshots for upcoming games — use "
                "**Load cached odds** (0 credits) or **Pull Live Odds** in the sidebar.")
        _scanner_diagnostics(diag)
        return
    ev = state.apply_bankroll(priced, settings)

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    cats = [c for c in CATEGORY_ORDER if c in set(ev["market_category"])]
    sel_cats = f1.multiselect("Market types", cats, default=cats, key="scan_cats")
    sel_games = f2.multiselect("Games", sorted(ev["game"].unique()), key="scan_games", placeholder="All games")
    sel_books = f3.multiselect("Best book", sorted(ev["best_book"].unique()), key="scan_books", placeholder="All books")
    with f4:
        positive_only = st.toggle("+EV only", value=True, key="scan_pos")
        alts = st.toggle("Alternates", value=True, key="scan_alts")
    g1, g2, g3, _ = st.columns([1, 1, 1, 1])
    max_edge = g1.number_input("Max edge %", min_value=0.5, max_value=100.0, value=state.DEFAULT_MAX_EDGE_PCT,
                               step=0.5, key="scan_max_edge",
                               help="Hide selections whose edge exceeds this cap: edges this large are usually "
                                    "model error (e.g. thin-history players), not a mispriced market.")
    min_games = int(g2.number_input("Min historical games played", min_value=0, max_value=50,
                                    value=state.DEFAULT_MIN_HIST_GAMES, step=1, key="scan_min_games",
                                    help="Player props only: games with >20% snap share before this game. "
                                         "Game markets and players you activated by hand (starting-QB "
                                         "override, re-activated) are unaffected."))
    min_books = int(g3.number_input("Min books", min_value=1, max_value=20, value=state.DEFAULT_MIN_BOOKS, step=1,
                                    key="scan_min_books", help="Hide selections quoted by fewer sportsbooks."))
    filtered = state.filter_evaluated(ev, positive_only, sel_cats, sel_games, sel_books, alts, max_edge, min_games,
                                      min_books)
    table = state.format_scanner(filtered)

    bets = filtered[filtered["is_bet"]]
    capped = int((ev["is_bet"] & (ev["edge_pct"] > max_edge)).sum())
    manual = ev["manual_depth"].fillna(False).astype(bool) if "manual_depth" in ev else False
    thin = int((ev["is_bet"] & ev["market_key"].str.startswith("player_")
                & (ev["hist_games"].fillna(0) < min_games) & ~manual).sum()) if min_games else 0
    few = int((ev["is_bet"] & (ev["n_books"].fillna(0) < min_books)).sum())
    if capped or thin or few:
        st.caption(f"Hidden by filters: {capped} bet(s) above the {max_edge:g}% edge cap, "
                   f"{thin} prop bet(s) on players with < {min_games} qualifying games, "
                   f"{few} bet(s) quoted by < {min_books} books.")
    ui.kpi_row([
        ("+EV bets", f"{len(bets):,}", f"{settings.min_edge_pct:.1f} ≤ edge ≤ {max_edge:g} pp and EV > 0"),
        ("Total Kelly stake", f"${bets['kelly_stake'].sum():,.0f}", f"{settings.kelly:.2f}× Kelly on ${settings.bankroll:,.0f}"),
        ("Avg edge", f"{bets['edge_pct'].mean():.1f} pp" if len(bets) else "—", None),
        ("Sharp-anchored", f"{(ev['fair_source'] != 'consensus').mean():.0%}", "Share of selections whose fair "
                                                                                 "price comes from a sharp book"),
        ("Selections priced", f"{len(ev):,}", None),
    ])
    ui.edge_table(table, key="scanner_table")
    st.download_button("⬇ Export CSV", table.to_csv(index=False).encode("utf-8"), file_name="ev_bets.csv",
                       mime="text/csv", disabled=table.empty)
    st.markdown("<span class='small-note'>Fair odds are always devigged (sharp book first, consensus fallback); "
                "one-sided markets (e.g. anytime TD “Yes” only) use an assumed 8% margin. Game-model points "
                "may move at most ±3 from the market. Props of players with < 3 qualifying games are blended "
                "toward the devigged market line. Inactive players' props are not priced.</span>",
                unsafe_allow_html=True)
    _scanner_diagnostics(diag)


def _scanner_diagnostics(diag: dict) -> None:
    issues = diag.get("unmatched_players") or diag.get("unsupported_markets") or diag.get("games_without_model")
    if not issues:
        return
    with st.expander("Pricing diagnostics"):
        if diag.get("unmatched_players"):
            st.write(f"**{len(diag['unmatched_players'])} unmatched player names** (skipped, never guessed — "
                     "review via `python -m src.utils.entity_resolution`):")
            st.write(", ".join(diag["unmatched_players"][:60]))
        if diag.get("unsupported_markets"):
            st.write("**Not priced by the model:**", diag["unsupported_markets"])
        if diag.get("games_without_model"):
            st.write("**Games without model inputs:**", diag["games_without_model"])


# ======================================================================================
# Tab 2: player sim explorer
# ======================================================================================
def explorer_tab(version: str, settings: state.Settings) -> None:
    h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
    h1.subheader("Player Sim & Alt-Line Explorer")
    with h2:
        rerun_button(key="rerun_explorer", primary=False, where=st)
    include_past = st.toggle("Include past games", value=False, key="exp_past")
    games = state.cached_explorer_games(version, include_past)
    if games.empty:
        st.info("No simulatable games (need `player_features` rows). Run the feature builders, "
                "or enable *Include past games*.")
        return
    depth = state.depth_json(state.applied_depth_spec())
    c1, c2, c3 = st.columns([2, 2, 1.4])
    gid = c1.selectbox("Game", games["game_id"], key="exp_game",
                       format_func=dict(zip(games["game_id"], games["label"])).get)
    weather = state.game_weather(version, gid)
    wjson = state.weather_json(weather)
    ui.weather_badge(weather)
    try:
        base_sim = state.cached_simulation(version, gid, "{}", settings.n_trials, settings.seed, depth, wjson)
    except ValueError as exc:
        st.warning(str(exc))
        return
    players = base_sim.players
    pool = players[players["position"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    pool["opp"] = (pool["target_share"] + pool["carry_share"] + pool["is_starting_qb"].astype(float)
                   + pool["active"].astype(float))
    pool = pool.sort_values(["team", "opp"], ascending=[True, False])

    def label(r) -> str:
        tag = "" if r.gsis_id in base_sim.stats else (" · backup QB" if r.position == "QB" and not r.is_out
                                                      else " · inactive")
        return f"{r.player_name} ({r.position}, {r.team}){tag}"
    labels = {r.gsis_id: label(r) for r in pool.itertuples()}
    pid = c2.selectbox("Player", pool["gsis_id"], key="exp_player", format_func=labels.get)
    player = pool.set_index("gsis_id").loc[pid].copy()
    player["gsis_id"] = pid
    _team_adjustment_note(base_sim, str(player["team"]), labels)
    if pid not in base_sim.stats:
        why = ("is a backup QB — pick him under **Starting QB / key replacement** in the sidebar"
               if player["position"] == "QB" else "is on the **Inactive / Out** list — remove him there")
        st.info(f"{player['player_name']} {why}, then click **{RERUN_LABEL}**.")
        return

    base_ts, base_cs = float(player["target_share"]), float(player["carry_share"])
    base_snap = float(player["snap_share_ewm5"]) if pd.notna(player["snap_share_ewm5"]) else 0.0
    k = f"{gid}_{pid}"
    st.markdown("**Usage overrides** — the simulator rescales teammates so team totals are unchanged.")
    s1, s2, s3, s4 = st.columns([1, 1, 1, 0.5], vertical_alignment="bottom")
    ts = s1.slider("Target share", 0.0, 0.6, value=round(base_ts, 3), step=0.005, key=f"ts_{k}",
                   help=f"Projected: {base_ts:.1%}")
    cs = s2.slider("Carry share", 0.0, 0.9, value=round(base_cs, 3), step=0.005, key=f"cs_{k}",
                   help=f"Projected: {base_cs:.1%}")
    ss = s3.slider("Snap share", 0.0, 1.0, value=round(base_snap, 3), step=0.01, key=f"ss_{k}",
                   disabled=base_snap <= 0, help=f"Projected: {base_snap:.1%}. Scales target & carry share by "
                                                 "new / projected snap share.")
    s4.button("Reset", key=f"reset_{k}", on_click=_reset_sliders,
              args=(k, round(base_ts, 3), round(base_cs, 3), round(base_snap, 3)))
    overrides = state.usage_overrides(player, ts, cs, ss if base_snap > 0 else base_snap)

    sim = state.cached_simulation(version, gid, json.dumps(overrides, sort_keys=True), settings.n_trials,
                                  settings.seed, depth, wjson) if overrides else base_sim
    stats = [s for s in state.STAT_ORDER if s in sim.stats.get(pid, {})]
    useful = [s for s in stats if float(np.mean(base_sim.stats[pid][s])) > 0.05] or stats
    stat = c3.selectbox("Stat", useful, key=f"exp_stat_{pid}", format_func=lambda s: state.STAT_LABELS.get(s, s))
    x, x0 = sim.get(pid, stat), base_sim.get(pid, stat)

    lines = state.cached_player_lines(version, gid, pid, stat)
    main_line = (float(lines.loc[~lines["is_alternate"], "line"].median()) if (~lines["is_alternate"]).any()
                 else float(np.floor(np.median(x0))) + 0.5)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean", f"{x.mean():.1f}", f"{x.mean() - x0.mean():+.1f} vs projection" if overrides else None)
    m2.metric("Median", f"{np.median(x):.1f}")
    m3.metric("10th–90th pct", f"{np.percentile(x, 10):.0f} – {np.percentile(x, 90):.0f}")
    over, push = state.bt.p_over(x, main_line)
    m4.metric(f"P(over {main_line:g})", f"{over:.1%}",
              f"{100 * (over - state.bt.p_over(x0, main_line)[0]):+.1f} pp" if overrides else None)

    name = str(player["player_name"])
    st.plotly_chart(ui.sim_histogram(x, lines, state.STAT_LABELS.get(stat, stat), name, main_line),
                    width="stretch", key="exp_hist")
    if lines.empty:
        st.caption("No sportsbook lines cached for this player/stat — showing model fair odds at a custom line.")
    if bool(player.get("cold_start", False)):
        st.caption(f"{name} has < {state.gs.COLD_START_GAMES} qualifying games: usage is blended with a positional "
                   "prior, and the +EV scanner also blends his prop probabilities toward the market line.")
    custom = st.number_input("Custom line", value=main_line, step=0.5, key=f"exp_line_{pid}_{stat}")
    probs = state.line_probabilities(x, list(lines["line"]) + [custom])
    if not lines.empty:
        probs = probs.merge(lines, on="line", how="left")
        probs["edge_over_pp"] = 100 * (probs["p_over"] / (1 - probs["p_push"]) - 1 / dv.american_to_decimal(
            probs["Over"].fillna(-1e9).replace(-1e9, np.nan)))
    show = probs.rename(columns={"bookmaker_key": "Book", "market_key": "Market", "line": "Line",
                                 "p_over": "Model P(over)", "p_under": "Model P(under)", "fair_over": "Fair over",
                                 "fair_under": "Fair under", "Over": "Book over", "Under": "Book under",
                                 "edge_over_pp": "Over edge (raw pp)"}).drop(columns=["p_push", "is_alternate"],
                                                                            errors="ignore")
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Model P(over)": st.column_config.NumberColumn(format="percent"),
        "Model P(under)": st.column_config.NumberColumn(format="percent"),
        "Fair over": st.column_config.NumberColumn(format="%+d"), "Fair under": st.column_config.NumberColumn(format="%+d"),
        "Book over": st.column_config.NumberColumn(format="%+d"), "Book under": st.column_config.NumberColumn(format="%+d"),
        "Over edge (raw pp)": st.column_config.NumberColumn(format="%.1f", help="vs the book's vigged price; the "
                                                            "Scanner tab compares against devigged fair odds")})
    if overrides:
        st.caption(f"Overrides applied: `{json.dumps(overrides[pid])}`")


def _team_adjustment_note(sim, team: str, labels: dict) -> None:
    """One line on what the applied depth overrides did to this player's team."""
    adj = sim.adjustments.get(team, {})
    if not adj:
        return
    parts = []
    qb = adj.get("starting_qb")
    if qb:
        parts.append(f"starting QB {labels.get(qb, qb).split(' (')[0]}")
    if abs(adj.get("qb_efficiency", 1.0) - 1) > 1e-9:
        parts.append(f"backup-QB efficiency ×{adj['qb_efficiency']:.2f}")
    if adj.get("missing_starters"):
        parts.append(f"missing {', '.join(adj['missing_starters'])} (×{adj['missing_starter_factor']:.2f})")
    if abs(adj.get("qb_attempt_share", 1.0) - 1) > 1e-9:
        parts.append(f"QB attempt share {adj['qb_attempt_share']:.0%}")
    if abs(adj.get("pass_efficiency", 1.0) - 1) > 1e-9:
        parts.append(f"team YPA ×{adj['pass_efficiency']:.2f}")
    wf = adj.get("weather") or {}
    w_eff = float(wf.get("wind_pass_efficiency", 1.0)) * float(wf.get("wet_cold", 1.0))
    if abs(w_eff - 1) > 1e-9 or abs(float(wf.get("pass_rate", 1.0)) - 1) > 1e-9:
        parts.append(f"weather: pass efficiency ×{w_eff:.2f}, pass rate ×{float(wf.get('pass_rate', 1.0)):.2f}")
    if parts:
        st.caption(f"**{team} depth adjustments:** " + " · ".join(parts))


def _reset_sliders(k: str, ts: float, cs: float, ss: float) -> None:
    st.session_state[f"ts_{k}"], st.session_state[f"cs_{k}"], st.session_state[f"ss_{k}"] = ts, cs, ss


# ======================================================================================
# Tab 3: line shopper
# ======================================================================================
def shopper_tab(version: str, settings: state.Settings) -> None:
    st.subheader("Book Line Shopper")
    odds = state.cached_latest_odds(version, st.session_state.get("shop_started", False))
    if odds.empty:
        st.toggle("Include started games", value=False, key="shop_started")
        st.info("No cached odds. Use **Load cached odds** or **Pull Live Odds** in the sidebar.")
        return
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    events = odds.drop_duplicates("event_id").sort_values("commence_time_utc")
    ev_label = {r.event_id: f"{r.game} · {pd.Timestamp(r.commence_time_utc):%a %b %d %H:%M} UTC"
                for r in events.itertuples()}
    eid = c1.selectbox("Game", events["event_id"], key="shop_event", format_func=ev_label.get)
    shop_gid = events.set_index("event_id")["game_id"].get(eid) if "game_id" in events else None
    if isinstance(shop_gid, str):
        ui.weather_badge(state.game_weather(version, shop_gid))
    markets = sorted(odds.loc[odds["event_id"] == eid, "market_key"].unique(),
                     key=lambda m: (m.startswith("player_"), "alternate" in m, m))
    mk = c2.selectbox("Market", markets, key="shop_market")
    stale = c3.number_input("Stale after (min)", 1, 720, 30, key="shop_stale")
    off = c4.number_input("Off-market (pp)", 0.5, 20.0, 3.0, step=0.5, key="shop_off")
    st.toggle("Include started games", value=False, key="shop_started")

    price, flags, detail = state.line_shop_matrix(odds, eid, mk, settings.devig_method, settings.sharp_priority(),
                                                  stale, off)
    if price.empty:
        st.info("No quotes for this market.")
        return
    if mk.startswith("player_"):
        players = sorted({lbl.rsplit(" Over", 1)[0].rsplit(" Under", 1)[0].rsplit(" Yes", 1)[0].rsplit(" No", 1)[0]
                          for lbl in price.index})
        pick = st.multiselect("Players", players, key=f"shop_players_{mk}", placeholder="All players")
        if pick:
            keep = [lbl for lbl in price.index if any(lbl.startswith(p + " ") for p in pick)]
            price, flags = price.loc[keep], flags.loc[keep]
    n_off = int(flags.drop(columns="Fair (no-vig)").apply(lambda c: c.str.contains("off")).to_numpy().sum())
    n_stale = int(flags.drop(columns="Fair (no-vig)").apply(lambda c: c.str.contains("stale")).to_numpy().sum())
    ui.kpi_row([("Books", str(price.shape[1] - 1), None), ("Selections", str(len(price)), None),
                ("Off-market quotes", str(n_off), f"no-vig ≥ {off:g} pp from fair"),
                ("Stale quotes", str(n_stale), f"updated ≥ {stale} min before the freshest book")])
    ui.flag_legend()
    st.dataframe(ui.line_shop_styler(price, flags), width="stretch", height=min(40 + 35 * len(price), 700))
    with st.expander("Off-market chart & per-book detail"):
        st.plotly_chart(ui.off_market_fig(detail[detail["label"].isin(price.index)], off), width="stretch",
                        key="shop_off_fig")
        st.dataframe(detail, hide_index=True, width="stretch")


# ======================================================================================
# Tab 4: backtest
# ======================================================================================
def backtest_tab(version: str) -> None:
    st.subheader("Backtest & CLV Performance")
    runs = state.cached_backtest_runs(version)
    if runs.empty:
        st.info("No backtest runs yet. Run `python -m src.evaluation.backtest`.")
        return
    run_labels = {r.run_id: f"{pd.Timestamp(r.created_at_utc):%Y-%m-%d %H:%M} · {r.run_id}" for r in runs.itertuples()}
    run_id = st.selectbox("Run", runs["run_id"], key="bt_run", format_func=run_labels.get)
    results, bets = state.cached_backtest(version, run_id)
    bankroll = state.run_bankroll(results)
    placed = bets[bets["is_bet"].astype(bool)] if not bets.empty else bets
    staked = float(placed["stake"].sum()) if len(placed) else 0.0
    profit = float(placed["profit"].sum()) if len(placed) else 0.0
    clv = placed["clv_pct"].dropna() if len(placed) else pd.Series(dtype=float)
    seasons = sorted(results["season"].dropna().astype(int).unique())
    ui.kpi_row([
        ("Seasons", f"{seasons[0]}–{seasons[-1]}" if seasons else "—", "Walk-forward: each season trained on prior seasons only"),
        ("Bets", f"{len(placed):,}", None), ("Staked", f"${staked:,.0f}", None),
        ("Profit", f"{'-' if profit < 0 else '+'}${abs(profit):,.0f}", None), ("ROI", f"{100 * profit / staked:+.2f}%" if staked else "—", None),
        ("Avg CLV", f"{clv.mean():+.2f}%" if len(clv) else "n/a", "Needs raw_odds snapshots (bet vs close)"),
    ])
    if bets.empty:
        st.info("This run has no bet-level rows.")
        return
    curve = state.equity_curve(bets, bankroll)
    st.plotly_chart(ui.equity_curve_fig(curve, bankroll) if not curve.empty else ui.empty_fig("No bets placed"),
                    width="stretch", key="bt_equity")

    c1, c2 = st.columns(2)
    roi = state.roi_by_market(results)
    with c1:
        st.plotly_chart(ui.roi_bar_fig(roi) if not roi.empty else ui.empty_fig("No bets placed"), width="stretch",
                        key="bt_roi")
    with c2:
        cats = sorted(bets["market_category"].unique())
        pick = st.multiselect("Calibration markets", cats, default=cats, key="bt_cal_cats")
        nb = st.slider("Bins", 5, 20, 10, key="bt_bins")
        cal = state.calibration_table(bets, pick, nb)
        st.plotly_chart(ui.calibration_fig(cal) if not cal.empty else ui.empty_fig("No settled predictions"),
                        width="stretch", key="bt_cal")

    st.markdown("##### Market breakdown")
    metrics = results[results["scope"] == "all"][["market_category", "n_predictions", "brier_model", "brier_market",
                                                   "logloss_model", "logloss_market", "n_bets", "roi_pct",
                                                   "flat_roi_pct", "hit_rate", "avg_edge_pct", "avg_clv_pct"]]
    st.dataframe(metrics, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.4f") for c in ("brier_model", "brier_market", "logloss_model",
                                                                  "logloss_market")} | {
        "roi_pct": st.column_config.NumberColumn("ROI %", format="%+.2f"),
        "flat_roi_pct": st.column_config.NumberColumn("Flat ROI %", format="%+.2f"),
        "hit_rate": st.column_config.NumberColumn(format="percent"),
        "avg_edge_pct": st.column_config.NumberColumn("Avg edge pp", format="%.2f"),
        "avg_clv_pct": st.column_config.NumberColumn("Avg CLV %", format="%+.2f")})

    st.markdown("##### Closing Line Value")
    clv_tab = state.clv_summary(bets)
    if clv_tab.empty:
        st.info("No CLV in this run: bets were priced at nflverse closing lines, where bet price = close by "
                "definition. CLV appears once `raw_odds` holds pre-close snapshots (bet at first, close at last).")
    else:
        c3, c4 = st.columns([1, 1.4])
        c3.dataframe(clv_tab, hide_index=True, width="stretch", column_config={
            "avg_clv_pct": st.column_config.NumberColumn(format="%+.2f"),
            "median_clv_pct": st.column_config.NumberColumn(format="%+.2f"),
            "beat_close_pct": st.column_config.NumberColumn(format="%.1f%%")})
        c4.plotly_chart(ui.clv_hist_fig(bets), width="stretch", key="bt_clv")


# ======================================================================================
def password_gate() -> bool:
    """Optional access screen: only shown when APP_PASSWORD is set (env/.env or st.secrets).

    Keeps third parties away from the dashboard and, above all, the credit-spending
    "Pull Live Odds" button. The comparison runs server-side and the flag lives in session_state."""
    if state.is_authenticated():
        return True
    st.title("🏈 NFL +EV Model")
    with st.form("login", border=True):
        pwd = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Enter", type="primary")
    if submitted:
        if state.check_password(pwd):
            st.session_state["authenticated"] = True
            st.session_state.pop("login_password", None)
            st.rerun()
        st.error("Incorrect password.")
    return False


def main() -> None:
    st.set_page_config(page_title="NFL +EV Model", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
    ui.inject_css()
    if not password_gate():
        return
    state.init_session_state()
    if not state.db_path().exists():
        st.error(f"Database not found at `{state.db_path()}`. Run `python -m src.ingestion.nfl_data` first.")
        return
    version = state.db_version()
    settings = sidebar(version)
    t1, t2, t3, t4 = st.tabs(["📈 +EV Scanner", "🎲 Player Sim & Alt Lines", "🛒 Line Shopper", "📊 Backtest & CLV"])
    with t1:
        scanner_tab(version, settings)
    with t2:
        explorer_tab(version, settings)
    with t3:
        shopper_tab(version, settings)
    with t4:
        backtest_tab(version)


if __name__ == "__main__":
    main()
