"""Map Odds API player names to nflverse ``gsis_id`` / team / position.

Pipeline
--------
1. ``normalize_name`` – accents stripped, lowercased, punctuation removed, initials joined
   ("A. J." -> "aj"), generational suffixes dropped ("Kenneth Walker III" -> "kenneth walker"),
   and common nicknames canonicalised ("Matt" -> "matthew").
2. Cached lookup in DuckDB ``player_id_map`` (keyed by normalized name + team; team ``'*'`` = any).
3. Candidate pool = latest roster entry per player (rosters for the last two seasons, plus the
   ``players`` directory as a fallback), restricted to the two teams in the event when known.
4. rapidfuzz ``token_sort_ratio`` against every alias of every candidate. The market's implied
   position (pass -> QB, tackles -> defense, ...) breaks ties.
5. Score >= ``threshold`` (90) and unambiguous and on one of the event teams -> written to
   ``player_id_map``. Everything else -> ``data/unmatched_players.csv`` for manual review.
   Fill in ``approved_gsis_id`` and run ``--import-reviewed`` to promote rows into the map.

Usage
-----
    python -m src.utils.entity_resolution                     # resolve every player name in raw_odds
    python -m src.utils.entity_resolution --import-reviewed   # load approved rows from the CSV
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"
DEFAULT_UNMATCHED_PATH = PROJECT_ROOT / "data" / "unmatched_players.csv"
MATCH_THRESHOLD = 90.0
AMBIGUITY_MARGIN = 2.0
ANY_TEAM = "*"

# Odds API full team names -> nflverse abbreviations.
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
_ABBRS = set(TEAM_NAME_TO_ABBR.values())
_ABBR_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
NICKNAMES: dict[str, str] = {
    "matt": "matthew", "mike": "michael", "chris": "christopher", "nick": "nicholas", "josh": "joshua",
    "will": "william", "bill": "william", "tony": "anthony", "gabe": "gabriel", "cam": "cameron",
    "zach": "zachary", "zack": "zachary", "jake": "jacob", "dan": "daniel", "danny": "daniel",
    "ben": "benjamin", "sam": "samuel", "tom": "thomas", "rob": "robert", "bob": "robert", "bobby": "robert",
    "jon": "jonathan", "alex": "alexander", "joe": "joseph", "dave": "david", "pat": "patrick",
    "ken": "kenneth", "kenny": "kenneth", "jeff": "jeffrey", "greg": "gregory", "steve": "steven",
    "tim": "timothy", "andy": "andrew", "drew": "andrew", "rick": "richard", "jim": "james", "jimmy": "james",
    "nate": "nathan", "ed": "edward", "eddie": "edward", "fred": "frederick", "hank": "henry",
}

QB = {"QB"}
SKILL = {"QB", "RB", "FB", "WR", "TE"}
DEFENSE = {"LB", "ILB", "OLB", "MLB", "DL", "DE", "DT", "NT", "EDGE", "DB", "CB", "S", "SS", "FS", "SAF"}
KICKER = {"K", "PK"}


def market_positions(market: str | None) -> set[str] | None:
    """Positions plausibly attached to an Odds API player market (``None`` = no hint)."""
    if not market:
        return None
    if market.startswith("player_pass"):
        return QB
    if market.startswith("player_rush"):
        return {"QB", "RB", "FB", "WR", "TE"}
    if market.startswith("player_reception"):
        return {"RB", "FB", "WR", "TE"}
    if market in {"player_kicking_points", "player_field_goals"}:
        return KICKER
    if market in {"player_tackles_assists", "player_sacks"} or market.startswith("player_tackles") \
            or market.startswith("player_sacks"):
        return DEFENSE
    if market in {"player_anytime_td", "player_1st_td", "player_last_td"}:
        return SKILL
    return None


def team_to_abbr(team: str | None) -> str | None:
    if not team:
        return None
    t = team.strip()
    if t in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[t]
    up = t.upper()
    up = _ABBR_ALIASES.get(up, up)
    return up if up in _ABBRS else None


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("’", "").replace("'", "").replace("`", "")
    s = re.sub(r"\b([a-z])\.\s*(?=[a-z]\.)", r"\1", s)   # "a.j." / "a. j." -> "aj."
    s = re.sub(r"[.\-_,]", " ", s)
    joined = [t for t in re.sub(r"[^a-z0-9 ]", " ", s).split() if t]
    while len(joined) > 2 and joined[-1] in SUFFIXES:
        joined.pop()
    if joined and joined[0] in NICKNAMES and len(joined) >= 2:
        joined[0] = NICKNAMES[joined[0]]
    return " ".join(joined)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    gsis_id: str
    name: str
    team: str | None
    position: str | None
    aliases: tuple[str, ...]


@dataclass
class MatchResult:
    odds_name: str
    odds_name_norm: str
    context_teams: tuple[str, ...]
    market: str | None
    gsis_id: str | None = None
    matched_name: str | None = None
    team: str | None = None
    position: str | None = None
    score: float = 0.0
    method: str = "none"
    status: str = "review"          # "matched" | "review"
    reason: str = ""
    alternatives: list[tuple[str, str, str | None, str | None, float]] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.status == "matched"


PLAYER_ID_MAP_DDL = """
CREATE TABLE IF NOT EXISTS player_id_map (
    odds_name_norm VARCHAR NOT NULL,
    team           VARCHAR NOT NULL,      -- nflverse abbr, or '*' when valid for any team
    odds_name      VARCHAR,
    gsis_id        VARCHAR NOT NULL,
    matched_name   VARCHAR,
    position       VARCHAR,
    score          DOUBLE,
    match_method   VARCHAR,               -- exact | fuzzy | manual
    updated_at_utc TIMESTAMP,
    PRIMARY KEY (odds_name_norm, team)
)
"""

UNMATCHED_COLUMNS = [
    "odds_name", "context_teams", "market", "best_score", "reason", "suggested_gsis_id", "suggested_name",
    "suggested_team", "suggested_position", "alternatives", "approved_gsis_id",
]


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchall()[0][0])


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()}


def load_candidates(con: duckdb.DuckDBPyConnection, season: int | None = None) -> list[Candidate]:
    """Latest roster entry per gsis_id for ``season`` and the season before (falling back to the
    ``players`` directory for anyone not on a recent roster)."""
    rows: dict[str, dict] = {}
    if _table_exists(con, "rosters"):
        cols = _columns(con, "rosters")
        season = season or con.execute("SELECT MAX(season) FROM rosters").fetchall()[0][0]
        week_order = "week DESC NULLS LAST," if "week" in cols else ""
        football = "football_name" if "football_name" in cols else "NULL"
        first = "first_name" if "first_name" in cols else "NULL"
        last = "last_name" if "last_name" in cols else "NULL"
        df = con.execute(f"""
            SELECT gsis_id, full_name, team, position, {football} AS football_name,
                   {first} AS first_name, {last} AS last_name
            FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY gsis_id ORDER BY season DESC, {week_order} team) AS rn
                  FROM rosters WHERE gsis_id IS NOT NULL AND full_name IS NOT NULL AND season >= ?)
            WHERE rn = 1""", [int(season) - 1]).df()
        for r in df.to_dict("records"):
            rows[r["gsis_id"]] = r
    if _table_exists(con, "players"):
        cols = _columns(con, "players")
        recent = "AND last_season >= ?" if season and "last_season" in cols else ""
        params = [int(season) - 1] if recent else []
        col = lambda c: c if c in cols else "NULL"  # noqa: E731 - tolerate older/partial directories
        df = con.execute(f"""
            SELECT gsis_id, display_name AS full_name, {col("latest_team")} AS team, {col("position")} AS position,
                   {col("football_name")} AS football_name, {col("first_name")} AS first_name,
                   {col("last_name")} AS last_name
            FROM players WHERE gsis_id IS NOT NULL AND display_name IS NOT NULL {recent}""", params).df()
        for r in df.to_dict("records"):
            rows.setdefault(r["gsis_id"], r)

    out = []
    for r in rows.values():
        aliases = {normalize_name(r["full_name"])}
        fb, ln = r.get("football_name"), r.get("last_name")
        if isinstance(fb, str) and isinstance(ln, str) and fb and ln:
            aliases.add(normalize_name(f"{fb} {ln}"))
        team = r.get("team") if isinstance(r.get("team"), str) else None
        pos = r.get("position") if isinstance(r.get("position"), str) else None
        out.append(Candidate(r["gsis_id"], r["full_name"], team, pos, tuple(a for a in aliases if a)))
    return out


# --------------------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------------------
class PlayerResolver:
    def __init__(self, con: duckdb.DuckDBPyConnection, candidates: Sequence[Candidate] | None = None,
                 threshold: float = MATCH_THRESHOLD, season: int | None = None):
        self.con = con
        self.threshold = threshold
        con.execute(PLAYER_ID_MAP_DDL)
        self.candidates = list(candidates) if candidates is not None else load_candidates(con, season)
        if not self.candidates:
            logger.warning("No candidates loaded; run src.ingestion.nfl_data first (rosters/players).")
        self._by_id = {c.gsis_id: c for c in self.candidates}

    # -- cache ---------------------------------------------------------------------------
    def lookup_cache(self, norm: str, teams: Sequence[str]) -> MatchResult | None:
        rows = self.con.execute(
            "SELECT team, gsis_id, matched_name, position, score, match_method, odds_name FROM player_id_map "
            "WHERE odds_name_norm = ?", [norm]).fetchall()
        if not rows:
            return None
        if teams:
            hits = [r for r in rows if r[0] in teams] or [r for r in rows if r[0] == ANY_TEAM]
        else:
            hits = rows if len({r[1] for r in rows}) == 1 else []
        if len({r[1] for r in hits}) != 1:
            return None
        team, gsis, name, pos, score, method, odds_name = hits[0]
        return MatchResult(odds_name=odds_name, odds_name_norm=norm, context_teams=tuple(teams), market=None,
                           gsis_id=gsis, matched_name=name, team=None if team == ANY_TEAM else team,
                           position=pos, score=score, method=f"cache:{method}", status="matched")

    def save(self, results: Iterable[MatchResult], method_override: str | None = None) -> int:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = [(r.odds_name_norm, r.team or ANY_TEAM, r.odds_name, r.gsis_id, r.matched_name, r.position,
                 float(r.score), method_override or r.method, now)
                for r in results if r.matched and r.gsis_id and not r.method.startswith("cache")]
        if rows:
            self.con.executemany("INSERT OR REPLACE INTO player_id_map VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    # -- matching ------------------------------------------------------------------------
    def _rank(self, norm: str, pool: Sequence[Candidate], positions: set[str] | None):
        choices: list[str] = []
        owners: list[Candidate] = []
        for c in pool:
            for a in c.aliases:
                choices.append(a)
                owners.append(c)
        if not choices:
            return []
        hits = process.extract(norm, choices, scorer=fuzz.token_sort_ratio, limit=max(10, len(choices) // 50))
        best: dict[str, tuple[float, Candidate]] = {}
        for _, score, idx in hits:
            c = owners[idx]
            if c.gsis_id not in best or score > best[c.gsis_id][0]:
                best[c.gsis_id] = (float(score), c)
        fits = lambda c: positions is None or (c.position or "").upper() in positions  # noqa: E731
        return sorted(best.values(), key=lambda t: (t[0], fits(t[1])), reverse=True)

    def resolve(self, name: str, teams: Sequence[str] | None = None, market: str | None = None,
                use_cache: bool = True) -> MatchResult:
        norm = normalize_name(name)
        team_abbrs = tuple(sorted({a for a in (team_to_abbr(t) for t in (teams or [])) if a}))
        result = MatchResult(odds_name=name, odds_name_norm=norm, context_teams=team_abbrs, market=market)
        if not norm:
            result.reason = "empty_name"
            return result
        if use_cache and (cached := self.lookup_cache(norm, team_abbrs)) is not None:
            cached.odds_name, cached.market = name, market
            return cached

        positions = market_positions(market)
        in_teams = [c for c in self.candidates if c.team in team_abbrs] if team_abbrs else list(self.candidates)
        ranked = self._rank(norm, in_teams, positions)
        off_team = False
        if team_abbrs and (not ranked or ranked[0][0] < self.threshold):
            ranked_all = self._rank(norm, self.candidates, positions)
            if ranked_all and (not ranked or ranked_all[0][0] > ranked[0][0]):
                ranked, off_team = ranked_all, True
        if not ranked:
            result.reason = "no_candidates"
            return result

        top_score, top = ranked[0]
        result.alternatives = [(c.gsis_id, c.name, c.team, c.position, s) for s, c in ranked[:3]]
        result.gsis_id, result.matched_name, result.team, result.position = top.gsis_id, top.name, top.team, top.position
        result.score = top_score
        result.method = "exact" if top_score >= 100 else "fuzzy"

        close = [c for s, c in ranked if top_score - s <= AMBIGUITY_MARGIN]
        if len(close) > 1 and positions is not None:
            fitting = [c for c in close if (c.position or "").upper() in positions]
            if len(fitting) == 1 and fitting[0].gsis_id == top.gsis_id:
                close = fitting
        if top_score < self.threshold:
            result.reason = "low_score"
        elif len(close) > 1:
            result.reason = "ambiguous:" + ",".join(f"{c.name}({c.team})" for c in close)
        elif off_team:
            result.reason = f"team_mismatch: candidate on {top.team}, event teams {','.join(team_abbrs)}"
        else:
            result.status = "matched"
        return result

    def resolve_many(self, records: Iterable[tuple[str, Sequence[str] | None, str | None]]) -> list[MatchResult]:
        return [self.resolve(n, t, m) for n, t, m in records]

    # -- manual review loop --------------------------------------------------------------
    def export_unmatched(self, results: Iterable[MatchResult], path: Path = DEFAULT_UNMATCHED_PATH) -> int:
        """Write every non-matched result to CSV, preserving any ``approved_gsis_id`` already
        entered for the same (odds_name, context_teams)."""
        path = Path(path)
        records = []
        for r in results:
            if r.matched:
                continue
            records.append({
                "odds_name": r.odds_name, "context_teams": "|".join(r.context_teams), "market": r.market or "",
                "best_score": round(r.score, 1), "reason": r.reason, "suggested_gsis_id": r.gsis_id or "",
                "suggested_name": r.matched_name or "", "suggested_team": r.team or "",
                "suggested_position": r.position or "",
                "alternatives": "; ".join(f"{n} ({t},{p}) {g} [{s:.0f}]" for g, n, t, p, s in r.alternatives),
                "approved_gsis_id": "",
            })
        df = pd.DataFrame(records, columns=UNMATCHED_COLUMNS).drop_duplicates(["odds_name", "context_teams"])
        if path.exists():
            prev = pd.read_csv(path, dtype=str, keep_default_na=False)
            if {"odds_name", "context_teams", "approved_gsis_id"} <= set(prev.columns):
                approved = prev[prev["approved_gsis_id"].str.strip() != ""]
                keep = approved.set_index(["odds_name", "context_teams"])["approved_gsis_id"].to_dict()
                df["approved_gsis_id"] = [keep.get((a, b), "") for a, b in zip(df["odds_name"], df["context_teams"])]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["best_score", "odds_name"], ascending=[False, True]).to_csv(path, index=False)
        logger.info("Wrote %d players needing review -> %s", len(df), path)
        return len(df)

    def import_reviewed(self, path: Path = DEFAULT_UNMATCHED_PATH) -> int:
        """Promote CSV rows with a filled ``approved_gsis_id`` into ``player_id_map`` (method=manual)."""
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        promoted = []
        for row in df.to_dict("records"):
            gsis = (row.get("approved_gsis_id") or "").strip()
            if not gsis:
                continue
            cand = self._by_id.get(gsis)
            if cand is None:
                logger.warning("approved_gsis_id %s for %r not found in candidates; skipping", gsis, row["odds_name"])
                continue
            context = tuple(t for t in (row.get("context_teams") or "").split("|") if t)
            team = cand.team if cand.team in context else ANY_TEAM
            promoted.append(MatchResult(
                odds_name=row["odds_name"], odds_name_norm=normalize_name(row["odds_name"]), context_teams=context,
                market=row.get("market") or None, gsis_id=gsis, matched_name=cand.name,
                team=None if team == ANY_TEAM else team, position=cand.position, score=100.0,
                method="manual", status="matched"))
        n = self.save(promoted)
        logger.info("Imported %d manually reviewed mappings", n)
        return n


# --------------------------------------------------------------------------------------
# raw_odds integration
# --------------------------------------------------------------------------------------
def odds_player_records(con: duckdb.DuckDBPyConnection) -> list[tuple[str, tuple[str, str], str | None]]:
    """Distinct (player name, (home, away), most informative market) from ``raw_odds``."""
    if not _table_exists(con, "raw_odds"):
        return []
    rows = con.execute("""
        SELECT outcome_description, home_team, away_team, LIST(DISTINCT market_key ORDER BY market_key)
        FROM raw_odds
        WHERE is_player_prop AND outcome_description IS NOT NULL AND outcome_description <> ''
        GROUP BY 1, 2, 3 ORDER BY 1""").fetchall()
    out = []
    for name, home, away, markets in rows:
        # Prefer a market with a narrow position hint (pass/kick/defense) over TD markets.
        hinted = sorted(markets, key=lambda m: len(market_positions(m) or SKILL | DEFENSE))
        out.append((name, (home, away), hinted[0] if hinted else None))
    return out


def resolve_odds_players(con: duckdb.DuckDBPyConnection, unmatched_path: Path = DEFAULT_UNMATCHED_PATH,
                         threshold: float = MATCH_THRESHOLD) -> list[MatchResult]:
    resolver = PlayerResolver(con, threshold=threshold)
    results = resolver.resolve_many(odds_player_records(con))
    saved = resolver.save(results)
    n_review = resolver.export_unmatched(results, unmatched_path)
    logger.info("Resolved %d players: %d newly cached, %d from cache, %d need review",
                len(results), saved, sum(r.method.startswith("cache") for r in results), n_review)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resolve Odds API player names to nflverse gsis_id.")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--unmatched-path", type=Path, default=DEFAULT_UNMATCHED_PATH)
    p.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    p.add_argument("--import-reviewed", nargs="?", const=str(DEFAULT_UNMATCHED_PATH), default=None,
                   help="Import rows with approved_gsis_id filled from the review CSV.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with duckdb.connect(str(args.db_path)) as con:
        if args.import_reviewed:
            PlayerResolver(con, threshold=args.threshold).import_reviewed(Path(args.import_reviewed))
        else:
            resolve_odds_players(con, args.unmatched_path, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
