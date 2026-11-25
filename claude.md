# NFL All-Markets & Player Prop Betting Model + Streamlit UI

## Tech Stack
- **Language:** Python 3.11+
- **Data & Storage:** DuckDB (`data/nfl_model.duckdb`), raw cached JSONs (`data/raw_odds/`)
- **Data & ML Libraries:** `nflreadpy`, `httpx`, `pandas`, `polars`, `lightgbm`, `scipy`, `numpy`, `rapidfuzz`, `pydantic`, `python-dotenv`, `pytest`
- **UI Libraries:** `streamlit`, `plotly`

## Directory Structure
```text
nfl_prop_model/
├── CLAUDE.md
├── .env
├── pyproject.toml
├── app.py                      # Main Streamlit dashboard entry point
├── data/
│   ├── raw_odds/               # Timestamped JSON responses from The Odds API
│   └── nfl_model.duckdb        # DuckDB database for PBP, features, odds, and sim outputs
├── src/
│   ├── ingestion/
│   │   ├── odds_client.py      # Async Odds API client with credit guards & caching
│   │   └── nfl_data.py         # nflreadpy ingestion into DuckDB
│   ├── utils/
│   │   └── entity_resolution.py # Fuzzy player name -> gsis_id mapper
│   ├── features/
│   │   ├── team_ratings.py     # Garbage-time filtered EPA, pace, game script features
│   │   └── player_usage.py     # Rolling player usage, efficiency & injury spillover
│   ├── models/
│   │   ├── game_sim.py         # Game script + 10,000-trial Monte Carlo player simulator
│   │   └── td_model.py         # Calibrated LightGBM classifier for TD props
│   ├── evaluation/
│   │   ├── devig.py            # Multiplicative & Shin devigging (sharp book anchored)
│   │   └── backtest.py         # Walk-forward validation, CLV & fractional Kelly sizing
│   └── app/
│       ├── state.py            # Streamlit session_state & DuckDB query caching
│       └── components.py       # Plotly charts, edge tables, line shopping grids
└── tests/
```

## Non-Negotiable Rules (NEVER VIOLATE)
1. **Strict API Credit Protection:** NEVER make live HTTP calls to `://the-odds-api.com` inside `pytest` or automated scripts unless `--live` is explicitly passed via CLI or the user clicks the explicit `"Pull Live Odds"` button in Streamlit. Always inspect `data/raw_odds/` cache first. Log `x-requests-remaining` and `x-requests-last` after every live call.
2. **Streamlit Rerun Safety:** Streamlit re-runs `app.py` from top to bottom on every widget interaction. NEVER place live API calls outside an explicit `st.button` callback, and cache DuckDB reads and simulation outputs in `st.session_state` or `@st.cache_data`.
3. **Zero Data Leakage:** All feature transformations MUST enforce strict point-in-time filtering (`game_date < target_game_date`). Never leak full-season stats or future games into historical training rows.
4. **Entity Resolution:** Always map player names from The Odds API through `src/utils/entity_resolution.py` to `nflverse` `gsis_id` before joining odds to player features.
5. **Devigging Standard:** Never compare model probabilities to raw sportsbook implied odds. Always devig using multiplicative or Shin devigging (`src/evaluation/devig.py`), prioritizing sharp books (`pinnacle`, `circa`, `bookmaker`) when present in the payload.

## Workflow Standard
- After building each module, write unit tests in `tests/` and run `pytest` to confirm everything passes before reporting completion.
