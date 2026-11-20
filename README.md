# PA-MS Trading Bot

A **read-only** MetaTrader 5 research bot (Deriv demo account) that studies
price-action market structure. It does not place orders:
`trading.execution_enabled` is `false` in `config/settings.yaml`.

## What it does

| Layer | Code | Purpose |
|---|---|---|
| Connection and data | `src/` | MT5 connection, closed-candle retrieval, candle validation |
| Swing / structure engine | `research/swing_detection/` | Fractals, ATR, trend, BOS, CHoCH, protected levels, H4/H1 alignment |
| Strategy | `strategy/` | M15 retest/rejection setups, M5 confirmation, multi-timeframe orchestration |
| Research | `research/` | Historical collection, storage, validation and strategy replay |
| Tests | `tests/` | Offline unit tests plus `verify_*` scripts that need a live MT5 terminal |

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env      # then fill in MT5 details if needed
```

Open the MT5 terminal and log in to the Deriv demo account first.
Credentials come from `.env` (never committed). Leave them blank to use the
account already logged in to the terminal.

## Run

```powershell
python main.py        # Phase 1 connectivity + candle validation check
pytest                # offline test suite (no MT5 needed)
```

`main.py` exits with code 0 if every check passed or warned, and 1 if
any check failed.

`tests/verify_*.py` and `tests/diagnostics/*` need a live terminal and are
excluded from normal `pytest` runs.

## Layout

```
config/       settings.yaml (instruments, timeframes, strategy parameters)
data/         stored historical candles (parquet) and event reports
reports/      validation reports, exports and diagnostic results
archive/      old copies and backups kept for reference, not used by the code
```

## Safety notes

- Demo only: recovery and collection scripts refuse to run unless the
  server name contains `Deriv-Demo`.
- Never commit `.env`, virtual environments or log files (see `.gitignore`).
