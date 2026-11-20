# PA-MS Trading Bot --- Project Handoff & Pipeline Backup

**Snapshot date:** 20 November 2026\
**Project:** PA-MS Trading Bot (Price Action + Market Structure)\
**Purpose:** Compact, verifiable handoff for Claude or another
development assistant.

## 1. Project objective

Build a Windows-based, Python + MetaTrader 5 semi-automated trading
system for Deriv.

**Intended workflow:** collect and validate market data → analyse
multi-timeframe structure → qualify setups → calculate risk → present
trade for manual approval → execute only after approval → monitor/manage
approved positions → journal and evaluate performance.

**Execution policy:** Demo-first. No live trading and no order placement
without explicit user approval. The user must retain manual control over
trade approval.

## 2. Trading configuration

-   Broker: Deriv
-   Initial account: Deriv-Demo, USD, leverage 1:1000 (historically
    verified)
-   Platform: MetaTrader 5
-   Language: Python
-   Instruments and exact MT5 symbols:
    -   XAUUSD: `XAUUSD.0`
    -   GBPUSD: `GBPUSD.0`
    -   BTCUSD: `BTCUSD.0`
    -   EURUSD: `EURUSD.0`
    -   US30: `Wall Street 30.0`
    -   NAS100: `US Tech 100.0`
-   Analysis timeframes: H4 bias → H1 structure → M15 setup → M5
    confirmation.
-   Recorded risk preferences (to be implemented and revalidated): 1%
    risk/trade personal account; 0.5% prop account; maximum 2
    trades/day; weekly loss limits 4% personal / 2% prop; target 3R; no
    Monday trading; focus on London and New York sessions.
-   Position sizing must use live broker symbol specifications and MT5
    profit/margin calculations, not hard-coded pip/contract assumptions.

## 3. Recorded implementation and validation

The following is a historical project record, not a claim that every
component is present in the currently open workspace.

### Foundation

-   Python virtual environment and dependencies established.
-   MT5 connection and Deriv-Demo authentication verified.
-   Trading permissions previously verified as enabled.
-   161 broker instruments discovered.
-   Six target instruments and exact broker symbols verified.
-   Contract sizes historically recorded: forex 100,000; XAUUSD 100;
    BTCUSD 1; indices 1.
-   Initial market-data retrieval tested across six instruments and
    H4/H1/M15/M5.

### Data and market structure

-   Historical candle pipeline, validation/storage and data-integrity
    work implemented.
-   Swing detection and HH/HL/LH/LL/EQH/EQL classification implemented.
-   BOS and CHoCH detection implemented.
-   Protected swing/level registry and structure lifecycle work
    implemented.
-   Historical tests and replay work exist in the broader project
    history.

### Test history

-   Historical checkpoint on 17 November 2026 recorded 364 tests and 14
    subtests passing.
-   An earlier 10 November checkpoint recorded 376 tests and 14 subtests
    passing.
-   These are historical reported results. They must not be represented
    as the current test count until the active repository is inspected
    and the tests are rerun.

## 4. Known outstanding technical issue: MT5 Bid/Ask discrepancy

Diagnostics on 2 November 2026 found: - `mt5.symbol_info_tick()`
returned identical Bid and Ask for all six instruments. - MT5 Market
Watch screenshot also displayed identical Bid and Ask. - Python
`copy_ticks_range()` and native MQL5 `CopyTicks()` returned a positive
one-point Bid/Ask spread. - The latest raw tick and `SymbolInfoTick()`
timestamps matched; the snapshot Bid appeared to duplicate Ask. - The
underlying cause was not established.

**Current engineering direction:** use raw tick history as the candidate
source for Bid/Ask, with freshness, validity, spread and timestamp
checks. Do not treat the discrepancy as fully resolved until the current
terminal/feed behaviour and quote values are independently validated. Do
not use the suspect snapshot quote for strategy or position sizing.

## 5. Last recorded replay-validation work

The 17 November checkpoint identified historical strategy replay as the
next major validation task.

Recorded files: - `research\historical_strategy_replay.py` -
`tests\test_historical_strategy_replay_recovered.py`

Recorded issue: - M5 detector expected exact five-minute continuity. -
Real market sessions can contain legitimate gaps. - Fix must accommodate
valid session gaps without weakening data-integrity checks or allowing
future-data leakage.

Replay must validate H4/H1/M15/M5 synchronization, incomplete/future
candle filtering, and decisions using only information available at the
simulated decision time.

## 6. Current workspace discrepancy --- MUST VERIFY

A VS Code screenshot in the 20 November 2026 conversation showed this
workspace:

``` text
PA_MS_Trading_Bot_Phase1/
├── config/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── market_data.py
│   ├── mt5_connection.py
│   └── phase1_check.py
├── .env.example
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

This is not the full structure reflected in the 17 November replay
checkpoint. It may be a different/older copy or workspace. Do not assume
it is the latest complete project.

Previously recorded paths: - Phase 1:
`C:\Users\tembo\Documents\Forex Folder\PA_MS_Trading_Bot_Phase1` - Phase
2: `C:\Users\tembo\Documents\Forex Folder\PA_MS_Trading_Bot_Phase 2` - A
Phase 1 backup was recorded as
`PA_MS_Trading_Bot_Phase1_VALIDATED_20261116`.

**The active Phase 1/Phase 2 paths, backup contents, branches and
current repository state have not yet been reconciled.**

## 7. Current status by workstream

  ------------------------------------------------------------------------
  Workstream              Recorded status         Verification required
  ----------------------- ----------------------- ------------------------
  MT5 / Deriv connection  Previously validated    Confirm in active
                                                  workspace

  Instrument mapping      Previously validated    Confirm
                                                  symbols/specifications

  Historical data         Implemented             Inspect and rerun tests
  pipeline                historically            

  Data validation/storage Implemented             Inspect and rerun tests
                          historically            

  Swing / structure       Implemented             Inspect and rerun tests
  classification          historically            

  BOS / CHoCH             Implemented             Inspect and rerun tests
                          historically            

  Protected-level         Implemented             Inspect and rerun tests
  registry                historically            

  Historical replay       In progress             Resolve M5 session-gap
                                                  issue; rerun replay
                                                  suite

  Strategy qualification  Not yet confirmed       Define and test explicit
  / entry rules           complete                rules

  Risk engine             Pending                 Design and validate

  Manual approval /       Pending                 Keep execution disabled
  execution                                       until authorized

  Trade management        Pending                 Design and test

  Performance evaluation  Pending                 Backtest with realistic
                                                  costs

  Demo deployment         Pending                 Only after prior gates
                                                  pass
  ------------------------------------------------------------------------

## 8. Roadmap and completion gates

1.  **Reconcile project versions:** identify active Phase 1 and Phase 2
    folders, backup, Git status and latest validated source.
2.  **Restore test baseline:** run tests from the active project; record
    exact command, pass/fail counts and failures.
3.  **Complete replay validation:** resolve M5 session gaps, verify no
    look-ahead/future-data leakage, validate multi-timeframe alignment.
4.  **Complete market-data validation:** integrate raw tick retrieval
    into the correct module; validate all six symbols, timestamps,
    freshness, spread and quote rejection.
5.  **Validate market structure:** test swings, HH/HL/LH/LL, EQH/EQL,
    BOS, CHoCH and protected levels through replay.
6.  **Implement strategy qualification:** formalize H4 bias, H1
    structure, M15 setup, M5 confirmation, invalidation and rejection
    reasons.
7.  **Implement risk management:** position sizing, SL/TP, R-multiple,
    daily/weekly limits, exposure, spread/slippage limits using broker
    calculations.
8.  **Implement manual approval and demo execution:** proposal →
    explicit user approval → order submission → execution verification.
    No unattended order placement.
9.  **Implement trade management:** position monitoring, SL/TP
    verification, optional pre-defined break-even/trailing rules,
    restart recovery and journaling.
10. **Backtest and evaluate:** expectancy, profit factor, drawdown, win
    rate, average R, costs, instrument/session/setup breakdown and
    robustness.
11. **Controlled demo deployment:** monitor reliability, safety
    controls, connection recovery and reporting before considering any
    further deployment.

## 9. Required working method and safeguards

-   Work in the user's VS Code integrated PowerShell terminal on
    Windows.
-   Give beginner-friendly, one-step-at-a-time instructions.
-   Clearly distinguish PowerShell commands from Python source code.
-   For code edits, prefer complete-file replacements; preserve existing
    behaviour and avoid unnecessary refactoring.
-   Inspect the actual file and its callers before changing interfaces.
-   Do not ask the user to paste large files by default if
    repository/file tools can inspect them; ask only for targeted
    missing output when needed.
-   Do not create duplicate project folders or modules without first
    checking the active architecture.
-   Never claim a test passed unless it was run against the current
    working copy and its output was inspected.
-   Keep secrets, account credentials and `.env` contents out of reports
    and prompts.
-   Keep real execution disabled. Do not call `order_send()` or place
    trades unless the user explicitly authorizes a specific, reviewed
    demo test.
-   Preserve backups and use version control/checkpoints before risky
    changes.

## 10. First actions for Claude

Do not immediately write or replace code. First:

1.  Ask the user to open the intended project in VS Code.
2.  Establish the actual working directory (`Get-Location`).
3.  Inspect top-level folders and files (`Get-ChildItem -Force`).
4.  Check Git status and branch if Git is initialized.
5.  Identify Phase 1, Phase 2 and the recorded validated backup; compare
    their contents without overwriting anything.
6.  Locate the replay engine and replay tests named above.
7.  Run the existing test suite using the project's documented
    environment and record the exact baseline.
8.  Produce a concise discrepancy report: what exists, what is missing,
    what differs from this handoff, and the safest next task.
9.  Wait for user confirmation before major restructuring or destructive
    edits.

## 11. Definition of done

The project is not complete merely because it connects to MT5 or detects
BOS/CHoCH. Completion requires validated data, reproducible strategy
decisions, risk controls, explicit human approval, verified demo
execution, robust position management, historical performance evidence
and safe failure recovery.

**Handoff principle:** Treat this document as a map of recorded project
history and intended pipeline. The repository and fresh test output are
the source of truth for the current implementation.
