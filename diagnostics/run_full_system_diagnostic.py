
from __future__ import annotations

import ast
import importlib.util
import json
import platform
import subprocess
import sys
import traceback

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# REPORT CONFIGURATION
# ============================================================

RUN_TIME = datetime.now(timezone.utc)
RUN_ID = RUN_TIME.strftime("%Y%m%dT%H%M%S%fZ")

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "reports"
    / "system_diagnostic"
)

SUMMARY_ROWS = []
DETAIL_ROWS = []
ERRORS = []


def record(section, item, status, message="", **extra):
    SUMMARY_ROWS.append({
        "section": section,
        "item": item,
        "status": status,
        "message": str(message),
        **extra,
    })


def add_error(section, item, error):
    message = f"{type(error).__name__}: {error}"

    ERRORS.append({
        "section": section,
        "item": item,
        "error": message,
    })

    record(section, item, "ERROR", message)


def heading(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def safe_write_reports():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORT_DIR / f"system_summary_{RUN_ID}.csv"
    details_path = REPORT_DIR / f"system_details_{RUN_ID}.csv"
    errors_path = REPORT_DIR / f"system_errors_{RUN_ID}.json"

    pd.DataFrame(SUMMARY_ROWS).to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(DETAIL_ROWS).to_csv(
        details_path,
        index=False,
    )

    with errors_path.open("w", encoding="utf-8") as file:
        json.dump(ERRORS, file, indent=4, default=str)

    return summary_path, details_path, errors_path


# ============================================================
# DERIV BROKER SYMBOL MAPPING
# ============================================================

# These mappings were confirmed against the connected Deriv-Demo
# account. They are used by the diagnostic only.
#
# Production MarketData.resolve_symbol() remains unchanged.

BROKER_SYMBOL_ALIASES = {
    "XAUUSD": [
        "XAUUSD.0",
    ],
    "GBPUSD": [
        "GBPUSD.0",
    ],
    "BTCUSD": [
        "BTCUSD.0",
    ],
    "EURUSD": [
        "EURUSD.0",
    ],
    "US30": [
        "Wall Street 30.0",
    ],
    "NAS100": [
        "US Tech 100.0",
    ],
}


def resolve_diagnostic_symbol(
    logical_symbol,
    available_symbols,
    market_data,
):
    """
    Resolve a configured logical symbol to an available broker symbol.

    Resolution order:
    1. Existing MarketData exact/case-insensitive resolver.
    2. Explicit confirmed Deriv broker aliases.
    """

    resolved = market_data.resolve_symbol(
        logical_symbol,
        available_symbols,
    )

    if resolved is not None:
        return resolved, "EXACT_OR_CASE_INSENSITIVE"

    available_lookup = {
        symbol.upper(): symbol
        for symbol in available_symbols
    }

    candidates = BROKER_SYMBOL_ALIASES.get(
        logical_symbol.upper(),
        [],
    )

    for candidate in candidates:
        resolved = available_lookup.get(candidate.upper())

        if resolved is not None:
            return resolved, "CONFIRMED_DERIV_ALIAS"

    return None, "NOT_FOUND"


# ============================================================
# PROJECT STRUCTURE
# ============================================================

def check_project_structure():
    heading("1. PROJECT STRUCTURE")

    required_paths = [
        "config/settings.yaml",
        "src",
        "src/config.py",
        "src/logger.py",
        "src/market_data.py",
        "src/mt5_connection.py",
        "research",
        "research/historical_data.py",
        "research/historical_pipeline.py",
        "research/historical_storage.py",
        "tests",
        "tests/diagnostics",
        "data",
    ]

    for relative in required_paths:
        path = PROJECT_ROOT / relative
        exists = path.exists()

        message = (
            str(path)
            if exists
            else "Required path not found."
        )

        status = "PASS" if exists else "MISSING"

        record(
            "PROJECT_STRUCTURE",
            relative,
            status,
            message,
        )

        print(f"{status:8} {relative}")

    # .env is optional. The project can use environment variables
    # supplied outside a local .env file.
    env_path = PROJECT_ROOT / ".env"

    record(
        "PROJECT_STRUCTURE",
        ".env",
        "OPTIONAL_PRESENT" if env_path.exists() else "OPTIONAL_ABSENT",
        (
            "Present; contents intentionally not inspected."
            if env_path.exists()
            else "Optional file not present. This is not a failure."
        ),
    )

    print(
        f"{'PRESENT' if env_path.exists() else 'OPTIONAL'} "
        ".env"
    )

    source_dirs = ["src", "research", "tests"]

    for directory in source_dirs:
        base = PROJECT_ROOT / directory

        if not base.exists():
            continue

        files = sorted(base.rglob("*.py"))

        print(f"\n{directory}: {len(files)} Python files")

        for path in files:
            relative = path.relative_to(PROJECT_ROOT)
            print(f"  {relative}")


# ============================================================
# PYTHON ENVIRONMENT AND DEPENDENCIES
# ============================================================

def check_environment():
    heading("2. PYTHON ENVIRONMENT")

    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print("Platform:", platform.platform())
    print("Project root:", PROJECT_ROOT)
    print("Virtual environment active:", sys.prefix != sys.base_prefix)

    record(
        "ENVIRONMENT",
        "python_version",
        "INFO",
        sys.version.split()[0],
    )

    record(
        "ENVIRONMENT",
        "virtual_environment",
        "PASS" if sys.prefix != sys.base_prefix else "WARNING",
        sys.prefix,
    )

    required_dependencies = [
        "MetaTrader5",
        "pandas",
        "numpy",
        "yaml",
        "dotenv",
        "pytest",
        "pyarrow",
    ]

    optional_dependencies = [
        "matplotlib",
        "requests",
        "websocket",
    ]

    for dependency in required_dependencies:
        try:
            available = (
                importlib.util.find_spec(dependency) is not None
            )
        except (ImportError, ValueError):
            available = False

        status = "PASS" if available else "MISSING"

        record(
            "REQUIRED_DEPENDENCIES",
            dependency,
            status,
            "Installed" if available else "Required module not found.",
        )

        print(f"{status:8} {dependency}")

    for dependency in optional_dependencies:
        try:
            available = (
                importlib.util.find_spec(dependency) is not None
            )
        except (ImportError, ValueError):
            available = False

        status = "OPTIONAL_PRESENT" if available else "OPTIONAL_MISSING"

        record(
            "OPTIONAL_DEPENDENCIES",
            dependency,
            status,
            (
                "Installed."
                if available
                else "Not installed. Not treated as a system failure."
            ),
        )

        print(f"{status:18} {dependency}")


# ============================================================
# SOURCE CODE AND IMPLEMENTATION INVENTORY
# ============================================================

def inspect_source_code():
    heading("3. SOURCE CODE AND IMPLEMENTATION INVENTORY")

    directories = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "research",
    ]

    for directory in directories:
        if not directory.exists():
            continue

        for path in sorted(directory.rglob("*.py")):
            relative = str(path.relative_to(PROJECT_ROOT))

            try:
                # utf-8-sig safely handles files with or without a BOM.
                source = path.read_text(encoding="utf-8-sig")

                tree = ast.parse(
                    source,
                    filename=relative,
                )

                classes = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]

                functions = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                ]

                record(
                    "SOURCE_INVENTORY",
                    relative,
                    "PASS",
                    "Python source parsed successfully.",
                    classes=", ".join(classes),
                    functions=", ".join(functions),
                    lines=len(source.splitlines()),
                )

            except Exception as error:
                add_error(
                    "SOURCE_INVENTORY",
                    relative,
                    error,
                )


def inspect_market_structure():
    heading("4. MARKET STRUCTURE IMPLEMENTATION")

    research_dir = PROJECT_ROOT / "research"

    expected_files = [
        "research/swing_detection/swing_models.py",
        "research/swing_detection/trend_models.py",
        "research/swing_detection/atr_calculator.py",
    ]

    for relative in expected_files:
        path = PROJECT_ROOT / relative

        status = "PASS" if path.exists() else "MISSING"

        record(
            "MARKET_STRUCTURE",
            relative,
            status,
            "File exists." if path.exists() else "File not found.",
        )

        print(f"{status:8} {relative}")

    search_terms = [
        "swing",
        "high",
        "low",
        "HH",
        "HL",
        "LH",
        "LL",
        "EQH",
        "EQL",
        "BOS",
        "CHoCH",
        "ATR",
        "trend",
        "structure",
    ]

    if not research_dir.exists():
        return

    for path in sorted(research_dir.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8-sig")

            matches = [
                term
                for term in search_terms
                if term.lower() in source.lower()
            ]

            if matches:
                relative = str(path.relative_to(PROJECT_ROOT))

                record(
                    "MARKET_STRUCTURE",
                    relative,
                    "INFO",
                    "Relevant implementation terms found.",
                    terms=", ".join(matches),
                )

                print(f"\n{relative}")
                print("  Relevant terms:", ", ".join(matches))

        except Exception as error:
            add_error(
                "MARKET_STRUCTURE",
                str(path),
                error,
            )


# ============================================================
# AUTOMATED TEST SUITE
# ============================================================

def run_automated_tests():
    heading("5. AUTOMATED TEST SUITE")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--import-mode=prepend",
        ".\\tests",
    ]

    print(
        "Running:",
        " ".join(command),
    )

    print("Only the main tests directory will be collected.")
    print("Backup directories will not be collected.")
    print("This may take several minutes.")

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        output = (
            (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        test_log = REPORT_DIR / f"pytest_output_{RUN_ID}.txt"

        test_log.write_text(
            output,
            encoding="utf-8",
        )

        print("\nLast 50 lines of test output:")

        for line in output.splitlines()[-50:]:
            print(line)

        status = "PASS" if result.returncode == 0 else "FAIL"

        record(
            "AUTOMATED_TESTS",
            "pytest",
            status,
            f"Exit code: {result.returncode}",
            log_file=str(test_log),
            command="python -m pytest -q --import-mode=prepend .\\tests",
        )

    except subprocess.TimeoutExpired as error:
        add_error(
            "AUTOMATED_TESTS",
            "pytest",
            error,
        )

    except Exception as error:
        add_error(
            "AUTOMATED_TESTS",
            "pytest",
            error,
        )


# ============================================================
# HISTORICAL DATA RECONCILIATION
# ============================================================

TIMEFRAME_SECONDS = {
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "M1": 60,
    "D1": 86400,
}

OHLC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]


def find_dataset(
    output_directory,
    logical_symbol,
    broker_symbol,
    timeframe,
):
    raw_directory = (
        PROJECT_ROOT
        / output_directory
        / "raw"
    )

    candidates = [
        raw_directory
        / logical_symbol
        / f"{logical_symbol}_{timeframe}.parquet",

        raw_directory
        / broker_symbol
        / f"{broker_symbol}_{timeframe}.parquet",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if raw_directory.exists():
        expected_names = {
            f"{logical_symbol}_{timeframe}.parquet".lower(),
            f"{broker_symbol}_{timeframe}.parquet".lower(),
        }

        for candidate in raw_directory.rglob("*.parquet"):
            if candidate.name.lower() in expected_names:
                return candidate

    return None


def normalize_dataset(dataframe):
    df = dataframe.copy()

    if "time" not in df.columns:
        raise ValueError("Dataset has no 'time' column.")

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["time"])

    missing = [
        column
        for column in OHLC_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset missing OHLC columns: {missing}"
        )

    if "is_forming_candle" in df.columns:
        df = df[
            ~df["is_forming_candle"]
            .fillna(False)
            .astype(bool)
        ]

    return (
        df.sort_values("time")
        .reset_index(drop=True)
    )


def compare_historical_dataset(
    collector,
    logical_symbol,
    broker_symbol,
    timeframe,
    dataset_path,
):
    result = {
        "logical_symbol": logical_symbol,
        "broker_symbol": broker_symbol,
        "timeframe": timeframe,
        "dataset_path": str(dataset_path),
    }

    try:
        stored_original = pd.read_parquet(dataset_path)
        stored = normalize_dataset(stored_original)

        timeframe_seconds = TIMEFRAME_SECONDS[timeframe]
        now = datetime.now(timezone.utc)

        # Keep only candles whose closing time has passed.
        stored = stored[
            stored["time"]
            + pd.to_timedelta(
                timeframe_seconds,
                unit="s",
            )
            <= pd.Timestamp(now)
        ].copy()

        stored_rows = len(stored)

        duplicate_rows = int(
            stored["time"].duplicated(keep=False).sum()
        )

        result["stored_rows"] = stored_rows
        result["stored_duplicate_rows"] = duplicate_rows

        if stored.empty:
            result["status"] = "NO_COMPLETED_STORED_CANDLES"
            return result

        if duplicate_rows:
            result["status"] = "REVIEW_DUPLICATE_STORED_TIMESTAMPS"
            return result

        start = stored["time"].iloc[0].to_pydatetime()

        end = (
            stored["time"].iloc[-1]
            + pd.Timedelta(seconds=timeframe_seconds)
        ).to_pydatetime()

        if end > now:
            result["status"] = "REVIEW_END_TIME"
            return result

        broker, metadata = collector.collect_with_metadata(
            broker_symbol,
            timeframe,
            start,
            end,
        )

        if broker is None or broker.empty:
            result["status"] = "NO_BROKER_DATA"
            return result

        broker = normalize_dataset(broker)

        broker_duplicate_rows = int(
            broker["time"].duplicated(keep=False).sum()
        )

        result["broker_rows"] = len(broker)
        result["broker_duplicate_rows"] = broker_duplicate_rows

        if broker_duplicate_rows:
            result["status"] = "REVIEW_DUPLICATE_BROKER_TIMESTAMPS"
            return result

        stored_compare = stored[
            ["time"] + OHLC_COLUMNS
        ].copy()

        broker_compare = broker[
            ["time"] + OHLC_COLUMNS
        ].copy()

        merged = stored_compare.merge(
            broker_compare,
            on="time",
            how="outer",
            suffixes=("_stored", "_broker"),
            indicator=True,
            validate="one_to_one",
        )

        matched = merged[
            merged["_merge"] == "both"
        ].copy()

        stored_only = merged[
            merged["_merge"] == "left_only"
        ]

        broker_only = merged[
            merged["_merge"] == "right_only"
        ]

        changed_mask = pd.Series(
            False,
            index=matched.index,
        )

        for column in OHLC_COLUMNS:
            difference = (
                matched[f"{column}_stored"]
                - matched[f"{column}_broker"]
            ).abs()

            changed_mask |= difference > 1e-8

        changed = matched[changed_mask]
        unchanged = matched[~changed_mask]

        for _, row in changed.iterrows():
            detail = {
                "logical_symbol": logical_symbol,
                "broker_symbol": broker_symbol,
                "timeframe": timeframe,
                "difference_type": "OHLC_CHANGED",
                "time": row["time"],
            }

            for column in OHLC_COLUMNS:
                stored_value = row[f"{column}_stored"]
                broker_value = row[f"{column}_broker"]

                detail[f"{column}_stored"] = stored_value
                detail[f"{column}_broker"] = broker_value
                detail[f"{column}_difference"] = (
                    stored_value - broker_value
                )

            DETAIL_ROWS.append(detail)

        for _, row in stored_only.iterrows():
            DETAIL_ROWS.append({
                "logical_symbol": logical_symbol,
                "broker_symbol": broker_symbol,
                "timeframe": timeframe,
                "difference_type": "STORED_ONLY",
                "time": row["time"],
            })

        for _, row in broker_only.iterrows():
            DETAIL_ROWS.append({
                "logical_symbol": logical_symbol,
                "broker_symbol": broker_symbol,
                "timeframe": timeframe,
                "difference_type": "BROKER_ONLY",
                "time": row["time"],
            })

        result.update({
            "matched_timestamps": len(matched),
            "unchanged_candles": len(unchanged),
            "changed_candles": len(changed),
            "stored_only": len(stored_only),
            "broker_only": len(broker_only),
            "collection_batch_count": metadata.get("batch_count"),
            "status": (
                "PASS"
                if (
                    len(changed) == 0
                    and len(stored_only) == 0
                    and len(broker_only) == 0
                )
                else "REVIEW_DIFFERENCES"
            ),
        })

        return result

    except Exception as error:
        result["status"] = "ERROR"
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )
        return result


def run_market_data_checks(
    market_data,
    broker_symbol,
    logical_symbol,
):
    heading(
        f"MARKET DATA CHECKS: {logical_symbol} -> {broker_symbol}"
    )

    try:
        info = market_data.enable_symbol(broker_symbol)

        record(
            "MARKET_DATA",
            f"{logical_symbol}/symbol_info",
            "PASS",
            "Broker symbol enabled.",
            digits=info.digits,
            point=info.point,
            tick_size=info.trade_tick_size,
            visible=info.visible,
        )

    except Exception as error:
        add_error(
            "MARKET_DATA",
            f"{logical_symbol}/symbol_info",
            error,
        )
        return

    try:
        quote = market_data.get_recent_quote(broker_symbol)

        record(
            "MARKET_DATA",
            f"{logical_symbol}/live_quote",
            "PASS",
            quote.get("message", ""),
            bid=quote.get("bid"),
            ask=quote.get("ask"),
            spread_points=quote.get("spread_points"),
            age_seconds=quote.get("age_seconds"),
        )

        print(
            f"Quote PASS | Bid={quote['bid']} "
            f"Ask={quote['ask']} "
            f"Spread={quote['spread_points']} points"
        )

    except Exception as error:
        add_error(
            "MARKET_DATA",
            f"{logical_symbol}/live_quote",
            error,
        )

    for timeframe in ["H4", "H1", "M15", "M5"]:
        try:
            candles = market_data.get_closed_candles(
                broker_symbol,
                timeframe,
                count=50,
            )

            status = (
                "PASS"
                if not candles.empty
                else "WARNING"
            )

            record(
                "MARKET_DATA",
                f"{logical_symbol}/{timeframe}/closed_candles",
                status,
                f"Retrieved {len(candles)} completed candles.",
                candle_count=len(candles),
            )

            print(
                f"{timeframe}: {status} "
                f"({len(candles)} completed candles)"
            )

        except Exception as error:
            add_error(
                "MARKET_DATA",
                f"{logical_symbol}/{timeframe}/closed_candles",
                error,
            )


def run_historical_reconciliation(
    settings,
    market_data,
    collector,
):
    heading("6. HISTORICAL DATA RECONCILIATION")

    research_cfg = settings.get(
        "research",
        {},
    ).get(
        "historical_data",
        {},
    )

    if not research_cfg.get("enabled", False):
        record(
            "HISTORICAL_DATA",
            "historical_data_enabled",
            "BLOCKED",
            "Historical data collection is disabled in settings.",
        )
        return

    output_directory = research_cfg.get(
        "output_directory",
        "data/historical",
    )

    instruments = settings.get(
        "market",
        {},
    ).get(
        "instruments",
        [],
    )

    configured_timeframes = settings.get(
        "market",
        {},
    ).get(
        "timeframes",
        {},
    )

    available_symbols = market_data.discover_symbols()

    print("Configured instruments:", instruments)
    print(
        "Configured timeframes:",
        list(configured_timeframes.keys()),
    )
    print(
        "Broker symbols discovered:",
        len(available_symbols),
    )

    for logical_symbol in instruments:
        broker_symbol, resolution_method = (
            resolve_diagnostic_symbol(
                logical_symbol,
                available_symbols,
                market_data,
            )
        )

        if broker_symbol is None:
            record(
                "SYMBOL_RESOLUTION",
                logical_symbol,
                "NOT_FOUND",
                "No exact match or confirmed Deriv alias found.",
            )

            print(f"\nNOT FOUND: {logical_symbol}")
            continue

        record(
            "SYMBOL_RESOLUTION",
            logical_symbol,
            "PASS",
            f"Resolved to {broker_symbol}",
            broker_symbol=broker_symbol,
            resolution_method=resolution_method,
        )

        print(
            f"\nSYMBOL PASS: {logical_symbol} -> "
            f"{broker_symbol} ({resolution_method})"
        )

        run_market_data_checks(
            market_data,
            broker_symbol,
            logical_symbol,
        )

        for timeframe in configured_timeframes:
            dataset_path = find_dataset(
                output_directory,
                logical_symbol,
                broker_symbol,
                timeframe,
            )

            if dataset_path is None:
                record(
                    "HISTORICAL_DATA",
                    f"{logical_symbol}/{timeframe}",
                    "NO_DATASET",
                    "No matching stored Parquet dataset found.",
                )

                print(
                    f"NO DATASET: {logical_symbol}/{timeframe}"
                )
                continue

            result = compare_historical_dataset(
                collector,
                logical_symbol,
                broker_symbol,
                timeframe,
                dataset_path,
            )

            status = result.pop(
                "status",
                "UNKNOWN",
            )

            message = result.pop(
                "error",
                "",
            )

            record(
                "HISTORICAL_RECONCILIATION",
                f"{logical_symbol}/{timeframe}",
                status,
                message,
                **result,
            )

            print(
                f"HISTORY {logical_symbol}/{timeframe}: "
                f"{status}"
            )

            for key in [
                "stored_rows",
                "broker_rows",
                "matched_timestamps",
                "unchanged_candles",
                "changed_candles",
                "stored_only",
                "broker_only",
                "stored_duplicate_rows",
                "broker_duplicate_rows",
            ]:
                if key in result:
                    print(f"  {key}: {result[key]}")


# ============================================================
# MAIN
# ============================================================

def main():
    heading("PA-MS TRADING BOT: FULL SYSTEM DIAGNOSTIC")

    print("Run ID:", RUN_ID)
    print("Run time UTC:", RUN_TIME.isoformat())
    print("Project:", PROJECT_ROOT)
    print("Execution mode: READ-ONLY")
    print("Order placement: NOT USED")
    print("Historical dataset overwriting: NOT USED")

    connection = None

    check_project_structure()
    check_environment()
    inspect_source_code()
    inspect_market_structure()

    try:
        from src.config import load_settings
        from src.market_data import MarketData
        from src.mt5_connection import MT5Connection
        from research.historical_data import HistoricalDataCollector

        settings = load_settings()

        execution_enabled = settings.get(
            "trading",
            {},
        ).get(
            "execution_enabled",
            False,
        )

        record(
            "SAFETY",
            "trading_execution_enabled",
            "PASS" if execution_enabled is False else "WARNING",
            f"Configured value: {execution_enabled}",
        )

        if execution_enabled is not False:
            raise RuntimeError(
                "Diagnostic stopped because trading execution "
                "is not explicitly disabled."
            )

        connection = MT5Connection()
        terminal, account = connection.connect()

        print("\nMT5 connection: PASS")
        print("Account server:", account.server)
        print("Terminal connected:", terminal.connected)
        print("Terminal trade allowed:", terminal.trade_allowed)
        print("Account trade allowed:", account.trade_allowed)

        record(
            "MT5_CONNECTION",
            "connection",
            "PASS",
            "MT5 connection established.",
            server=account.server,
            terminal_connected=terminal.connected,
            terminal_trade_allowed=terminal.trade_allowed,
            account_trade_allowed=account.trade_allowed,
        )

        market_data = MarketData()
        collector = HistoricalDataCollector()

        run_historical_reconciliation(
            settings,
            market_data,
            collector,
        )

    except Exception as error:
        add_error(
            "SYSTEM_DIAGNOSTIC",
            "main",
            error,
        )

        print("\nSYSTEM DIAGNOSTIC ERROR:")
        traceback.print_exc()

    finally:
        if connection is not None:
            try:
                connection.disconnect()

                record(
                    "MT5_CONNECTION",
                    "disconnect",
                    "PASS",
                    "MT5 connection closed.",
                )

            except Exception as error:
                add_error(
                    "MT5_CONNECTION",
                    "disconnect",
                    error,
                )

    run_automated_tests()

    heading("7. FINAL DIAGNOSTIC SUMMARY")

    summary = pd.DataFrame(SUMMARY_ROWS)

    if not summary.empty:
        print("\nStatus counts:")
        print(
            summary["status"]
            .value_counts()
            .to_string()
        )

        print("\nErrors and missing required components:")

        problem_rows = summary[
            summary["status"].isin([
                "ERROR",
                "FAIL",
                "MISSING",
            ])
        ]

        if problem_rows.empty:
            print("No ERROR, FAIL, or MISSING entries.")
        else:
            print(
                problem_rows[
                    [
                        "section",
                        "item",
                        "status",
                        "message",
                    ]
                ].to_string(index=False)
            )

    try:
        summary_path, details_path, errors_path = (
            safe_write_reports()
        )

        print("\nREPORT FILES")
        print("Summary:", summary_path)
        print("Details:", details_path)
        print("Errors:", errors_path)

    except Exception:
        print("\nCould not save diagnostic reports.")
        traceback.print_exc()

    heading("DIAGNOSTIC COMPLETE")

    print("Run ID:", RUN_ID)
    print("No trades were placed.")
    print("No historical datasets were overwritten.")
    print("Review the report files before making code changes.")


if __name__ == "__main__":
    main()
