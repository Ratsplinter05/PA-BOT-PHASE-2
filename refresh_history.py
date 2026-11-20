"""Collect fresh historical candles into a NEW folder.

Safe by design:
  * writes only to a new folder; refuses to use the existing data folder
  * connects read-only and refuses to run unless the server is Deriv-Demo
  * never overwrites anything (the pipeline skips files that already exist)

Examples (MT5 must be open and logged in to the Deriv demo account):
    python -m research.refresh_history --days 200
    python -m research.refresh_history --days 200 --output data/historical_refresh_test
"""

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.historical_pipeline import HistoricalDataPipeline
from research.historical_storage import HistoricalDataStorage
from src.config import load_settings
from src.mt5_connection import MT5Connection

EXPECTED_SERVER = "Deriv-Demo"


class DemoOnlyConnection:
    """MT5 connection that refuses to continue on a non-demo server."""

    def __init__(self, inner=None):
        self.inner = inner if inner is not None else MT5Connection()

    def connect(self):
        terminal, account = self.inner.connect()
        if EXPECTED_SERVER not in str(account.server):
            self.inner.disconnect()
            raise RuntimeError(
                f"Refresh requires {EXPECTED_SERVER}; "
                f"connected to {account.server}."
            )
        print(f"{EXPECTED_SERVER} verification: PASSED ({account.server})")
        return terminal, account

    def disconnect(self):
        self.inner.disconnect()


def make_storage(output_directory, protected_directory):
    """Storage pointed at a new folder. Refuses the protected (live) folder."""
    target = Path(output_directory).resolve()
    if target == Path(protected_directory).resolve():
        raise ValueError(
            "Refusing to write into the existing data folder "
            f"({protected_directory}). Choose a new folder."
        )
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Output folder is not empty: {target}")

    storage = HistoricalDataStorage()
    storage.output_directory = Path(output_directory)
    storage.raw_directory = storage.output_directory / "raw"
    storage.reports_directory = storage.output_directory / "reports"
    storage.raw_directory.mkdir(parents=True, exist_ok=True)
    storage.reports_directory.mkdir(parents=True, exist_ok=True)
    return storage


def run_refresh(output_directory, days, settings=None, connection=None,
                market_data=None, collector=None, validator=None):
    settings = copy.deepcopy(settings or load_settings())
    historical = settings["research"]["historical_data"]
    protected = historical.get("output_directory", "data/historical")

    if days is not None:
        historical["history_days"] = int(days)

    storage = make_storage(output_directory, protected)

    pipeline = HistoricalDataPipeline(
        settings=settings,
        connection=connection or DemoOnlyConnection(),
        market_data=market_data,
        collector=collector,
        validator=validator,
        storage=storage,
    )
    summary = pipeline.run()
    return storage, summary


def print_report(summary):
    report = pd.read_csv(summary["report_path"])
    columns = ["instrument", "timeframe", "overall_status",
               "retained_candles", "first_candle", "last_candle",
               "gap_count"]
    columns = [c for c in columns if c in report.columns]
    print("\n=== REFRESH RESULT ===")
    print(report[columns].to_string(index=False))
    print(f"\nStatus counts: {summary['status_counts']}")
    print(f"Report: {summary['report_path']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=None,
                        help="days of history to request (default: settings)")
    parser.add_argument("--output", default=None,
                        help="new folder to write into")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"data/historical_refresh_{stamp}"

    storage, summary = run_refresh(output, args.days)
    print_report(summary)
    print(f"\nNew data saved under: {storage.output_directory}")
    print("Your existing data folder was not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())