
from collections import Counter
from datetime import datetime, timedelta, timezone

from src.config import load_settings
from src.logger import get_logger
from src.mt5_connection import MT5Connection
from src.market_data import MarketData

from research.historical_data import HistoricalDataCollector
from research.historical_validation import HistoricalDataValidator
from research.historical_storage import HistoricalDataStorage


log = get_logger("M15_M5_RESEARCH_COLLECTION")

INSTRUMENT = "EURUSD"
TIMEFRAMES = ("M15", "M5")
EXPECTED_DEMO_SERVER = "Deriv-Demo"


def validate_settings(settings):
    """Validate research configuration and execution safety."""

    trading = settings.get("trading", {})
    research = settings.get("research", {})
    historical = research.get("historical_data", {})
    market = settings.get("market", {})

    if trading.get("execution_enabled") is not False:
        raise RuntimeError(
            "Collection blocked: trade execution must be disabled."
        )

    if historical.get("enabled") is not True:
        raise RuntimeError(
            "Historical research is disabled."
        )

    if historical.get("timezone", "UTC") != "UTC":
        raise RuntimeError(
            "Historical collection requires UTC timezone."
        )

    if historical.get("exclude_forming_candles") is not True:
        raise RuntimeError(
            "Collection requires forming-candle exclusion."
        )

    history_days = historical.get("history_days", 180)

    if (
        isinstance(history_days, bool)
        or not isinstance(history_days, int)
        or history_days <= 0
    ):
        raise ValueError(
            "history_days must be a positive integer."
        )

    configured_timeframes = market.get("timeframes", {})

    missing_timeframes = [
        timeframe
        for timeframe in TIMEFRAMES
        if timeframe not in configured_timeframes
    ]

    if missing_timeframes:
        raise RuntimeError(
            "Required timeframes are missing from settings: "
            f"{missing_timeframes}"
        )

    return history_days


def build_report(
    timeframe,
    broker_symbol,
    metadata,
    validation,
    dataset_path,
):
    """Build a detailed collection and validation report."""

    return {
        **metadata,
        "requested_instrument": INSTRUMENT,
        "broker_symbol": broker_symbol,
        "timeframe": timeframe,
        "validation_status": validation["status"],
        "validation_valid": validation["valid"],
        "candle_count": validation["candle_count"],
        "duplicate_timestamps": validation["duplicate_timestamps"],
        "non_chronological_pairs": (
            validation["non_chronological_pairs"]
        ),
        "invalid_ohlc_rows": validation["invalid_ohlc_rows"],
        "missing_values": validation["missing_values"],
        "non_finite_values": validation["non_finite_values"],
        "non_positive_price_rows": (
            validation["non_positive_price_rows"]
        ),
        "negative_value_rows": validation["negative_value_rows"],
        "forming_candles": validation["forming_candles"],
        "future_timestamps": validation["future_timestamps"],
        "gap_count": validation["gap_count"],
        "estimated_missing_intervals": (
            validation["estimated_missing_intervals"]
        ),
        "warnings": " | ".join(validation["warnings"]),
        "errors": " | ".join(validation["errors"]),
        "dataset_path": str(dataset_path),
    }


def main():
    settings = load_settings()

    print("\n========== CONFIGURATION SAFETY CHECKS ==========")

    history_days = validate_settings(settings)

    print("Historical research: ENABLED")
    print("Trade execution: DISABLED")
    print("Forming-candle exclusion: ENABLED")
    print(f"History period: {history_days} days")
    print(f"Instrument: {INSTRUMENT}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")

    connection = MT5Connection()
    market_data = MarketData()
    collector = HistoricalDataCollector()
    validator = HistoricalDataValidator()
    storage = HistoricalDataStorage()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=history_days)

    results = []
    connection_started = False

    try:
        print("\n========== MT5 DEMO CONNECTION ==========")

        terminal, account = connection.connect()
        connection_started = True

        server = str(account.server)

        print(f"Terminal: {terminal.name}")
        print(f"Account server: {server}")

        if EXPECTED_DEMO_SERVER not in server:
            raise RuntimeError(
                f"Unexpected MT5 server: {server}. "
                "This collection requires Deriv-Demo."
            )

        print("Deriv-Demo verification: PASSED")

        print("\n========== EURUSD SYMBOL RESOLUTION ==========")

        available_symbols = market_data.discover_symbols()

        broker_symbol = market_data.resolve_symbol(
            "EURUSD.0",
            available_symbols,
        )

        if broker_symbol is None:
            broker_symbol = market_data.resolve_symbol(
                "EURUSD",
                available_symbols,
            )

        if broker_symbol is None:
            raise RuntimeError(
                "Could not resolve EURUSD on the Deriv account."
            )

        print(f"Requested instrument: {INSTRUMENT}")
        print(f"Resolved broker symbol: {broker_symbol}")

        for timeframe in TIMEFRAMES:

            print(
                f"\n========== {timeframe} COLLECTION =========="
            )

            dataset_path = (
                storage.raw_directory
                / INSTRUMENT
                / f"{INSTRUMENT}_{timeframe}.parquet"
            )

            if dataset_path.exists():
                print(
                    f"SKIPPED: Dataset already exists: {dataset_path}"
                )

                results.append({
                    "timeframe": timeframe,
                    "status": "SKIPPED_EXISTING",
                    "dataset_path": str(dataset_path),
                })

                continue

            print(f"Start: {start.isoformat()}")
            print(f"End: {end.isoformat()}")
            print(f"History days: {history_days}")

            data, metadata = collector.collect_with_metadata(
                broker_symbol,
                timeframe,
                start,
                end,
            )

            print(
                "Raw retrieved candles: "
                f"{metadata['raw_retrieved_candles']}"
            )
            print(
                "Retained candles: "
                f"{metadata['retained_candles']}"
            )
            print(
                "Excluded candles: "
                f"{metadata['excluded_candles']}"
            )
            print(f"First candle: {metadata['first_candle']}")
            print(f"Last candle: {metadata['last_candle']}")

            if data.empty:
                raise RuntimeError(
                    f"{timeframe} collection returned no candles."
                )

            print(f"\n========== {timeframe} VALIDATION ==========")

            validation = validator.validate(
                data,
                timeframe,
            )

            print(f"Validation status: {validation['status']}")
            print(f"Valid: {validation['valid']}")
            print(f"Candle count: {validation['candle_count']}")
            print(
                "Duplicate timestamps: "
                f"{validation['duplicate_timestamps']}"
            )
            print(
                "Invalid OHLC rows: "
                f"{validation['invalid_ohlc_rows']}"
            )
            print(
                "Missing values: "
                f"{validation['missing_values']}"
            )
            print(
                "Forming candles: "
                f"{validation['forming_candles']}"
            )
            print(f"Gap count: {validation['gap_count']}")
            print(
                "Estimated missing intervals: "
                f"{validation['estimated_missing_intervals']}"
            )

            for warning in validation["warnings"]:
                print(f"WARNING: {warning}")

            for error in validation["errors"]:
                print(f"ERROR: {error}")

            if (
                validation["valid"] is not True
                or validation["status"] not in {"PASS", "WARNING"}
            ):
                raise RuntimeError(
                    f"{timeframe} dataset failed validation. "
                    "Dataset will not be saved."
                )

            print(f"\n========== {timeframe} STORAGE ==========")

            saved_path = storage.save_dataset(
                data,
                INSTRUMENT,
                timeframe,
                overwrite=False,
            )

            report = build_report(
                timeframe,
                broker_symbol,
                metadata,
                validation,
                saved_path,
            )

            report_name = (
                f"EURUSD_{timeframe}_collection_"
                + datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S_%fZ"
                )
            )

            report_path = storage.save_report(
                [report],
                report_name,
                overwrite=False,
            )

            results.append({
                "timeframe": timeframe,
                "status": validation["status"],
                "candles": len(data),
                "dataset_path": str(saved_path),
                "report_path": str(report_path),
            })

            print(f"Dataset saved: {saved_path}")
            print(f"Report saved: {report_path}")

        print("\n========== FINAL COLLECTION SUMMARY ==========")

        for result in results:
            print(
                f"{result['timeframe']}: "
                f"{result['status']}"
            )

            if "candles" in result:
                print(f"  Candles: {result['candles']}")

            if "dataset_path" in result:
                print(f"  Dataset: {result['dataset_path']}")

            if "report_path" in result:
                print(f"  Report: {result['report_path']}")

        counts = Counter(
            result["status"]
            for result in results
        )

        print(f"\nStatus counts: {dict(counts)}")

        failed = any(
            result["status"] not in {
                "PASS",
                "WARNING",
                "SKIPPED_EXISTING",
            }
            for result in results
        )

        if failed:
            raise RuntimeError(
                "One or more timeframe collections failed."
            )

        print("\nM15/M5 HISTORICAL COLLECTION COMPLETED.")

    finally:
        if connection_started:
            connection.disconnect()

        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()
