from datetime import datetime, timedelta, timezone

from src.config import load_settings
from src.logger import get_logger
from src.mt5_connection import MT5Connection
from src.market_data import MarketData

from research.historical_data import HistoricalDataCollector
from research.historical_validation import HistoricalDataValidator
from research.historical_storage import HistoricalDataStorage


log = get_logger("H4_RESEARCH_COLLECTION")


def main():
    settings = load_settings()

    # --------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------

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

    if "H4" not in market.get("timeframes", {}):
        raise RuntimeError(
            "H4 is missing from market.timeframes in settings.yaml."
        )

    instrument = "EURUSD"
    timeframe = "H4"

    history_days = historical.get("history_days", 180)

    if (
        isinstance(history_days, bool)
        or not isinstance(history_days, int)
        or history_days <= 0
    ):
        raise ValueError(
            "Historical history_days must be a positive integer."
        )

    # --------------------------------------------------
    # INITIALISE COMPONENTS
    # --------------------------------------------------

    connection = MT5Connection()
    market_data = MarketData()
    collector = HistoricalDataCollector()
    validator = HistoricalDataValidator()
    storage = HistoricalDataStorage()

    dataset_path = (
        storage.raw_directory
        / instrument
        / f"{instrument}_{timeframe}.parquet"
    )

    if dataset_path.exists():
        raise FileExistsError(
            f"Dataset already exists: {dataset_path}. "
            "Existing data will not be overwritten."
        )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=history_days)

    try:
        # --------------------------------------------------
        # CONNECT TO DEMO ACCOUNT
        # --------------------------------------------------

        print("\n========== MT5 CONNECTION ==========")

        terminal, account = connection.connect()

        server = str(account.server)

        print(f"Terminal: {terminal.name}")
        print(f"Account server: {server}")

        if "Deriv-Demo" not in server:
            raise RuntimeError(
                f"Unexpected MT5 server: {server}. "
                "This research collection requires Deriv-Demo."
            )

        print("Demo account verification: PASSED")

        # --------------------------------------------------
        # RESOLVE BROKER SYMBOL
        # --------------------------------------------------

        print("\n========== SYMBOL RESOLUTION ==========")

        available_symbols = market_data.discover_symbols()

        broker_symbol = market_data.resolve_symbol(
            "EURUSD.0",
            available_symbols
        )

        if broker_symbol is None:
            broker_symbol = market_data.resolve_symbol(
                "EURUSD",
                available_symbols
            )

        if broker_symbol is None:
            raise RuntimeError(
                "Could not resolve EURUSD on the Deriv account."
            )

        print(f"Requested instrument: {instrument}")
        print(f"Resolved broker symbol: {broker_symbol}")

        # --------------------------------------------------
        # COLLECT H4 HISTORICAL DATA
        # --------------------------------------------------

        print("\n========== H4 HISTORICAL COLLECTION ==========")

        print(f"Start: {start.isoformat()}")
        print(f"End: {end.isoformat()}")
        print(f"History days: {history_days}")

        data, metadata = collector.collect_with_metadata(
            broker_symbol,
            timeframe,
            start,
            end
        )

        print(f"Raw retrieved candles: {metadata['raw_retrieved_candles']}")
        print(f"Retained candles: {metadata['retained_candles']}")
        print(f"Excluded candles: {metadata['excluded_candles']}")
        print(f"First candle: {metadata['first_candle']}")
        print(f"Last candle: {metadata['last_candle']}")

        if data.empty:
            raise RuntimeError(
                "Historical collection returned no retained candles."
            )

        # --------------------------------------------------
        # VALIDATE DATA
        # --------------------------------------------------

        print("\n========== H4 DATA VALIDATION ==========")

        validation = validator.validate(
            data,
            timeframe
        )

        print(f"Validation status: {validation['status']}")
        print(f"Valid: {validation['valid']}")
        print(f"Candle count: {validation['candle_count']}")
        print(f"Duplicate timestamps: {validation['duplicate_timestamps']}")
        print(f"Invalid OHLC rows: {validation['invalid_ohlc_rows']}")
        print(f"Missing values: {validation['missing_values']}")
        print(f"Forming candles: {validation['forming_candles']}")
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
                "H4 dataset failed validation. "
                "Dataset will not be saved."
            )

        # --------------------------------------------------
        # SAVE DATASET
        # --------------------------------------------------

        print("\n========== H4 DATA STORAGE ==========")

        saved_path = storage.save_dataset(
            data,
            instrument,
            timeframe,
            overwrite=False
        )

        # --------------------------------------------------
        # SAVE COLLECTION REPORT
        # --------------------------------------------------

        report = {
            **metadata,
            "requested_instrument": instrument,
            "broker_symbol": broker_symbol,
            "validation_status": validation["status"],
            "validation_valid": validation["valid"],
            "duplicate_timestamps": validation["duplicate_timestamps"],
            "non_chronological_pairs": validation["non_chronological_pairs"],
            "invalid_ohlc_rows": validation["invalid_ohlc_rows"],
            "missing_values": validation["missing_values"],
            "non_finite_values": validation["non_finite_values"],
            "forming_candles": validation["forming_candles"],
            "future_timestamps": validation["future_timestamps"],
            "gap_count": validation["gap_count"],
            "estimated_missing_intervals": (
                validation["estimated_missing_intervals"]
            ),
            "warnings": " | ".join(validation["warnings"]),
            "errors": " | ".join(validation["errors"]),
            "dataset_path": str(saved_path),
        }

        report_name = (
            "EURUSD_H4_collection_"
            + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S_%fZ"
            )
        )

        report_path = storage.save_report(
            [report],
            report_name,
            overwrite=False
        )

        # --------------------------------------------------
        # FINAL SUMMARY
        # --------------------------------------------------

        print("\n========== COLLECTION SUMMARY ==========")

        print(f"Instrument: {instrument}")
        print(f"Timeframe: {timeframe}")
        print(f"Retained candles: {len(data)}")
        print(f"Validation: {validation['status']}")
        print(f"Dataset: {saved_path}")
        print(f"Report: {report_path}")

        print("\nH4 HISTORICAL COLLECTION COMPLETED.")

    finally:
        connection.disconnect()


if __name__ == "__main__":
    main()
