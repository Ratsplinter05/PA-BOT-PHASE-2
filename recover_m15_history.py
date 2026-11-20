
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil

import pandas as pd

from src.config import load_settings
from src.logger import get_logger
from src.mt5_connection import MT5Connection
from src.market_data import MarketData

from research.historical_data import HistoricalDataCollector
from research.historical_validation import HistoricalDataValidator
from research.historical_storage import HistoricalDataStorage


log = get_logger("M15_HISTORY_RECOVERY")

INSTRUMENT = "EURUSD"
TIMEFRAME = "M15"
EXPECTED_DEMO_SERVER = "Deriv-Demo"
TIMEFRAME_SECONDS = 900


def validate_settings(settings):
    """Confirm research and trading safety settings."""

    trading = settings.get("trading", {})
    research = settings.get("research", {})
    historical = research.get("historical_data", {})
    market = settings.get("market", {})

    if trading.get("execution_enabled") is not False:
        raise RuntimeError(
            "Recovery blocked: trade execution must remain disabled."
        )

    if historical.get("enabled") is not True:
        raise RuntimeError(
            "Historical research is disabled."
        )

    if historical.get("timezone", "UTC") != "UTC":
        raise RuntimeError(
            "Recovery requires UTC timezone."
        )

    if historical.get("exclude_forming_candles") is not True:
        raise RuntimeError(
            "Recovery requires forming-candle exclusion."
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

    if TIMEFRAME not in market.get("timeframes", {}):
        raise RuntimeError(
            "M15 is not configured in market.timeframes."
        )

    return history_days


def get_expected_latest_candle(end):
    """
    Calculate the latest M15 candle that should be fully closed
    at the requested collection end time.
    """

    latest_epoch = (
        int(end.timestamp()) // TIMEFRAME_SECONDS
    ) * TIMEFRAME_SECONDS - TIMEFRAME_SECONDS

    return pd.Timestamp(
        latest_epoch,
        unit="s",
        tz="UTC",
    )


def validate_recovered_dataset(
    data,
    validation,
    expected_latest,
    existing_data,
):
    """Apply recovery-specific checks before replacing anything."""

    if data.empty:
        raise RuntimeError(
            "Recovery returned an empty dataset."
        )

    if (
        validation["valid"] is not True
        or validation["status"] not in {"PASS", "WARNING"}
    ):
        raise RuntimeError(
            "Recovered M15 dataset failed historical validation. "
            "Existing dataset will not be replaced."
        )

    actual_latest = pd.to_datetime(
        data["time"],
        utc=True,
    ).max()

    if actual_latest != expected_latest:
        raise RuntimeError(
            "Latest completed M15 candle check failed. "
            f"Expected={expected_latest.isoformat()} | "
            f"Actual={actual_latest.isoformat()}. "
            "Existing dataset will not be replaced."
        )

    existing_latest = pd.to_datetime(
        existing_data["time"],
        utc=True,
    ).max()

    if actual_latest <= existing_latest:
        raise RuntimeError(
            "Recovered dataset does not extend beyond the existing "
            "dataset. Existing dataset will not be replaced."
        )

    if len(data) <= len(existing_data):
        raise RuntimeError(
            "Recovered dataset does not contain more candles than "
            "the existing dataset. Existing dataset will not be replaced."
        )

    return actual_latest


def build_recovery_report(
    broker_symbol,
    metadata,
    validation,
    expected_latest,
    existing_count,
    existing_latest,
    backup_path,
    dataset_path,
):
    """Prepare a recovery audit report."""

    return {
        **metadata,
        "requested_instrument": INSTRUMENT,
        "broker_symbol": broker_symbol,
        "timeframe": TIMEFRAME,
        "recovery_status": "SUCCESS",
        "expected_latest_completed_candle": expected_latest.isoformat(),
        "existing_candle_count": existing_count,
        "existing_latest_candle": existing_latest.isoformat(),
        "recovered_candle_count": len(validation.get(
            "validated_data", []
        )) if "validated_data" in validation else metadata[
            "retained_candles"
        ],
        "validation_status": validation["status"],
        "validation_valid": validation["valid"],
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
        "backup_path": str(backup_path),
        "dataset_path": str(dataset_path),
    }


def main():
    settings = load_settings()

    print("\n========== M15 RECOVERY SAFETY CHECKS ==========")

    history_days = validate_settings(settings)

    print("Historical research: ENABLED")
    print("Trade execution: DISABLED")
    print("Forming-candle exclusion: ENABLED")
    print(f"History period: {history_days} days")
    print(f"Instrument: {INSTRUMENT}")
    print(f"Timeframe: {TIMEFRAME}")

    connection = MT5Connection()
    market_data = MarketData()
    collector = HistoricalDataCollector()
    validator = HistoricalDataValidator()
    storage = HistoricalDataStorage()

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
                "Recovery requires Deriv-Demo."
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

        print(f"Resolved broker symbol: {broker_symbol}")

        dataset_path = (
            storage.raw_directory
            / INSTRUMENT
            / f"{INSTRUMENT}_{TIMEFRAME}.parquet"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Existing M15 dataset not found: {dataset_path}"
            )

        print("\n========== EXISTING DATASET ==========")

        existing_data = storage.load_dataset(
            INSTRUMENT,
            TIMEFRAME,
        )

        if existing_data.empty:
            raise RuntimeError(
                "Existing M15 dataset is empty. "
                "Recovery stopped."
            )

        existing_latest = pd.to_datetime(
            existing_data["time"],
            utc=True,
        ).max()

        existing_count = len(existing_data)

        print(f"Existing candles: {existing_count}")
        print(f"Existing latest candle: {existing_latest}")

        print("\n========== NEW M15 COLLECTION ==========")

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=history_days)

        expected_latest = get_expected_latest_candle(end)

        print(f"Start: {start.isoformat()}")
        print(f"End: {end.isoformat()}")
        print(
            "Expected latest completed M15 candle: "
            f"{expected_latest.isoformat()}"
        )

        data, metadata = collector.collect_with_metadata(
            broker_symbol,
            TIMEFRAME,
            start,
            end,
        )

        print(f"Raw retrieved candles: {metadata['raw_retrieved_candles']}")
        print(f"Retained candles: {metadata['retained_candles']}")
        print(f"Excluded candles: {metadata['excluded_candles']}")
        print(f"First candle: {metadata['first_candle']}")
        print(f"Last candle: {metadata['last_candle']}")

        print("\n========== M15 VALIDATION ==========")

        validation = validator.validate(
            data,
            TIMEFRAME,
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

        actual_latest = validate_recovered_dataset(
            data,
            validation,
            expected_latest,
            existing_data,
        )

        print("\n========== RECOVERY COMPARISON ==========")
        print(f"Existing candles: {existing_count}")
        print(f"Recovered candles: {len(data)}")
        print(f"Additional candles: {len(data) - existing_count}")
        print(f"Existing latest: {existing_latest}")
        print(f"Recovered latest: {actual_latest}")
        print("Recovery checks: PASSED")

        print("\n========== BACKUP EXISTING DATASET ==========")

        backup_directory = (
            storage.output_directory / "backups"
        )

        backup_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        backup_timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S_%fZ")

        backup_path = (
            backup_directory
            / f"EURUSD_M15_before_recovery_{backup_timestamp}.parquet"
        )

        shutil.copy2(
            dataset_path,
            backup_path,
        )

        if not backup_path.exists():
            raise RuntimeError(
                "Backup creation failed. Existing dataset was not replaced."
            )

        if backup_path.stat().st_size != dataset_path.stat().st_size:
            raise RuntimeError(
                "Backup size verification failed. "
                "Existing dataset was not replaced."
            )

        print(f"Backup created: {backup_path}")
        print("Backup verification: PASSED")

        print("\n========== REPLACE M15 DATASET ==========")

        saved_path = storage.save_dataset(
            data,
            INSTRUMENT,
            TIMEFRAME,
            overwrite=True,
        )

        # Read the saved dataset back and verify its contents.
        saved_data = storage.load_dataset(
            INSTRUMENT,
            TIMEFRAME,
        )

        saved_latest = pd.to_datetime(
            saved_data["time"],
            utc=True,
        ).max()

        if (
            len(saved_data) != len(data)
            or saved_latest != actual_latest
        ):
            # Restore the original file if post-save verification fails.
            shutil.copy2(
                backup_path,
                dataset_path,
            )

            raise RuntimeError(
                "Saved dataset verification failed. "
                "Original dataset has been restored from backup."
            )

        print(f"Dataset replaced: {saved_path}")
        print(f"Saved candles: {len(saved_data)}")
        print(f"Saved latest candle: {saved_latest}")
        print("Post-save verification: PASSED")

        print("\n========== SAVE RECOVERY REPORT ==========")

        report = build_recovery_report(
            broker_symbol,
            metadata,
            validation,
            expected_latest,
            existing_count,
            existing_latest,
            backup_path,
            saved_path,
        )

        report_name = (
            "EURUSD_M15_recovery_"
            + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S_%fZ"
            )
        )

        report_path = storage.save_report(
            [report],
            report_name,
            overwrite=False,
        )

        print(f"Recovery report: {report_path}")

        print("\n========== FINAL RESULT ==========")
        print("M15 RECOVERY: SUCCESS")
        print(f"Existing candles: {existing_count}")
        print(f"Recovered candles: {len(saved_data)}")
        print(f"Additional candles: {len(saved_data) - existing_count}")
        print(f"Latest completed candle: {saved_latest}")
        print(f"Backup: {backup_path}")
        print(f"Dataset: {saved_path}")
        print("M5 dataset: UNCHANGED")
        print("Trade execution: DISABLED")

    finally:
        if connection_started:
            connection.disconnect()

        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()
