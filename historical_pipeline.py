from collections import Counter
from datetime import datetime, timedelta, timezone
import re

from src.config import load_settings
from src.logger import get_logger
from src.mt5_connection import MT5Connection
from src.market_data import MarketData

from research.historical_data import HistoricalDataCollector
from research.historical_validation import HistoricalDataValidator
from research.historical_storage import HistoricalDataStorage


log = get_logger("HISTORICAL_PIPELINE")


BROKER_SYMBOL_CANDIDATES = {
    "XAUUSD": ("XAUUSD.0", "XAUUSD"),
    "GBPUSD": ("GBPUSD.0", "GBPUSD"),
    "BTCUSD": ("BTCUSD.0", "BTCUSD"),
    "EURUSD": ("EURUSD.0", "EURUSD"),
    "US30": ("Wall Street 30.0", "US30"),
    "NAS100": ("US Tech 100.0", "NAS100"),
}


SUPPORTED_TIMEFRAMES = {"M5", "M15", "H1", "H4"}


class HistoricalDataPipeline:
    """
    Coordinates historical data collection, validation and storage.

    Safety:
    - Requires historical research to be enabled.
    - Refuses to run if trade execution is enabled.
    - Uses one MT5 connection per pipeline run.
    - Never places or modifies trades.
    - Never overwrites existing historical datasets.
    - Continues processing after individual combination failures.
    """

    def __init__(
        self,
        settings=None,
        connection=None,
        market_data=None,
        collector=None,
        validator=None,
        storage=None,
    ):
        self.settings = settings if settings is not None else load_settings()

        self.connection = connection
        self.market_data = market_data
        self.collector = collector
        self.validator = validator
        self.storage = storage

    def _validate_configuration(self):
        """Validate configuration before creating directories or connecting."""

        if not isinstance(self.settings, dict):
            raise TypeError("Settings must be a dictionary.")

        research = self.settings.get("research", {})
        historical = research.get("historical_data", {})
        trading = self.settings.get("trading", {})
        market = self.settings.get("market", {})

        if historical.get("enabled") is not True:
            raise RuntimeError(
                "Historical research is disabled in settings.yaml."
            )

        if trading.get("execution_enabled") is not False:
            raise RuntimeError(
                "Historical research blocked: "
                "trading.execution_enabled must explicitly be false."
            )

        history_days = historical.get("history_days", 180)

        if (
            isinstance(history_days, bool)
            or not isinstance(history_days, int)
            or history_days <= 0
        ):
            raise ValueError(
                "research.historical_data.history_days "
                "must be a positive integer."
            )

        if historical.get("timezone", "UTC") != "UTC":
            raise ValueError(
                "Historical research currently requires UTC timezone."
            )

        instruments = market.get("instruments")
        timeframes = market.get("timeframes")

        if not isinstance(instruments, list) or not instruments:
            raise ValueError(
                "market.instruments must be a non-empty list."
            )

        if not all(
            isinstance(item, str) and item.strip()
            for item in instruments
        ):
            raise ValueError(
                "Every configured instrument must be a non-empty string."
            )

        instruments = [item.strip() for item in instruments]

        if len(instruments) != len(set(instruments)):
            raise ValueError(
                "Duplicate instruments detected in market.instruments."
            )

        for instrument in instruments:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", instrument):
                raise ValueError(
                    f"Invalid instrument name in configuration: {instrument}"
                )

        if not isinstance(timeframes, dict) or not timeframes:
            raise ValueError(
                "market.timeframes must be a non-empty dictionary."
            )

        configured_timeframes = list(timeframes.keys())

        unsupported = (
            set(configured_timeframes) - SUPPORTED_TIMEFRAMES
        )

        if unsupported:
            raise ValueError(
                f"Unsupported historical timeframes: {sorted(unsupported)}"
            )

        return {
            "history_days": history_days,
            "instruments": instruments,
            "timeframes": configured_timeframes,
        }

    @staticmethod
    def _resolve_broker_symbol(instrument, available_symbols, market_data):
        """Resolve an instrument using explicit broker symbol candidates."""

        candidates = BROKER_SYMBOL_CANDIDATES.get(
            instrument.upper(),
            (instrument,)
        )

        for candidate in candidates:
            resolved = market_data.resolve_symbol(
                candidate,
                available_symbols
            )

            if resolved is not None:
                return resolved

        return None

    @staticmethod
    def _empty_result(instrument, broker_symbol, timeframe, start, end):
        """Create a consistent report row before collection begins."""

        return {
            "requested_instrument": instrument,
            "broker_symbol": broker_symbol,
            "timeframe": timeframe,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "collection_time": None,
            "batch_count": 0,
            "raw_retrieved_candles": 0,
            "in_range_candles": 0,
            "out_of_range_candles": 0,
            "excluded_candles": 0,
            "retained_candles": 0,
            "first_candle": None,
            "last_candle": None,
            "duplicate_timestamps": 0,
            "gap_count": 0,
            "estimated_missing_intervals": 0,
            "collection_status": "NOT_STARTED",
            "validation_status": "NOT_RUN",
            "storage_status": "NOT_SAVED",
            "overall_status": "NOT_STARTED",
            "dataset_path": None,
            "errors": [],
            "warnings": [],
        }

    def _process_combination(
        self,
        instrument,
        broker_symbol,
        timeframe,
        start,
        end,
        collector,
        validator,
        storage,
    ):
        """Collect, validate and conditionally save one combination."""

        row = self._empty_result(
            instrument,
            broker_symbol,
            timeframe,
            start,
            end,
        )

        # Use the canonical instrument in storage paths.
        # Broker symbols can contain spaces and are retained in the report.
        dataset_path = (
            storage.raw_directory
            / instrument
            / f"{instrument}_{timeframe}.parquet"
        )

        if dataset_path.exists():
            row["collection_status"] = "SKIPPED"
            row["storage_status"] = "EXISTS"
            row["overall_status"] = "SKIPPED_EXISTING"
            row["dataset_path"] = str(dataset_path)

            log.warning(
                "Existing dataset skipped | Instrument=%s | Timeframe=%s",
                instrument,
                timeframe,
            )

            return row

        try:
            data, metadata = collector.collect_with_metadata(
                broker_symbol,
                timeframe,
                start,
                end,
            )

            row["collection_status"] = "SUCCESS"

            for field in (
                "collection_time",
                "batch_count",
                "raw_retrieved_candles",
                "in_range_candles",
                "out_of_range_candles",
                "excluded_candles",
                "retained_candles",
                "first_candle",
                "last_candle",
            ):
                row[field] = metadata.get(field)

            validation = validator.validate(
                data,
                timeframe,
                symbol=instrument,
            )

            row["validation_status"] = validation["status"]
            row["duplicate_timestamps"] = validation.get(
                "duplicate_timestamps", 0
            )
            row["gap_count"] = validation.get("gap_count", 0)
            row["estimated_missing_intervals"] = validation.get(
                "estimated_missing_intervals", 0
            )

            row["errors"].extend(validation.get("errors", []))
            row["warnings"].extend(validation.get("warnings", []))

            if (
                validation.get("valid") is not True
                or validation.get("status") not in {"PASS", "WARNING"}
            ):
                row["overall_status"] = "FAIL"
                row["storage_status"] = "NOT_SAVED"

                log.error(
                    "Historical validation failed | "
                    "Instrument=%s | Timeframe=%s",
                    instrument,
                    timeframe,
                )

                return row

            saved_path = storage.save_dataset(
                data,
                instrument,
                timeframe,
                overwrite=False,
            )

            row["storage_status"] = "SAVED"
            row["dataset_path"] = str(saved_path)
            row["overall_status"] = validation["status"]

            log.info(
                "Historical combination completed | "
                "Instrument=%s | Timeframe=%s | Status=%s",
                instrument,
                timeframe,
                row["overall_status"],
            )

        except FileExistsError as error:
            row["storage_status"] = "EXISTS"
            row["overall_status"] = "SKIPPED_EXISTING"
            row["errors"].append(str(error))

        except Exception as error:
            row["collection_status"] = (
                "ERROR"
                if row["collection_status"] == "NOT_STARTED"
                else row["collection_status"]
            )

            row["overall_status"] = "ERROR"
            row["errors"].append(str(error))

            log.exception(
                "Historical combination failed | "
                "Instrument=%s | BrokerSymbol=%s | Timeframe=%s",
                instrument,
                broker_symbol,
                timeframe,
            )

        return row

    def run(self):
        """Execute the complete historical research collection pipeline."""

        configuration = self._validate_configuration()

        history_days = configuration["history_days"]
        instruments = configuration["instruments"]
        timeframes = configuration["timeframes"]

        # Construct dependencies only after safety checks pass.
        connection = self.connection or MT5Connection()
        market_data = self.market_data or MarketData()
        collector = self.collector or HistoricalDataCollector()
        validator = self.validator or HistoricalDataValidator()
        storage = self.storage or HistoricalDataStorage()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=history_days)

        log.info(
            "Historical pipeline starting | "
            "Days=%s | Instruments=%s | Timeframes=%s",
            history_days,
            len(instruments),
            len(timeframes),
        )

        results = []

        try:
            connection.connect()

            available_symbols = market_data.discover_symbols()

            if not available_symbols:
                raise RuntimeError(
                    "MT5 returned no available broker symbols."
                )

            for instrument in instruments:

                broker_symbol = self._resolve_broker_symbol(
                    instrument,
                    available_symbols,
                    market_data,
                )

                for timeframe in timeframes:

                    if broker_symbol is None:
                        row = self._empty_result(
                            instrument,
                            None,
                            timeframe,
                            start,
                            end,
                        )

                        row["collection_status"] = "NOT_FOUND"
                        row["overall_status"] = "NOT_FOUND"
                        row["errors"].append(
                            "No matching broker symbol was found."
                        )

                        results.append(row)

                        log.error(
                            "Broker symbol not found | Instrument=%s",
                            instrument,
                        )

                        continue

                    row = self._process_combination(
                        instrument,
                        broker_symbol,
                        timeframe,
                        start,
                        end,
                        collector,
                        validator,
                        storage,
                    )

                    results.append(row)

            report_name = (
                "historical_collection_"
                + datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S_%fZ"
                )
            )

            report_path = storage.save_report(
                results,
                report_name,
                overwrite=False,
            )

        finally:
            connection.disconnect()

        status_counts = Counter(
            row["overall_status"]
            for row in results
        )

        summary = {
            "requested_instruments": len(instruments),
            "requested_timeframes": len(timeframes),
            "total_combinations": len(results),
            "status_counts": dict(status_counts),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "history_days": history_days,
            "report_path": str(report_path),
        }

        log.info(
            "Historical pipeline completed | "
            "Total=%s | Statuses=%s | Report=%s",
            len(results),
            dict(status_counts),
            report_path,
        )

        return summary
