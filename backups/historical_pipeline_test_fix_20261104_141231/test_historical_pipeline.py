import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.historical_pipeline import HistoricalDataPipeline


class FakeConnection:
    def __init__(self, connect_error=None):
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.connect_error = connect_error

    def connect(self):
        self.connect_calls += 1

        if self.connect_error:
            raise self.connect_error

    def disconnect(self):
        self.disconnect_calls += 1


class FakeMarketData:
    def __init__(self, symbols=None, discovery_error=None):
        self.symbols = symbols or []
        self.discovery_error = discovery_error
        self.discover_calls = 0
        self.resolve_calls = []

    def discover_symbols(self):
        self.discover_calls += 1

        if self.discovery_error:
            raise self.discovery_error

        return self.symbols

    @staticmethod
    def resolve_symbol(requested, available):
        for symbol in available:
            if symbol == requested:
                return symbol

        for symbol in available:
            if symbol.lower() == requested.lower():
                return symbol

        return None


class FakeCollector:
    def __init__(self, errors=None):
        self.errors = errors or {}
        self.calls = []

    def collect_with_metadata(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe, start, end))

        error = self.errors.get((symbol, timeframe))

        if error:
            raise error

        data = pd.DataFrame({
            "time": [
                pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
            ],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        })

        metadata = {
            "collection_time": "2026-11-04T08:00:00+00:00",
            "batch_count": 1,
            "raw_retrieved_candles": 10,
            "in_range_candles": 10,
            "out_of_range_candles": 0,
            "excluded_candles": 0,
            "retained_candles": 10,
            "first_candle": "2026-01-01T00:00:00+00:00",
            "last_candle": "2026-01-01T01:00:00+00:00",
        }

        return data, metadata


class FakeValidator:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def validate(self, data, timeframe):
        self.calls.append((data, timeframe))

        return self.results.get(
            timeframe,
            {
                "status": "PASS",
                "valid": True,
                "duplicate_timestamps": 0,
                "gap_count": 0,
                "estimated_missing_intervals": 0,
                "errors": [],
                "warnings": [],
            },
        )


class FakeStorage:
    def __init__(self, root):
        self.raw_directory = Path(root) / "raw"
        self.reports_directory = Path(root) / "reports"

        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.reports_directory.mkdir(parents=True, exist_ok=True)

        self.saved_datasets = []
        self.saved_reports = []
        self.dataset_error = None
        self.report_error = None

    def save_dataset(self, data, symbol, timeframe, overwrite=False):
        if self.dataset_error:
            raise self.dataset_error

        path = (
            self.raw_directory
            / symbol
            / f"{symbol}_{timeframe}.parquet"
        )

        self.saved_datasets.append({
            "data": data,
            "symbol": symbol,
            "timeframe": timeframe,
            "overwrite": overwrite,
            "path": path,
        })

        return path

    def save_report(self, report, report_name, overwrite=False):
        if self.report_error:
            raise self.report_error

        self.saved_reports.append({
            "report": report,
            "name": report_name,
            "overwrite": overwrite,
        })

        return self.reports_directory / f"{report_name}.csv"


class HistoricalPipelineTests(unittest.TestCase):

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)

        self.settings = {
            "research": {
                "historical_data": {
                    "enabled": True,
                    "history_days": 180,
                    "timezone": "UTC",
                }
            },
            "trading": {
                "execution_enabled": False,
            },
            "market": {
                "instruments": ["XAUUSD"],
                "timeframes": {
                    "H4": "H4",
                },
            },
        }

        self.connection = FakeConnection()
        self.market_data = FakeMarketData(
            symbols=["XAUUSD.0"]
        )
        self.collector = FakeCollector()
        self.validator = FakeValidator()
        self.storage = FakeStorage(self.root)

        self.pipeline = HistoricalDataPipeline(
            settings=self.settings,
            connection=self.connection,
            market_data=self.market_data,
            collector=self.collector,
            validator=self.validator,
            storage=self.storage,
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_successful_collection_saves_dataset_and_report(self):
        summary = self.pipeline.run()

        self.assertEqual(summary["total_combinations"], 1)
        self.assertEqual(summary["status_counts"], {"PASS": 1})

        self.assertEqual(self.connection.connect_calls, 1)
        self.assertEqual(self.connection.disconnect_calls, 1)

        self.assertEqual(len(self.collector.calls), 1)
        self.assertEqual(len(self.validator.calls), 1)
        self.assertEqual(len(self.storage.saved_datasets), 1)
        self.assertEqual(len(self.storage.saved_reports), 1)

        saved = self.storage.saved_datasets[0]

        self.assertEqual(saved["symbol"], "XAUUSD")
        self.assertEqual(saved["timeframe"], "H4")
        self.assertFalse(saved["overwrite"])

        report = self.storage.saved_reports[0]
        self.assertFalse(report["overwrite"])
        self.assertEqual(len(report["report"]), 1)

    def test_warning_validation_still_saves_dataset(self):
        self.validator.results["H4"] = {
            "status": "WARNING",
            "valid": True,
            "duplicate_timestamps": 0,
            "gap_count": 2,
            "estimated_missing_intervals": 5,
            "errors": [],
            "warnings": ["Potential candle gaps detected."],
        }

        summary = self.pipeline.run()

        self.assertEqual(summary["status_counts"], {"WARNING": 1})
        self.assertEqual(len(self.storage.saved_datasets), 1)

        row = self.storage.saved_reports[0]["report"][0]

        self.assertEqual(row["validation_status"], "WARNING")
        self.assertEqual(row["gap_count"], 2)
        self.assertEqual(row["estimated_missing_intervals"], 5)
        self.assertEqual(row["storage_status"], "SAVED")

    def test_failed_validation_does_not_save_dataset(self):
        self.validator.results["H4"] = {
            "status": "FAIL",
            "valid": False,
            "duplicate_timestamps": 2,
            "gap_count": 0,
            "estimated_missing_intervals": 0,
            "errors": ["Duplicate timestamps detected."],
            "warnings": [],
        }

        summary = self.pipeline.run()

        self.assertEqual(summary["status_counts"], {"FAIL": 1})
        self.assertEqual(len(self.storage.saved_datasets), 0)

        row = self.storage.saved_reports[0]["report"][0]

        self.assertEqual(row["collection_status"], "SUCCESS")
        self.assertEqual(row["validation_status"], "FAIL")
        self.assertEqual(row["storage_status"], "NOT_SAVED")

    def test_unresolved_symbol_records_each_timeframe(self):
        self.settings["market"]["timeframes"] = {
            "H4": "H4",
            "H1": "H1",
        }

        self.market_data.symbols = []

        summary = self.pipeline.run()

        self.assertEqual(summary["total_combinations"], 2)
        self.assertEqual(summary["status_counts"], {"NOT_FOUND": 2})

        self.assertEqual(len(self.collector.calls), 0)
        self.assertEqual(len(self.storage.saved_datasets), 0)

        report_rows = self.storage.saved_reports[0]["report"]

        self.assertTrue(
            all(row["collection_status"] == "NOT_FOUND"
                for row in report_rows)
        )

    def test_collection_error_does_not_stop_next_timeframe(self):
        self.settings["market"]["timeframes"] = {
            "H4": "H4",
            "H1": "H1",
        }

        self.collector.errors[("XAUUSD.0", "H4")] = RuntimeError(
            "Simulated collection failure."
        )

        summary = self.pipeline.run()

        self.assertEqual(summary["total_combinations"], 2)
        self.assertEqual(
            summary["status_counts"],
            {"ERROR": 1, "PASS": 1},
        )

        self.assertEqual(len(self.collector.calls), 2)
        self.assertEqual(len(self.storage.saved_datasets), 1)

    def test_existing_dataset_is_skipped_without_collection(self):
        existing_path = (
            self.storage.raw_directory
            / "XAUUSD"
            / "XAUUSD_H4.parquet"
        )

        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_bytes(b"existing dataset")

        summary = self.pipeline.run()

        self.assertEqual(
            summary["status_counts"],
            {"SKIPPED_EXISTING": 1},
        )

        self.assertEqual(len(self.collector.calls), 0)
        self.assertEqual(len(self.storage.saved_datasets), 0)

        row = self.storage.saved_reports[0]["report"][0]

        self.assertEqual(row["storage_status"], "EXISTS")
        self.assertEqual(row["collection_status"], "SKIPPED")

    def test_execution_enabled_blocks_pipeline_before_connection(self):
        self.settings["trading"]["execution_enabled"] = True

        with self.assertRaises(RuntimeError):
            self.pipeline.run()

        self.assertEqual(self.connection.connect_calls, 0)
        self.assertEqual(self.connection.disconnect_calls, 0)
        self.assertEqual(self.market_data.discover_calls, 0)

    def test_disabled_research_blocks_pipeline_before_connection(self):
        self.settings["research"]["historical_data"]["enabled"] = False

        with self.assertRaises(RuntimeError):
            self.pipeline.run()

        self.assertEqual(self.connection.connect_calls, 0)
        self.assertEqual(self.connection.disconnect_calls, 0)

    def test_connection_is_disconnected_when_symbol_discovery_fails(self):
        self.market_data.discovery_error = RuntimeError(
            "Simulated symbol discovery failure."
        )

        with self.assertRaises(RuntimeError):
            self.pipeline.run()

        self.assertEqual(self.connection.connect_calls, 1)
        self.assertEqual(self.connection.disconnect_calls, 1)
        self.assertEqual(len(self.storage.saved_reports), 0)

    def test_existing_file_race_does_not_overwrite_dataset(self):
        self.storage.dataset_error = FileExistsError(
            "Dataset appeared during collection."
        )

        summary = self.pipeline.run()

        self.assertEqual(
            summary["status_counts"],
            {"SKIPPED_EXISTING": 1},
        )

        row = self.storage.saved_reports[0]["report"][0]

        self.assertEqual(row["storage_status"], "EXISTS")
        self.assertEqual(row["overall_status"], "SKIPPED_EXISTING")

    def test_invalid_history_days_blocks_pipeline(self):
        self.settings["research"]["historical_data"]["history_days"] = 0

        with self.assertRaises(ValueError):
            self.pipeline.run()

        self.assertEqual(self.connection.connect_calls, 0)

    def test_duplicate_instruments_are_rejected(self):
        self.settings["market"]["instruments"] = [
            "XAUUSD",
            "XAUUSD",
        ]

        with self.assertRaises(ValueError):
            self.pipeline.run()

        self.assertEqual(self.connection.connect_calls, 0)

    def test_connection_is_disconnected_when_connect_fails(self):
        self.connection.connect_error = RuntimeError(
            "Simulated MT5 connection failure."
        )

        with self.assertRaises(RuntimeError):
            self.pipeline.run()

        self.assertEqual(self.connection.connect_calls, 1)
        self.assertEqual(self.connection.disconnect_calls, 1)
        self.assertEqual(self.market_data.discover_calls, 0)


if __name__ == "__main__":
    unittest.main()
