
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pandas as pd

from research.historical_data import HistoricalDataCollector


TEST_SETTINGS = {
    "research": {
        "historical_data": {
            "enabled": True,
            "history_days": 180,
            "batch_size": 5000,
            "storage_format": "parquet",
            "output_directory": "data/historical",
            "timezone": "UTC",
            "exclude_forming_candles": True,
        }
    },
    "market": {
        "timeframes": {
            "H4": "H4",
            "H1": "H1",
            "M15": "M15",
            "M5": "M5",
        }
    },
}


def make_rate(timestamp):
    """Create a mock MT5 candle record."""

    return {
        "time": int(timestamp.timestamp()),
        "open": 1.1000,
        "high": 1.1050,
        "low": 1.0950,
        "close": 1.1020,
        "tick_volume": 100,
        "spread": 10,
        "real_volume": 0,
    }


class TestHistoricalDataCollector(unittest.TestCase):

    def setUp(self):
        self.settings_patcher = patch(
            "research.historical_data.load_settings",
            return_value=TEST_SETTINGS,
        )

        self.market_data_patcher = patch(
            "research.historical_data.MarketData"
        )

        self.settings_patcher.start()
        self.market_data_class = self.market_data_patcher.start()
        self.market_data_mock = self.market_data_class.return_value

        self.addCleanup(self.settings_patcher.stop)
        self.addCleanup(self.market_data_patcher.stop)

        self.collector = HistoricalDataCollector()

        self.start = datetime(
            2025, 1, 1, 0, 0,
            tzinfo=timezone.utc,
        )

        self.end = datetime(
            2025, 1, 1, 0, 10,
            tzinfo=timezone.utc,
        )

        self.market_data_mock.enable_symbol.return_value = True

    def test_accepts_timezone_aware_datetime(self):
        result = self.collector._validate_datetime(
            self.start,
            "start",
        )

        self.assertEqual(result.tzinfo, timezone.utc)

    def test_rejects_naive_datetime(self):
        naive_datetime = datetime(2025, 1, 1, 0, 0)

        with self.assertRaises(ValueError):
            self.collector._validate_datetime(
                naive_datetime,
                "start",
            )

    def test_rejects_non_datetime_value(self):
        with self.assertRaises(TypeError):
            self.collector._validate_datetime(
                "2025-01-01",
                "start",
            )

    def test_rejects_invalid_date_range(self):
        with self.assertRaises(ValueError):
            self.collector.collect(
                "EURUSD",
                "M5",
                self.end,
                self.start,
            )

    def test_rejects_future_end_date(self):
        future_end = (
            datetime.now(timezone.utc)
            + timedelta(days=1)
        )

        with self.assertRaises(ValueError):
            self.collector.collect(
                "EURUSD",
                "M5",
                self.start,
                future_end,
            )

    def test_rejects_unsupported_timeframe(self):
        with self.assertRaises(ValueError):
            self.collector.collect(
                "EURUSD",
                "M1",
                self.start,
                self.end,
            )

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_excludes_candles_closing_after_requested_end(
        self,
        mock_copy_rates,
    ):
        timestamps = [
            self.start,
            self.start + timedelta(minutes=5),
            self.start + timedelta(minutes=10),
        ]

        mock_copy_rates.return_value = [
            make_rate(timestamp)
            for timestamp in timestamps
        ]

        result = self.collector.collect(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertEqual(len(result), 2)

        self.assertEqual(
            result["time"].tolist(),
            timestamps[:2],
        )

        self.assertTrue(
            (result["is_forming_candle"] == False).all()
        )

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_preserves_duplicate_timestamps(
        self,
        mock_copy_rates,
    ):
        duplicate_time = self.start
        second_time = self.start + timedelta(minutes=5)

        mock_copy_rates.return_value = [
            make_rate(duplicate_time),
            make_rate(duplicate_time),
            make_rate(second_time),
        ]

        result = self.collector.collect(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertEqual(len(result), 3)

        self.assertEqual(
            result["time"].duplicated().sum(),
            1,
        )

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_retrieves_multiple_batches_without_losing_candles(
        self,
        mock_copy_rates,
    ):
        self.collector.batch_size = 2

        first_candle = self.start
        second_candle = self.start + timedelta(minutes=5)
        third_candle = self.start + timedelta(minutes=10)

        mock_copy_rates.side_effect = [
            [
                make_rate(first_candle),
                make_rate(second_candle),
            ],
            [
                make_rate(third_candle),
            ],
        ]

        result = self.collector.collect(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertEqual(mock_copy_rates.call_count, 2)

        self.assertEqual(
            result["time"].tolist(),
            [first_candle, second_candle],
        )

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_handles_empty_historical_results(
        self,
        mock_copy_rates,
    ):
        mock_copy_rates.return_value = []

        result = self.collector.collect(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_raises_error_when_mt5_retrieval_fails(
        self,
        mock_copy_rates,
    ):
        mock_copy_rates.return_value = None

        with self.assertRaises(RuntimeError):
            self.collector.collect(
                "EURUSD",
                "M5",
                self.start,
                self.end,
            )



    @patch("research.historical_data.mt5.copy_rates_range")
    def test_collect_with_metadata_returns_collection_details(
        self,
        mock_copy_rates,
    ):
        timestamps = [
            self.start,
            self.start + timedelta(minutes=5),
            self.start + timedelta(minutes=10),
        ]

        mock_copy_rates.return_value = [
            make_rate(timestamp)
            for timestamp in timestamps
        ]

        result, metadata = self.collector.collect_with_metadata(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

        self.assertEqual(metadata["symbol"], "EURUSD")
        self.assertEqual(metadata["timeframe"], "M5")
        self.assertEqual(metadata["batch_count"], 1)
        self.assertEqual(metadata["raw_retrieved_candles"], 3)
        self.assertEqual(metadata["in_range_candles"], 3)
        self.assertEqual(metadata["out_of_range_candles"], 0)
        self.assertEqual(metadata["excluded_candles"], 1)
        self.assertEqual(metadata["retained_candles"], 2)

        self.assertEqual(
            metadata["first_candle"],
            self.start.isoformat(),
        )

        self.assertEqual(
            metadata["last_candle"],
            (self.start + timedelta(minutes=5)).isoformat(),
        )

        self.assertTrue(metadata["exclude_forming_candles"])

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_collect_with_metadata_handles_empty_results(
        self,
        mock_copy_rates,
    ):
        mock_copy_rates.return_value = []

        result, metadata = self.collector.collect_with_metadata(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        self.assertEqual(metadata["batch_count"], 1)
        self.assertEqual(metadata["raw_retrieved_candles"], 0)
        self.assertEqual(metadata["in_range_candles"], 0)
        self.assertEqual(metadata["out_of_range_candles"], 0)
        self.assertEqual(metadata["excluded_candles"], 0)
        self.assertEqual(metadata["retained_candles"], 0)
        self.assertIsNone(metadata["first_candle"])
        self.assertIsNone(metadata["last_candle"])

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_collect_with_metadata_tracks_multiple_batches(
        self,
        mock_copy_rates,
    ):
        self.collector.batch_size = 2

        first_candle = self.start
        second_candle = self.start + timedelta(minutes=5)
        third_candle = self.start + timedelta(minutes=10)

        mock_copy_rates.side_effect = [
            [
                make_rate(first_candle),
                make_rate(second_candle),
            ],
            [
                make_rate(third_candle),
            ],
        ]

        result, metadata = self.collector.collect_with_metadata(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertEqual(mock_copy_rates.call_count, 2)
        self.assertEqual(metadata["batch_count"], 2)
        self.assertEqual(metadata["raw_retrieved_candles"], 3)
        self.assertEqual(metadata["retained_candles"], 2)

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_symbol_enablement_failure_stops_collection(
        self,
        mock_copy_rates,
    ):
        self.market_data_mock.enable_symbol.return_value = False

        with self.assertRaisesRegex(
            RuntimeError,
            "Unable to enable MT5 broker symbol",
        ):
            self.collector.collect(
                "EURUSD",
                "M5",
                self.start,
                self.end,
            )

        mock_copy_rates.assert_not_called()

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_collect_remains_dataframe_compatible(
        self,
        mock_copy_rates,
    ):
        mock_copy_rates.return_value = [
            make_rate(self.start),
        ]

        result = self.collector.collect(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("time", result.columns)
        self.assertIn("is_forming_candle", result.columns)

    @patch("research.historical_data.mt5.copy_rates_range")
    def test_metadata_tracks_out_of_range_candles(
        self,
        mock_copy_rates,
    ):
        before_start = self.start - timedelta(minutes=5)
        first_candle = self.start
        second_candle = self.start + timedelta(minutes=5)

        mock_copy_rates.return_value = [
            make_rate(before_start),
            make_rate(first_candle),
            make_rate(second_candle),
        ]

        result, metadata = self.collector.collect_with_metadata(
            "EURUSD",
            "M5",
            self.start,
            self.end,
        )

        self.assertEqual(metadata["raw_retrieved_candles"], 3)
        self.assertEqual(metadata["in_range_candles"], 2)
        self.assertEqual(metadata["out_of_range_candles"], 1)
        self.assertEqual(metadata["retained_candles"], 2)
        self.assertEqual(len(result), 2)
if __name__ == "__main__":
    unittest.main()

