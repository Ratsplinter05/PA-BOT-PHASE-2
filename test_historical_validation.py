import unittest

import numpy as np
import pandas as pd

from research.historical_validation import HistoricalDataValidator


class TestHistoricalDataValidator(unittest.TestCase):

    def setUp(self):
        self.validator = HistoricalDataValidator()

        self.data = pd.DataFrame({
            "time": pd.date_range(
                "2025-01-01 00:00:00",
                periods=3,
                freq="5min",
                tz="UTC",
            ),
            "open": [1.1000, 1.1010, 1.1020],
            "high": [1.1050, 1.1060, 1.1070],
            "low": [1.0950, 1.0960, 1.0970],
            "close": [1.1020, 1.1030, 1.1040],
            "tick_volume": [100, 120, 110],
            "spread": [10, 10, 12],
            "real_volume": [0, 0, 0],
            "is_forming_candle": [False, False, False],
        })

    def test_valid_dataset_passes(self):
        result = self.validator.validate(self.data, "M5")

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["valid"])
        self.assertEqual(result["candle_count"], 3)

    def test_empty_dataset_fails(self):
        result = self.validator.validate(pd.DataFrame(), "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["valid"])

    def test_invalid_timeframe_fails(self):
        result = self.validator.validate(self.data, "M1")

        self.assertEqual(result["status"], "FAIL")

    def test_missing_columns_fail(self):
        data = self.data.drop(columns=["spread"])

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("Missing required columns" in error
                for error in result["errors"])
        )

    def test_naive_timestamps_fail(self):
        data = self.data.copy()
        data["time"] = data["time"].dt.tz_localize(None)

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")

    def test_non_utc_timestamps_fail(self):
        data = self.data.copy()
        data["time"] = data["time"].dt.tz_convert(
            "Africa/Blantyre"
        )

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")

    def test_duplicate_timestamps_fail(self):
        data = self.data.copy()
        data.loc[1, "time"] = data.loc[0, "time"]

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["duplicate_timestamps"], 1)

    def test_unsorted_timestamps_fail(self):
        data = self.data.copy()
        data.loc[1, "time"] = (
            data.loc[0, "time"] - pd.Timedelta(minutes=5)
        )

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(
            result["non_chronological_pairs"],
            0,
        )

    def test_missing_values_fail(self):
        data = self.data.copy()
        data.loc[1, "close"] = np.nan

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(result["missing_values"], 0)

    def test_non_finite_values_fail(self):
        data = self.data.copy()
        data.loc[1, "close"] = np.inf

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(result["non_finite_values"], 0)

    def test_non_positive_prices_fail(self):
        data = self.data.copy()
        data.loc[1, "close"] = 0

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["non_positive_price_rows"], 1)

    def test_invalid_ohlc_relationships_fail(self):
        data = self.data.copy()
        data.loc[1, "high"] = 1.0900

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["invalid_ohlc_rows"], 1)

    def test_negative_volume_fails(self):
        data = self.data.copy()
        data.loc[1, "tick_volume"] = -10

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["negative_value_rows"], 1)

    def test_forming_candles_fail(self):
        data = self.data.copy()
        data.loc[1, "is_forming_candle"] = True

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["forming_candles"], 1)

    def test_missing_candle_interval_produces_warning(self):
        data = self.data.copy()
        data.loc[2, "time"] = (
            data.loc[2, "time"] + pd.Timedelta(minutes=10)
        )

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "WARNING")
        self.assertTrue(result["valid"])
        self.assertEqual(result["gap_count"], 1)
        self.assertEqual(result["unexpected_gap_count"], 1)
        self.assertEqual(result["expected_gap_count"], 0)
        self.assertEqual(
            result["estimated_missing_intervals"],
            2,
        )

    def test_future_timestamps_fail(self):
        data = self.data.copy()
        data.loc[2, "time"] = pd.Timestamp(
            "2099-01-01 00:00:00",
            tz="UTC",
        )

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(result["future_timestamps"], 0)

    def test_original_dataset_is_not_modified(self):
        original = self.data.copy(deep=True)

        self.validator.validate(self.data, "M5")

        pd.testing.assert_frame_equal(self.data, original)

    # New regression tests

    def test_duplicate_dataframe_index_does_not_break_gap_detection(self):
        data = self.data.copy()
        data.loc[2, "time"] = (
            data.loc[2, "time"] + pd.Timedelta(minutes=10)
        )
        data.index = [5, 5, 5]

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["gap_count"], 1)
        self.assertEqual(
            result["gap_details"][0]["previous_candle"],
            "2025-01-01T00:05:00+00:00",
        )

    def test_non_consecutive_dataframe_index_does_not_break_gap_detection(self):
        data = self.data.copy()
        data.loc[2, "time"] = (
            data.loc[2, "time"] + pd.Timedelta(minutes=10)
        )
        data.index = [10, 30, 90]

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["gap_count"], 1)
        self.assertEqual(
            result["estimated_missing_intervals"],
            2,
        )

    def test_multiple_non_numeric_columns_are_reported(self):
        data = self.data.copy()
        data["open"] = data["open"].astype(object)
        data["close"] = data["close"].astype(object)
        data.loc[1, "open"] = "invalid"
        data.loc[2, "close"] = "invalid"

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("Column open must contain numeric values." == error
                for error in result["errors"])
        )
        self.assertTrue(
            any("Column close must contain numeric values." == error
                for error in result["errors"])
        )

    def test_missing_timestamp_fails_without_breaking_gap_detection(self):
        data = self.data.copy()
        data.loc[1, "time"] = pd.NaT

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(result["missing_values"], 0)
        self.assertEqual(result["gap_count"], 1)
        self.assertEqual(
            result["estimated_missing_intervals"],
            1,
        )
        self.assertTrue(
            any("missing values" in error.lower()
                for error in result["errors"])
        )

    def test_valid_dataset_with_non_default_index_passes(self):
        data = self.data.copy()
        data.index = [100, 200, 300]

        result = self.validator.validate(data, "M5")

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()

