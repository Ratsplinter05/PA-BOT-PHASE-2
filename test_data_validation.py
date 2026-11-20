
import unittest

import pandas as pd

from src.data_validation import CandleValidator


class TestCandleValidator(unittest.TestCase):

    def setUp(self):

        self.validator = CandleValidator()

        self.valid_candles = pd.DataFrame({
            "time": pd.date_range(
                start="2026-11-03 10:00:00",
                periods=3,
                freq="5min",
                tz="UTC"
            ),
            "open": [1.1000, 1.1005, 1.1010],
            "high": [1.1010, 1.1015, 1.1020],
            "low": [1.0990, 1.1000, 1.1005],
            "close": [1.1005, 1.1010, 1.1015],
            "tick_volume": [100, 120, 110],
            "spread": [10, 10, 10],
            "real_volume": [0, 0, 0],
            "is_forming_candle": [False, False, False]
        })

    def test_valid_candles_pass(self):

        result = self.validator.validate(
            self.valid_candles,
            timeframe="M5"
        )

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["valid"])
        self.assertEqual(result["candle_count"], 3)

    def test_empty_dataset_fails(self):

        result = self.validator.validate(
            pd.DataFrame()
        )

        self.assertEqual(result["status"], "FAIL")

    def test_missing_columns_fail(self):

        candles = self.valid_candles.drop(
            columns=["open"]
        )

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("Missing required columns" in error
                for error in result["errors"])
        )

    def test_duplicate_timestamps_fail(self):

        candles = self.valid_candles.copy()

        candles.loc[2, "time"] = candles.loc[1, "time"]

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")

    def test_unsorted_timestamps_fail(self):

        candles = self.valid_candles.iloc[
            [1, 0, 2]
        ].reset_index(drop=True)

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")

    def test_missing_price_values_fail(self):

        candles = self.valid_candles.copy()

        candles.loc[1, "close"] = float("nan")

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")

    def test_invalid_ohlc_fails(self):

        candles = self.valid_candles.copy()

        candles.loc[1, "high"] = 1.0000

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")

    def test_non_positive_prices_fail(self):

        candles = self.valid_candles.copy()

        candles.loc[1, "close"] = 0

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")

    def test_forming_candle_fails(self):

        candles = self.valid_candles.copy()

        candles.loc[2, "is_forming_candle"] = True

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")

    def test_numeric_forming_flags_fail(self):

        for values in ([0, 0, 0], [0, 1, 0]):

            with self.subTest(values=values):

                candles = self.valid_candles.copy()
                candles["is_forming_candle"] = values

                result = self.validator.validate(candles)

                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(
                    any(
                        "Boolean values only" in error
                        for error in result["errors"]
                    )
                )

    def test_string_forming_flags_fail(self):

        candles = self.valid_candles.copy()
        candles["is_forming_candle"] = [
            "False",
            "False",
            "False"
        ]

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "Boolean values only" in error
                for error in result["errors"]
            )
        )

    def test_missing_forming_flag_fails(self):

        candles = self.valid_candles.copy()
        candles["is_forming_candle"] = pd.Series(
            [False, pd.NA, False],
            dtype="boolean"
        )

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "missing values" in error
                for error in result["errors"]
            )
        )

    def test_nullable_boolean_false_flags_pass(self):

        candles = self.valid_candles.copy()
        candles["is_forming_candle"] = pd.Series(
            [False, False, False],
            dtype="boolean"
        )

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["valid"])

    def test_future_timestamps_fail(self):

        candles = self.valid_candles.copy()

        candles["time"] = pd.date_range(
            start="2099-01-01 10:00:00",
            periods=3,
            freq="5min",
            tz="UTC"
        )

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")

    def test_non_utc_timestamps_fail(self):

        candles = self.valid_candles.copy()

        candles["time"] = candles["time"].dt.tz_convert(
            "Africa/Blantyre"
        )

        result = self.validator.validate(candles)

        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
