
import unittest

import numpy as np
import pandas as pd

from research.swing_detection.atr_calculator import WilderATR


def make_candles():
    """Create a deterministic UTC OHLC dataset."""

    times = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=5,
        freq="15min",
        tz="UTC",
    )

    return pd.DataFrame({
        "time": times,
        "open": [9, 10, 11, 12, 13],
        "high": [10, 12, 13, 14, 15],
        "low": [8, 9, 10, 11, 12],
        "close": [9, 11, 12, 13, 14],
        "is_forming_candle": [False] * 5,
    })


def make_gap_candles():
    """Create candles containing upward and downward price gaps."""

    times = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=3,
        freq="15min",
        tz="UTC",
    )

    return pd.DataFrame({
        "time": times,
        "open": [100, 104, 102],
        "high": [102, 106, 104],
        "low": [100, 104, 101],
        "close": [101, 105, 102],
        "is_forming_candle": [False] * 3,
    })


class TestWilderATR(unittest.TestCase):

    # ---------------------------------------------------------
    # MATHEMATICAL VALIDATION
    # ---------------------------------------------------------

    def test_initial_atr_uses_arithmetic_mean(self):
        candles = make_candles()
        result = WilderATR(period=3).calculate(candles)

        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[1]))

        self.assertAlmostEqual(result.iloc[2], 8 / 3, places=10)

    def test_wilder_smoothing_is_correct(self):
        candles = make_candles()
        result = WilderATR(period=3).calculate(candles)

        expected = (((8 / 3) * 2) + 3) / 3

        self.assertAlmostEqual(result.iloc[3], expected, places=10)

    def test_multiple_wilder_smoothing_steps(self):
        candles = make_candles()
        result = WilderATR(period=3).calculate(candles)

        first_atr = 8 / 3
        second_atr = ((first_atr * 2) + 3) / 3
        third_atr = ((second_atr * 2) + 3) / 3

        self.assertAlmostEqual(result.iloc[3], second_atr, places=10)
        self.assertAlmostEqual(result.iloc[4], third_atr, places=10)

    def test_price_gaps_are_included_in_true_range(self):
        candles = make_gap_candles()
        result = WilderATR(period=2).calculate(candles)

        self.assertAlmostEqual(result.iloc[1], 3.5, places=10)
        self.assertAlmostEqual(result.iloc[2], 3.75, places=10)

    def test_period_one_returns_true_range(self):
        candles = make_candles()
        result = WilderATR(period=1).calculate(candles)

        expected = [2, 3, 3, 3, 3]

        np.testing.assert_allclose(result.to_numpy(), expected)

    # ---------------------------------------------------------
    # OUTPUT VALIDATION
    # ---------------------------------------------------------

    def test_output_preserves_original_index(self):
        candles = make_candles()
        candles.index = [10, 20, 30, 40, 50]

        result = WilderATR(period=3).calculate(candles)

        self.assertEqual(result.index.tolist(), candles.index.tolist())

    def test_output_has_expected_name(self):
        result = WilderATR(period=14).calculate(make_candles())

        self.assertEqual(result.name, "atr_14")

    def test_output_is_float_series(self):
        result = WilderATR(period=3).calculate(make_candles())

        self.assertIsInstance(result, pd.Series)
        self.assertTrue(pd.api.types.is_float_dtype(result.dtype))

    def test_insufficient_history_returns_nan(self):
        candles = make_candles().iloc[:3]
        result = WilderATR(period=14).calculate(candles)

        self.assertTrue(result.isna().all())

    def test_calculation_does_not_modify_original_data(self):
        candles = make_candles()
        original = candles.copy(deep=True)

        WilderATR(period=3).calculate(candles)

        pd.testing.assert_frame_equal(candles, original)

    # ---------------------------------------------------------
    # CANDLE VALIDATION
    # ---------------------------------------------------------

    def test_forming_candles_are_rejected(self):
        candles = make_candles()
        candles.loc[4, "is_forming_candle"] = True

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_missing_forming_candle_values_are_rejected(self):
        candles = make_candles()
        candles["is_forming_candle"] = pd.Series(
            [False, False, None, False, False],
            dtype="boolean",
        )

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_non_boolean_forming_indicators_are_rejected(self):
        candles = make_candles()
        candles["is_forming_candle"] = [0, 0, 0, 0, 0]

        with self.assertRaises(TypeError):
            WilderATR().calculate(candles)

    def test_duplicate_timestamps_are_rejected(self):
        candles = make_candles()
        candles.loc[4, "time"] = candles.loc[3, "time"]

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_non_chronological_timestamps_are_rejected(self):
        candles = make_candles().iloc[::-1].copy()

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_randomly_ordered_timestamps_are_rejected(self):
        candles = make_candles()
        candles = candles.iloc[[0, 2, 1, 3, 4]].copy()

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_non_utc_timestamps_are_rejected(self):
        candles = make_candles()
        candles["time"] = candles["time"].dt.tz_convert(
            "Africa/Blantyre"
        )

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_invalid_timestamps_are_rejected(self):
        candles = make_candles()
        candles["time"] = candles["time"].astype(object)
        candles.loc[2, "time"] = "invalid timestamp"

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_empty_dataframe_is_rejected(self):
        candles = make_candles().iloc[:0]

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_missing_required_columns_are_rejected(self):
        candles = make_candles().drop(columns=["high"])

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    # ---------------------------------------------------------
    # PRICE VALIDATION
    # ---------------------------------------------------------

    def test_invalid_ohlc_is_rejected(self):
        candles = make_candles()
        candles.loc[2, "high"] = 7

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_non_finite_prices_are_rejected(self):
        candles = make_candles()
        candles.loc[2, "close"] = np.nan

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_infinite_prices_are_rejected(self):
        candles = make_candles()
        candles["high"] = candles["high"].astype(float)
        candles.loc[2, "high"] = np.inf

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_zero_prices_are_rejected(self):
        candles = make_candles()
        candles.loc[2, "low"] = 0

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    def test_negative_prices_are_rejected(self):
        candles = make_candles()
        candles.loc[2, "close"] = -1

        with self.assertRaises(ValueError):
            WilderATR().calculate(candles)

    # ---------------------------------------------------------
    # PERIOD VALIDATION
    # ---------------------------------------------------------

    def test_invalid_period_is_rejected(self):
        with self.assertRaises(ValueError):
            WilderATR(period=0)

    def test_negative_period_is_rejected(self):
        with self.assertRaises(ValueError):
            WilderATR(period=-5)

    def test_boolean_period_is_rejected(self):
        with self.assertRaises(TypeError):
            WilderATR(period=True)

    def test_float_period_is_rejected(self):
        with self.assertRaises(TypeError):
            WilderATR(period=14.5)


if __name__ == "__main__":
    unittest.main()
