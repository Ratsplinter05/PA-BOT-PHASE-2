import unittest

import pandas as pd

from research.swing_detection.fractal_detector import FractalSwingDetector
from research.swing_detection.swing_models import SwingPoint


class TestFractalSwingDetector(unittest.TestCase):

    def make_candles(self, highs, lows):
        times = pd.date_range(
            start="2026-01-01 00:00:00",
            periods=len(highs),
            freq="5min",
            tz="UTC",
        )

        rows = []

        for high, low in zip(highs, lows):
            rows.append({
                "time": times[len(rows)],
                "open": low + (high - low) * 0.40,
                "high": high,
                "low": low,
                "close": low + (high - low) * 0.60,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0,
                "is_forming_candle": False,
            })

        return pd.DataFrame(rows)

    def test_detects_swing_high(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        detector = FractalSwingDetector(left_bars=2, right_bars=2)
        swings = detector.detect(candles)

        highs = [s for s in swings if s.swing_type == "HIGH"]

        self.assertEqual(len(highs), 1)
        self.assertEqual(highs[0].swing_index, 2)
        self.assertEqual(highs[0].price, 15.0)

    def test_detects_swing_low(self):
        candles = self.make_candles(
            highs=[20, 21, 22, 23, 24],
            lows=[10, 8, 5, 8, 9],
        )

        detector = FractalSwingDetector(left_bars=2, right_bars=2)
        swings = detector.detect(candles)

        lows = [s for s in swings if s.swing_type == "LOW"]

        self.assertEqual(len(lows), 1)
        self.assertEqual(lows[0].swing_index, 2)
        self.assertEqual(lows[0].price, 5.0)

    def test_confirmation_timing_is_correct(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        detector = FractalSwingDetector(2, 2)
        swings = detector.detect(candles)

        swing = next(s for s in swings if s.swing_type == "HIGH")

        self.assertEqual(swing.swing_index, 2)
        self.assertEqual(swing.confirmation_index, 4)

        self.assertEqual(
            swing.swing_time,
            candles.iloc[2]["time"].to_pydatetime(),
        )

        self.assertEqual(
            swing.confirmation_time,
            candles.iloc[4]["time"].to_pydatetime(),
        )

        self.assertEqual(swing.left_bars, 2)
        self.assertEqual(swing.right_bars, 2)

    def test_equal_highs_do_not_create_swing_high(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 15, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        detector = FractalSwingDetector(2, 2)
        swings = detector.detect(candles)

        self.assertFalse(
            any(s.swing_type == "HIGH" for s in swings)
        )

    def test_equal_lows_do_not_create_swing_low(self):
        candles = self.make_candles(
            highs=[20, 21, 22, 23, 24],
            lows=[10, 8, 5, 5, 9],
        )

        detector = FractalSwingDetector(2, 2)
        swings = detector.detect(candles)

        self.assertFalse(
            any(s.swing_type == "LOW" for s in swings)
        )

    def test_insufficient_candles_returns_empty_list(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12],
            lows=[5, 4, 4.5, 4.2],
        )

        detector = FractalSwingDetector(2, 2)

        self.assertEqual(detector.detect(candles), [])

    def test_forming_candle_is_rejected(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        candles.loc[4, "is_forming_candle"] = True

        detector = FractalSwingDetector(2, 2)

        with self.assertRaisesRegex(ValueError, "Forming candles"):
            detector.detect(candles)

    def test_naive_timestamps_are_rejected(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        candles["time"] = candles["time"].dt.tz_localize(None)

        detector = FractalSwingDetector(2, 2)

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            detector.detect(candles)

    def test_unsorted_timestamps_are_rejected(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        candles = candles.iloc[::-1].reset_index(drop=True)

        detector = FractalSwingDetector(2, 2)

        with self.assertRaisesRegex(ValueError, "ascending"):
            detector.detect(candles)

    def test_invalid_ohlc_is_rejected(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        candles.loc[2, "high"] = 1

        detector = FractalSwingDetector(2, 2)

        with self.assertRaisesRegex(ValueError, "OHLC"):
            detector.detect(candles)

    def test_non_default_dataframe_index_is_supported(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        candles.index = [100, 200, 300, 400, 500]

        detector = FractalSwingDetector(2, 2)
        swings = detector.detect(candles)

        swing = next(s for s in swings if s.swing_type == "HIGH")

        self.assertEqual(swing.swing_index, 2)

    def test_original_dataframe_is_not_modified(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[5, 4, 4.5, 4.2, 4.1],
        )

        original = candles.copy(deep=True)

        detector = FractalSwingDetector(2, 2)
        detector.detect(candles)

        pd.testing.assert_frame_equal(candles, original)

    def test_invalid_window_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            FractalSwingDetector(left_bars=0, right_bars=2)

        with self.assertRaises(ValueError):
            FractalSwingDetector(left_bars=2, right_bars=0)

        with self.assertRaises(TypeError):
            FractalSwingDetector(left_bars=2.5, right_bars=2)

        with self.assertRaises(TypeError):
            FractalSwingDetector(left_bars=True, right_bars=2)

    def test_both_swing_types_can_be_detected_on_same_candle(self):
        candles = self.make_candles(
            highs=[10, 12, 15, 12, 11],
            lows=[8, 7, 5, 7, 8],
        )

        detector = FractalSwingDetector(2, 2)
        swings = detector.detect(candles)

        swing_types = {
            swing.swing_type
            for swing in swings
            if swing.swing_index == 2
        }

        self.assertEqual(swing_types, {"HIGH", "LOW"})


if __name__ == "__main__":
    unittest.main()
