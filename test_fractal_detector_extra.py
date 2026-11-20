import random
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.fractal_detector import FractalSwingDetector

T0 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)


def make_candles(highs, lows):
    mids = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {
            "time": pd.date_range(
                T0, periods=len(highs), freq=pd.Timedelta(hours=1)
            ),
            "open": mids,
            "high": list(highs),
            "low": list(lows),
            "close": mids,
        }
    )


def random_walk_candles(bars=60):
    rng = random.Random(7)
    price = 1.1000
    highs, lows = [], []
    for _ in range(bars):
        price += rng.uniform(-0.002, 0.002)
        spread = rng.uniform(0.0005, 0.0015)
        highs.append(price + spread)
        lows.append(price - spread)
    return make_candles(highs, lows)


class TestFractalDetectorExtra(unittest.TestCase):

    def setUp(self):
        self.detector = FractalSwingDetector(left_bars=2, right_bars=2)

    def single_high_candles(self):
        return make_candles(
            highs=[1.10, 1.11, 1.15, 1.12, 1.11],
            lows=[1.09, 1.10, 1.13, 1.10, 1.09],
        )

    def test_swing_and_confirmation_fields(self):
        swings = self.detector.detect(self.single_high_candles())

        self.assertEqual(len(swings), 1)
        swing = swings[0]
        self.assertEqual(swing.swing_type, "HIGH")
        self.assertEqual(swing.price, 1.15)
        self.assertEqual(swing.swing_index, 2)
        self.assertEqual(swing.confirmation_index, 4)
        self.assertEqual(swing.swing_time, T0 + timedelta(hours=2))
        self.assertEqual(swing.confirmation_time, T0 + timedelta(hours=4))
        self.assertEqual((swing.left_bars, swing.right_bars), (2, 2))

    def test_swing_index_is_positional_not_label(self):
        candles = self.single_high_candles()
        candles.index = [100, 200, 300, 400, 500]

        swings = self.detector.detect(candles)

        self.assertEqual(swings[0].swing_index, 2)
        self.assertEqual(swings[0].confirmation_index, 4)

    def test_equal_adjacent_highs_produce_no_swing(self):
        # Records current behaviour: strict comparison, so a plateau of
        # two equal highs inside the window is not a swing.
        candles = make_candles(
            highs=[1.10, 1.11, 1.15, 1.15, 1.11, 1.10],
            lows=[1.09, 1.10, 1.12, 1.12, 1.10, 1.09],
        )

        self.assertEqual(self.detector.detect(candles), [])

    def test_outside_bar_is_both_high_and_low(self):
        candles = make_candles(
            highs=[1.10, 1.10, 1.20, 1.10, 1.10],
            lows=[1.05, 1.05, 1.00, 1.05, 1.05],
        )

        swings = self.detector.detect(candles)

        self.assertEqual([s.swing_type for s in swings], ["HIGH", "LOW"])
        self.assertEqual([s.price for s in swings], [1.20, 1.00])
        self.assertEqual([s.swing_index for s in swings], [2, 2])

    def test_asymmetric_windows(self):
        detector = FractalSwingDetector(left_bars=1, right_bars=3)
        candles = make_candles(
            highs=[1.10, 1.15, 1.12, 1.11, 1.10],
            lows=[1.08, 1.10, 1.09, 1.08, 1.07],
        )

        swings = detector.detect(candles)

        self.assertEqual(len(swings), 1)
        self.assertEqual(swings[0].swing_type, "HIGH")
        self.assertEqual(swings[0].swing_index, 1)
        self.assertEqual(swings[0].confirmation_index, 4)
        self.assertEqual(
            swings[0].confirmation_time, T0 + timedelta(hours=4)
        )
        self.assertEqual(
            (swings[0].left_bars, swings[0].right_bars), (1, 3)
        )

    def test_too_few_candles_returns_empty(self):
        candles = make_candles(
            highs=[1.10, 1.15, 1.12, 1.11],
            lows=[1.09, 1.13, 1.10, 1.09],
        )

        self.assertEqual(self.detector.detect(candles), [])

    def test_empty_frame_returns_empty(self):
        candles = pd.DataFrame(
            columns=["time", "open", "high", "low", "close"]
        )

        self.assertEqual(self.detector.detect(candles), [])

    def test_no_look_ahead_prefix_stability(self):
        # Adding later candles must never change swings already confirmed.
        candles = random_walk_candles()

        full = self.detector.detect(candles)
        self.assertGreater(len(full), 0)

        for n in range(1, len(candles) + 1):
            prefix = self.detector.detect(candles.iloc[:n])
            expected = [s for s in full if s.confirmation_index < n]
            self.assertEqual(prefix, expected, msg=f"prefix length {n}")

    def test_output_is_sorted(self):
        swings = self.detector.detect(random_walk_candles())

        keys = [
            (s.confirmation_index, s.swing_index, s.swing_type)
            for s in swings
        ]
        self.assertEqual(keys, sorted(keys))

    def test_input_frame_is_not_modified(self):
        candles = self.single_high_candles()
        before = candles.copy()

        self.detector.detect(candles)

        pd.testing.assert_frame_equal(candles, before)

    def test_naive_timestamps_rejected(self):
        candles = self.single_high_candles()
        candles["time"] = candles["time"].dt.tz_localize(None)

        with self.assertRaises(ValueError):
            self.detector.detect(candles)

    def test_non_utc_timestamps_rejected(self):
        candles = self.single_high_candles()
        candles["time"] = candles["time"].dt.tz_convert(
            timezone(timedelta(hours=2))
        )

        with self.assertRaises(ValueError):
            self.detector.detect(candles)

    def test_forming_candle_rejected(self):
        candles = self.single_high_candles()
        candles["is_forming_candle"] = [False, False, False, False, True]

        with self.assertRaisesRegex(ValueError, "Forming candles"):
            self.detector.detect(candles)

    def test_non_boolean_forming_flag_rejected(self):
        candles = self.single_high_candles()
        candles["is_forming_candle"] = "False"

        with self.assertRaises(TypeError):
            self.detector.detect(candles)

    def test_false_forming_flags_accepted(self):
        candles = self.single_high_candles()
        candles["is_forming_candle"] = False

        self.assertEqual(len(self.detector.detect(candles)), 1)

    def test_invalid_window_sizes_rejected(self):
        with self.assertRaises(ValueError):
            FractalSwingDetector(left_bars=0)
        with self.assertRaises(TypeError):
            FractalSwingDetector(right_bars=True)
        with self.assertRaises(TypeError):
            FractalSwingDetector(left_bars=2.0)


if __name__ == "__main__":
    unittest.main()