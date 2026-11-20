
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
)


BASE_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)

BULLISH_CLOSES = [
    105.0,
    104.0,
    102.0,
    106.0,
    108.0,
    109.0,
    107.0,
    105.0,
    106.0,
    108.0,
    111.0,
    112.0,
    110.0,
    111.0,
    116.0,
    102.0,
]


def make_bullish_candles():
    """Create completed candles containing a bullish BOS and CHoCH."""

    closes = BULLISH_CLOSES.copy()

    opens = [closes[0]] + closes[:-1]

    # Avoid creating an unnecessary upper wick on the CHoCH candle.
    opens[15] = 102.0

    highs = [
        max(open_price, close_price) + 0.1
        for open_price, close_price in zip(opens, closes)
    ]

    lows = [
        min(open_price, close_price) - 0.1
        for open_price, close_price in zip(opens, closes)
    ]

    # First confirmed swing low.
    lows[2] = 100.0

    # First confirmed swing high.
    highs[5] = 110.0

    # Higher low.
    lows[8] = 103.0

    # Higher high.
    highs[11] = 115.0

    return pd.DataFrame({
        "time": [
            BASE_TIME + timedelta(hours=index)
            for index in range(len(closes))
        ],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })


def make_bearish_candles():
    """Mirror the bullish sequence to create bearish market structure."""

    bullish = make_bullish_candles()

    return pd.DataFrame({
        "time": bullish["time"],
        "open": 200.0 - bullish["open"],
        "high": 200.0 - bullish["low"],
        "low": 200.0 - bullish["high"],
        "close": 200.0 - bullish["close"],
    })


class TestMarketStructureEndToEnd(unittest.TestCase):

    def setUp(self):
        self.engine = MarketStructureEngine(
            left_bars=2,
            right_bars=2,
        )

    def test_real_bullish_bos_and_bearish_choch_sequence(self):

        candles = make_bullish_candles()

        result = self.engine.analyze(
            candles=candles,
            timeframe="H1",
        )

        # Verify actual fractal detection.
        self.assertEqual(len(result.swings), 4)

        swing_labels = {
            (
                item.swing.swing_type,
                item.swing.swing_index,
            ): item.classification
            for item in result.classified_swings
        }

        self.assertIsNone(swing_labels[("LOW", 2)])
        self.assertIsNone(swing_labels[("HIGH", 5)])
        self.assertEqual(swing_labels[("LOW", 8)], "HL")
        self.assertEqual(swing_labels[("HIGH", 11)], "HH")

        # Verify actual bullish trend classification.
        self.assertEqual(
            result.trend_snapshots[-1].trend,
            "BULLISH",
        )

        self.assertEqual(
            result.trend_snapshots[-1].as_of_time,
            BASE_TIME + timedelta(hours=13),
        )

        # Verify actual BOS detection.
        self.assertEqual(len(result.bos_events), 1)

        bullish_bos = result.bos_events[0]

        self.assertEqual(bullish_bos.direction, "BULLISH")
        self.assertEqual(bullish_bos.break_index, 14)
        self.assertAlmostEqual(bullish_bos.level_price, 115.0)

        # Verify the correct HL becomes protected.
        self.assertEqual(len(result.protected_swings), 1)

        protected = result.protected_swings[0]

        self.assertEqual(protected.trend, "BULLISH")
        self.assertEqual(protected.swing.swing.swing_index, 8)
        self.assertEqual(protected.activation_index, 15)

        # Verify actual bearish CHoCH detection.
        self.assertEqual(len(result.choch_events), 1)

        bearish_choch = result.choch_events[0]

        self.assertEqual(bearish_choch.direction, "BEARISH")
        self.assertEqual(bearish_choch.break_index, 15)
        self.assertAlmostEqual(bearish_choch.level_price, 103.0)

        # Verify final protected-level lifecycle.
        self.assertEqual(len(result.protected_levels), 1)
        self.assertEqual(len(result.active_protected_levels), 0)
        self.assertEqual(len(result.retired_protected_levels), 1)

        self.assertFalse(
            result.retired_protected_levels[0].active
        )

    def test_real_bearish_bos_and_bullish_choch_sequence(self):

        candles = make_bearish_candles()

        result = self.engine.analyze(
            candles=candles,
            timeframe="H1",
        )

        # Verify actual fractal detection.
        self.assertEqual(len(result.swings), 4)

        swing_labels = {
            (
                item.swing.swing_type,
                item.swing.swing_index,
            ): item.classification
            for item in result.classified_swings
        }

        self.assertIsNone(swing_labels[("HIGH", 2)])
        self.assertIsNone(swing_labels[("LOW", 5)])
        self.assertEqual(swing_labels[("HIGH", 8)], "LH")
        self.assertEqual(swing_labels[("LOW", 11)], "LL")

        # Verify actual bearish trend classification.
        self.assertEqual(
            result.trend_snapshots[-1].trend,
            "BEARISH",
        )

        # Verify actual bearish BOS detection.
        self.assertEqual(len(result.bos_events), 1)

        bearish_bos = result.bos_events[0]

        self.assertEqual(bearish_bos.direction, "BEARISH")
        self.assertEqual(bearish_bos.break_index, 14)
        self.assertAlmostEqual(bearish_bos.level_price, 85.0)

        # Verify the correct LH becomes protected.
        self.assertEqual(len(result.protected_swings), 1)

        protected = result.protected_swings[0]

        self.assertEqual(protected.trend, "BEARISH")
        self.assertEqual(protected.swing.swing.swing_index, 8)
        self.assertEqual(protected.activation_index, 15)

        # Verify actual bullish CHoCH detection.
        self.assertEqual(len(result.choch_events), 1)

        bullish_choch = result.choch_events[0]

        self.assertEqual(bullish_choch.direction, "BULLISH")
        self.assertEqual(bullish_choch.break_index, 15)
        self.assertAlmostEqual(bullish_choch.level_price, 97.0)

        # Verify final protected-level lifecycle.
        self.assertEqual(len(result.protected_levels), 1)
        self.assertEqual(len(result.active_protected_levels), 0)
        self.assertEqual(len(result.retired_protected_levels), 1)

        self.assertFalse(
            result.retired_protected_levels[0].active
        )


if __name__ == "__main__":
    unittest.main()
