
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
)


BASE_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def make_multi_bos_bullish_candles():
    """Create completed candles containing two bullish BOS events
    and two independently retired protected levels.
    """

    closes = [
        105, 103, 101, 106, 108, 110,
        108, 106, 104, 109, 112, 114,
        111, 109, 107, 111, 114, 116,
        113, 110, 108, 112, 116, 119,
        110, 105,
    ]

    opens = [closes[0]] + closes[:-1]

    highs = [
        max(open_price, close_price) + 0.2
        for open_price, close_price in zip(opens, closes)
    ]

    lows = [
        min(open_price, close_price) - 0.2
        for open_price, close_price in zip(opens, closes)
    ]

    # Intended swing lows.
    for index, price in {
        2: 100,
        8: 103,
        14: 106,
        20: 107,
    }.items():
        lows[index] = price

    # Intended swing highs.
    for index, price in {
        5: 111,
        11: 115,
        17: 118,
    }.items():
        highs[index] = price

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


class TestMarketStructureMultipleBOS(unittest.TestCase):

    def setUp(self):
        self.engine = MarketStructureEngine(
            left_bars=2,
            right_bars=2,
        )

    def test_two_real_bos_events_create_and_retire_two_levels(self):

        candles = make_multi_bos_bullish_candles()

        result = self.engine.analyze(
            candles=candles,
            timeframe="H1",
        )

        # ---------------------------------------------------------
        # 1. Confirm real fractal swing detection.
        # ---------------------------------------------------------

        self.assertEqual(len(result.swings), 7)

        swing_labels = {
            (
                item.swing.swing_type,
                item.swing.swing_index,
            ): item.classification
            for item in result.classified_swings
        }

        self.assertEqual(swing_labels[("LOW", 8)], "HL")
        self.assertEqual(swing_labels[("HIGH", 11)], "HH")
        self.assertEqual(swing_labels[("LOW", 14)], "HL")
        self.assertEqual(swing_labels[("HIGH", 17)], "HH")
        self.assertEqual(swing_labels[("LOW", 20)], "HL")

        # ---------------------------------------------------------
        # 2. Confirm bullish trend classification.
        # ---------------------------------------------------------

        self.assertEqual(
            result.trend_snapshots[-1].trend,
            "BULLISH",
        )

        # ---------------------------------------------------------
        # 3. Confirm two real bullish BOS events.
        # ---------------------------------------------------------

        self.assertEqual(len(result.bos_events), 2)

        first_bos, second_bos = result.bos_events

        self.assertEqual(first_bos.direction, "BULLISH")
        self.assertEqual(first_bos.break_index, 17)
        self.assertAlmostEqual(first_bos.level_price, 115.0)

        self.assertEqual(second_bos.direction, "BULLISH")
        self.assertEqual(second_bos.break_index, 23)
        self.assertAlmostEqual(second_bos.level_price, 118.0)

        # ---------------------------------------------------------
        # 4. Confirm each BOS protects the correct HL.
        # ---------------------------------------------------------

        self.assertEqual(len(result.protected_swings), 2)

        first_protected, second_protected = (
            result.protected_swings
        )

        self.assertEqual(first_protected.trend, "BULLISH")
        self.assertEqual(
            first_protected.swing.swing.swing_index,
            14,
        )
        self.assertAlmostEqual(first_protected.price, 106.0)
        self.assertEqual(first_protected.activation_index, 18)

        self.assertEqual(second_protected.trend, "BULLISH")
        self.assertEqual(
            second_protected.swing.swing.swing_index,
            20,
        )
        self.assertAlmostEqual(second_protected.price, 107.0)
        self.assertEqual(second_protected.activation_index, 24)

        # Each BOS must establish its own protected swing.
        self.assertIs(
            first_protected.established_by,
            first_bos,
        )
        self.assertIs(
            second_protected.established_by,
            second_bos,
        )

        # ---------------------------------------------------------
        # 5. Confirm two real bearish CHoCH events.
        # ---------------------------------------------------------

        self.assertEqual(len(result.choch_events), 2)

        choch_by_level = {
            event.level_price: event
            for event in result.choch_events
        }

        self.assertEqual(
            set(choch_by_level),
            {106.0, 107.0},
        )

        for level_price in (106.0, 107.0):
            event = choch_by_level[level_price]

            self.assertEqual(event.event_type, "CHoCH")
            self.assertEqual(event.direction, "BEARISH")
            self.assertEqual(event.break_index, 25)
            self.assertEqual(event.prior_trend, "BULLISH")
            self.assertAlmostEqual(event.close_price, 105.0)

        # ---------------------------------------------------------
        # 6. Confirm final protected-level lifecycle.
        # ---------------------------------------------------------

        self.assertEqual(len(result.protected_levels), 2)
        self.assertEqual(len(result.active_protected_levels), 0)
        self.assertEqual(len(result.retired_protected_levels), 2)

        retired_by_price = {
            level.price: level
            for level in result.retired_protected_levels
        }

        self.assertEqual(set(retired_by_price), {106.0, 107.0})

        for level_price in (106.0, 107.0):
            level = retired_by_price[level_price]

            self.assertFalse(level.active)
            self.assertIsNotNone(level.retired_by)

            self.assertEqual(
                level.retired_by.event_type,
                "CHoCH",
            )

            self.assertEqual(
                level.retired_by.break_index,
                25,
            )

            self.assertAlmostEqual(
                level.retired_by.level_price,
                level_price,
            )


if __name__ == "__main__":
    unittest.main()
