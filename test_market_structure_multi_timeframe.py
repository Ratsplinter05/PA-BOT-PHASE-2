
import unittest
from datetime import timedelta

import pandas as pd

from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
)

from test_market_structure_end_to_end import (
    BASE_TIME,
    make_bullish_candles,
    make_bearish_candles,
)


def apply_timeframe(candles, hours):
    """Return a copy with timestamps spaced for the requested timeframe."""

    result = candles.copy()

    result["time"] = [
        BASE_TIME + timedelta(hours=index * hours)
        for index in range(len(result))
    ]

    return result


def make_flat_candles(hours=1, count=16):
    """Create valid completed candles with no identifiable swings."""

    return pd.DataFrame({
        "time": [
            BASE_TIME + timedelta(hours=index * hours)
            for index in range(count)
        ],
        "open": [100.0] * count,
        "high": [100.1] * count,
        "low": [99.9] * count,
        "close": [100.0] * count,
    })


class TestMarketStructureMultiTimeframe(unittest.TestCase):

    def setUp(self):
        self.engine = MarketStructureEngine(
            left_bars=2,
            right_bars=2,
        )

        self.assessment_time = BASE_TIME + timedelta(hours=60)

    def analyze_pair(self, h4_candles, h1_candles):
        return self.engine.analyze_multi_timeframe(
            h4_candles=h4_candles,
            h1_candles=h1_candles,
            assessment_time=self.assessment_time,
        )

    def test_h4_bullish_and_h1_bullish(self):

        result = self.analyze_pair(
            apply_timeframe(make_bullish_candles(), 4),
            apply_timeframe(make_bullish_candles(), 1),
        )

        self.assertEqual(
            result.h4_analysis.trend_snapshots[-1].trend,
            "BULLISH",
        )

        self.assertEqual(
            result.h1_analysis.trend_snapshots[-1].trend,
            "BULLISH",
        )

        self.assertTrue(result.alignment.directional_permission)
        self.assertEqual(
            result.alignment.permitted_direction,
            "BULLISH",
        )

    def test_h4_bearish_and_h1_bearish(self):

        result = self.analyze_pair(
            apply_timeframe(make_bearish_candles(), 4),
            apply_timeframe(make_bearish_candles(), 1),
        )

        self.assertEqual(
            result.h4_analysis.trend_snapshots[-1].trend,
            "BEARISH",
        )

        self.assertEqual(
            result.h1_analysis.trend_snapshots[-1].trend,
            "BEARISH",
        )

        self.assertTrue(result.alignment.directional_permission)
        self.assertEqual(
            result.alignment.permitted_direction,
            "BEARISH",
        )

    def test_h4_bullish_and_h1_bearish_conflict(self):

        result = self.analyze_pair(
            apply_timeframe(make_bullish_candles(), 4),
            apply_timeframe(make_bearish_candles(), 1),
        )

        self.assertEqual(
            result.h4_analysis.trend_snapshots[-1].trend,
            "BULLISH",
        )

        self.assertEqual(
            result.h1_analysis.trend_snapshots[-1].trend,
            "BEARISH",
        )

        self.assertFalse(result.alignment.directional_permission)
        self.assertIsNone(result.alignment.permitted_direction)

    def test_h4_bearish_and_h1_bullish_conflict(self):

        result = self.analyze_pair(
            apply_timeframe(make_bearish_candles(), 4),
            apply_timeframe(make_bullish_candles(), 1),
        )

        self.assertEqual(
            result.h4_analysis.trend_snapshots[-1].trend,
            "BEARISH",
        )

        self.assertEqual(
            result.h1_analysis.trend_snapshots[-1].trend,
            "BULLISH",
        )

        self.assertFalse(result.alignment.directional_permission)
        self.assertIsNone(result.alignment.permitted_direction)

    def test_missing_h1_structure_blocks_directional_permission(self):

        result = self.analyze_pair(
            apply_timeframe(make_bullish_candles(), 4),
            make_flat_candles(hours=1),
        )

        self.assertEqual(
            result.h4_analysis.trend_snapshots[-1].trend,
            "BULLISH",
        )

        self.assertEqual(
            result.h1_analysis.trend_snapshots,
            (),
        )

        self.assertFalse(result.alignment.directional_permission)
        self.assertIsNone(result.alignment.permitted_direction)

    def test_h4_snapshot_is_not_eligible_before_confirmation_boundary(self):

        h4_candles = apply_timeframe(
            make_bullish_candles(),
            4,
        )

        h1_candles = apply_timeframe(
            make_bullish_candles(),
            1,
        )

        assessment_time = BASE_TIME + timedelta(
            hours=55,
            minutes=59,
        )

        result = self.engine.analyze_multi_timeframe(
            h4_candles=h4_candles,
            h1_candles=h1_candles,
            assessment_time=assessment_time,
        )

        self.assertFalse(
            result.alignment.directional_permission
        )

        self.assertIsNone(
            result.alignment.permitted_direction
        )

    def test_h4_snapshot_becomes_eligible_at_confirmation_boundary(self):

        h4_candles = apply_timeframe(
            make_bullish_candles(),
            4,
        )

        h1_candles = apply_timeframe(
            make_bullish_candles(),
            1,
        )

        assessment_time = BASE_TIME + timedelta(hours=56)

        result = self.engine.analyze_multi_timeframe(
            h4_candles=h4_candles,
            h1_candles=h1_candles,
            assessment_time=assessment_time,
        )

        self.assertTrue(
            result.alignment.directional_permission
        )

        self.assertEqual(
            result.alignment.permitted_direction,
            "BULLISH",
        )

        self.assertEqual(
            result.alignment.h4_known_at_time,
            assessment_time,
        )


if __name__ == "__main__":
    unittest.main()
