
import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.alignment_engine import AlignmentEngine
from research.swing_detection.trend_models import TrendSnapshot


class TestAlignmentEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AlignmentEngine()
        self.base_time = datetime(
            2026, 11, 3, 12, 0, tzinfo=timezone.utc
        )

    def make_snapshot(self, trend, offset_hours):
        return TrendSnapshot(
            as_of_time=self.base_time + timedelta(hours=offset_hours),
            trend=trend,
            latest_high=None,
            latest_low=None,
            reason=f"Test trend: {trend}",
        )

    def evaluate(self, h4_trend, h1_trend):
        assessment_time = datetime(
            2026, 11, 4, 12, 0, tzinfo=timezone.utc
        )

        h4 = (
            [self.make_snapshot(h4_trend, 0)]
            if h4_trend is not None else []
        )

        h1 = (
            [self.make_snapshot(h1_trend, 0)]
            if h1_trend is not None else []
        )

        return self.engine.evaluate(
            h4_snapshots=h4,
            h1_snapshots=h1,
            assessment_time=assessment_time,
        )

    def test_aligned_bullish(self):
        result = self.evaluate("BULLISH", "BULLISH")

        self.assertEqual(result.alignment_state, "ALIGNED_BULLISH")
        self.assertTrue(result.directional_permission)
        self.assertEqual(result.permitted_direction, "BULLISH")

    def test_aligned_bearish(self):
        result = self.evaluate("BEARISH", "BEARISH")

        self.assertEqual(result.alignment_state, "ALIGNED_BEARISH")
        self.assertTrue(result.directional_permission)
        self.assertEqual(result.permitted_direction, "BEARISH")

    def test_bullish_bearish_conflict(self):
        result = self.evaluate("BULLISH", "BEARISH")

        self.assertEqual(result.alignment_state, "CONFLICT")
        self.assertFalse(result.directional_permission)
        self.assertIsNone(result.permitted_direction)

    def test_bearish_bullish_conflict(self):
        result = self.evaluate("BEARISH", "BULLISH")

        self.assertEqual(result.alignment_state, "CONFLICT")
        self.assertFalse(result.directional_permission)

    def test_h4_mixed_blocks_directional_permission(self):
        result = self.evaluate("MIXED", "BEARISH")

        self.assertEqual(result.alignment_state, "H4_MIXED")
        self.assertFalse(result.directional_permission)

    def test_h4_unconfirmed_blocks_directional_permission(self):
        result = self.evaluate("UNCONFIRMED", "BULLISH")

        self.assertEqual(result.alignment_state, "H4_UNCONFIRMED")
        self.assertFalse(result.directional_permission)

    def test_h1_mixed_blocks_directional_permission(self):
        result = self.evaluate("BULLISH", "MIXED")

        self.assertEqual(result.alignment_state, "H1_MIXED")
        self.assertFalse(result.directional_permission)

    def test_h1_unconfirmed_blocks_directional_permission(self):
        result = self.evaluate("BEARISH", "UNCONFIRMED")

        self.assertEqual(result.alignment_state, "H1_UNCONFIRMED")
        self.assertFalse(result.directional_permission)

    def test_missing_h4_snapshot(self):
        result = self.evaluate(None, "BULLISH")

        self.assertEqual(result.alignment_state, "H4_MISSING")
        self.assertFalse(result.directional_permission)
        self.assertIsNone(result.h4_snapshot)

    def test_missing_h1_snapshot(self):
        result = self.evaluate("BULLISH", None)

        self.assertEqual(result.alignment_state, "H1_MISSING")
        self.assertFalse(result.directional_permission)
        self.assertIsNone(result.h1_snapshot)

    def test_future_snapshots_are_not_selected(self):
        assessment_time = datetime(
            2026, 11, 3, 17, 0, tzinfo=timezone.utc
        )

        h4 = [
            self.make_snapshot("MIXED", -4),
            self.make_snapshot("BULLISH", 4),
        ]

        h1 = [
            self.make_snapshot("BEARISH", 3),
            self.make_snapshot("BULLISH", 6),
        ]

        result = self.engine.evaluate(
            h4_snapshots=h4,
            h1_snapshots=h1,
            assessment_time=assessment_time,
        )

        self.assertEqual(result.h4_snapshot.trend, "MIXED")
        self.assertEqual(result.h1_snapshot.trend, "BEARISH")

        self.assertLessEqual(
            result.h4_known_at_time,
            assessment_time,
        )

        self.assertLessEqual(
            result.h1_known_at_time,
            assessment_time,
        )

    def test_latest_eligible_snapshot_is_selected(self):
        assessment_time = datetime(
            2026, 11, 4, 12, 0, tzinfo=timezone.utc
        )

        h4 = [
            self.make_snapshot("BEARISH", -12),
            self.make_snapshot("BULLISH", -8),
        ]

        h1 = [
            self.make_snapshot("BEARISH", -3),
            self.make_snapshot("BULLISH", -2),
        ]

        result = self.engine.evaluate(
            h4_snapshots=h4,
            h1_snapshots=h1,
            assessment_time=assessment_time,
        )

        self.assertEqual(result.h4_snapshot.trend, "BULLISH")
        self.assertEqual(result.h1_snapshot.trend, "BULLISH")

    def test_snapshot_known_exactly_at_assessment_time_is_eligible(self):
        assessment_time = datetime(
            2026, 11, 3, 16, 0, tzinfo=timezone.utc
        )

        h4_snapshot = TrendSnapshot(
            as_of_time=assessment_time - timedelta(hours=4),
            trend="BULLISH",
            latest_high=None,
            latest_low=None,
            reason="Boundary test",
        )

        h1_snapshot = TrendSnapshot(
            as_of_time=assessment_time - timedelta(hours=1),
            trend="BULLISH",
            latest_high=None,
            latest_low=None,
            reason="Boundary test",
        )

        result = self.engine.evaluate(
            h4_snapshots=[h4_snapshot],
            h1_snapshots=[h1_snapshot],
            assessment_time=assessment_time,
        )

        self.assertEqual(result.alignment_state, "ALIGNED_BULLISH")
        self.assertEqual(result.h4_known_at_time, assessment_time)
        self.assertEqual(result.h1_known_at_time, assessment_time)

    def test_naive_assessment_timestamp_is_rejected(self):
        naive_time = datetime(2026, 11, 3, 17, 0)

        with self.assertRaises(ValueError):
            self.engine.evaluate(
                h4_snapshots=[],
                h1_snapshots=[],
                assessment_time=naive_time,
            )

    def test_naive_snapshot_timestamp_is_rejected(self):
        naive_snapshot = TrendSnapshot(
            as_of_time=datetime(2026, 11, 3, 12, 0),
            trend="BULLISH",
            latest_high=None,
            latest_low=None,
            reason="Invalid timestamp test",
        )

        with self.assertRaises(ValueError):
            self.engine.evaluate(
                h4_snapshots=[naive_snapshot],
                h1_snapshots=[],
                assessment_time=datetime(
                    2026, 11, 4, 12, 0, tzinfo=timezone.utc
                ),
            )

    # Additional validation tests

    def test_invalid_h4_trend_state_is_rejected(self):
        h4 = [self.make_snapshot("INVALID", 0)]
        h1 = [self.make_snapshot("MIXED", 0)]

        with self.assertRaisesRegex(ValueError, "Unsupported H4 trend"):
            self.engine.evaluate(
                h4_snapshots=h4,
                h1_snapshots=h1,
                assessment_time=datetime(
                    2026, 11, 4, 12, 0, tzinfo=timezone.utc
                ),
            )

    def test_invalid_h1_trend_state_is_rejected(self):
        h4 = [self.make_snapshot("MIXED", 0)]
        h1 = [self.make_snapshot("INVALID", 0)]

        with self.assertRaisesRegex(ValueError, "Unsupported H1 trend"):
            self.engine.evaluate(
                h4_snapshots=h4,
                h1_snapshots=h1,
                assessment_time=datetime(
                    2026, 11, 4, 12, 0, tzinfo=timezone.utc
                ),
            )

    def test_duplicate_snapshot_timestamps_are_rejected(self):
        assessment_time = datetime(
            2026, 11, 4, 12, 0, tzinfo=timezone.utc
        )

        duplicate_h4 = [
            self.make_snapshot("BULLISH", -8),
            self.make_snapshot("BEARISH", -8),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.engine.evaluate(
                h4_snapshots=duplicate_h4,
                h1_snapshots=[],
                assessment_time=assessment_time,
            )


if __name__ == "__main__":
    unittest.main()
