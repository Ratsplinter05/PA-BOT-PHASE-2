import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.alignment_engine import AlignmentEngine
from research.swing_detection.trend_models import TrendSnapshot

T0 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
H1 = timedelta(hours=1)
H4 = timedelta(hours=4)
TICK = timedelta(microseconds=1)


def snap(as_of, trend):
    return TrendSnapshot(
        as_of_time=as_of,
        trend=trend,
        latest_high=None,
        latest_low=None,
        reason="test",
    )


class TestAlignmentEngineExtra(unittest.TestCase):

    def setUp(self):
        self.engine = AlignmentEngine()

    def test_h4_snapshot_unusable_until_candle_closes(self):
        h4 = [snap(T0, "BULLISH")]
        h1 = [snap(T0, "BULLISH")]

        before = self.engine.evaluate(h4, h1, T0 + H4 - TICK)
        at_close = self.engine.evaluate(h4, h1, T0 + H4)

        self.assertEqual(before.alignment_state, "H4_MISSING")
        self.assertEqual(at_close.alignment_state, "ALIGNED_BULLISH")
        self.assertEqual(at_close.h4_known_at_time, T0 + H4)

    def test_h1_snapshot_unusable_until_candle_closes(self):
        h4 = [snap(T0 - H4 * 2, "BEARISH")]
        h1 = [snap(T0, "BEARISH")]

        before = self.engine.evaluate(h4, h1, T0 + H1 - TICK)
        at_close = self.engine.evaluate(h4, h1, T0 + H1)

        self.assertEqual(before.alignment_state, "H1_MISSING")
        self.assertEqual(at_close.alignment_state, "ALIGNED_BEARISH")
        self.assertEqual(at_close.h1_known_at_time, T0 + H1)

    def test_latest_eligible_snapshot_wins_and_future_is_ignored(self):
        first = snap(T0, "BULLISH")
        second = snap(T0 + H4, "BEARISH")
        future = snap(T0 + H4 * 2, "BULLISH")
        h1 = [snap(T0, "BEARISH")]

        result = self.engine.evaluate(
            [first, second, future], h1, T0 + H4 * 2 + H1
        )

        self.assertIs(result.h4_snapshot, second)
        self.assertEqual(result.alignment_state, "ALIGNED_BEARISH")

    def test_input_order_does_not_matter(self):
        h4 = [snap(T0, "BULLISH"), snap(T0 + H4, "BEARISH")]
        h1 = [snap(T0, "BEARISH")]
        when = T0 + H4 * 2 + H1

        forward = self.engine.evaluate(h4, h1, when)
        backward = self.engine.evaluate(list(reversed(h4)), h1, when)

        self.assertEqual(forward, backward)

    def test_old_snapshots_never_expire(self):
        # Records current behaviour: there is no maximum snapshot age.
        h4 = [snap(T0, "BULLISH")]
        h1 = [snap(T0, "BULLISH")]

        result = self.engine.evaluate(h4, h1, T0 + timedelta(days=30))

        self.assertEqual(result.alignment_state, "ALIGNED_BULLISH")
        self.assertTrue(result.directional_permission)

    def test_duplicate_timestamp_rejected(self):
        plus_two = timezone(timedelta(hours=2))
        same = [snap(T0, "BULLISH"), snap(T0, "BEARISH")]
        same_instant = [
            snap(T0, "BULLISH"),
            snap(T0.astimezone(plus_two), "BEARISH"),
        ]
        h1 = [snap(T0, "BULLISH")]

        for h4 in (same, same_instant):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.engine.evaluate(h4, h1, T0 + H4 * 2)

    def test_naive_assessment_time_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate([], [], datetime(2026, 1, 5, 8, 0))

    def test_non_snapshot_item_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.evaluate(["bad"], [], T0 + H4)

    def test_unsupported_trend_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported H1 trend"):
            self.engine.evaluate([], [snap(T0, "SIDEWAYS")], T0 + H4)

    def test_non_utc_assessment_time_is_normalized(self):
        plus_two = timezone(timedelta(hours=2))
        h4 = [snap(T0, "BULLISH")]
        h1 = [snap(T0, "BULLISH")]

        result = self.engine.evaluate(
            h4, h1, (T0 + H4).astimezone(plus_two)
        )

        self.assertEqual(result.alignment_state, "ALIGNED_BULLISH")
        self.assertEqual(result.assessment_time, T0 + H4)
        self.assertEqual(result.assessment_time.utcoffset(), timedelta(0))

    def test_full_alignment_truth_table(self):
        directional = {
            ("BULLISH", "BULLISH"): "ALIGNED_BULLISH",
            ("BULLISH", "BEARISH"): "CONFLICT",
            ("BULLISH", "MIXED"): "H1_MIXED",
            ("BULLISH", "UNCONFIRMED"): "H1_UNCONFIRMED",
            ("BEARISH", "BEARISH"): "ALIGNED_BEARISH",
            ("BEARISH", "BULLISH"): "CONFLICT",
            ("BEARISH", "MIXED"): "H1_MIXED",
            ("BEARISH", "UNCONFIRMED"): "H1_UNCONFIRMED",
        }
        trends = ("BULLISH", "BEARISH", "MIXED", "UNCONFIRMED")

        for h4_trend in trends:
            for h1_trend in trends:
                if h4_trend in ("MIXED", "UNCONFIRMED"):
                    want = f"H4_{h4_trend}"
                else:
                    want = directional[(h4_trend, h1_trend)]

                result = self.engine.evaluate(
                    [snap(T0, h4_trend)],
                    [snap(T0, h1_trend)],
                    T0 + H4,
                )
                label = f"{h4_trend}/{h1_trend}"

                self.assertEqual(result.alignment_state, want, msg=label)
                aligned = want.startswith("ALIGNED")
                self.assertEqual(
                    result.directional_permission, aligned, msg=label
                )
                self.assertEqual(
                    result.permitted_direction,
                    h4_trend if aligned else None,
                    msg=label,
                )

    def test_missing_data_states(self):
        both_empty = self.engine.evaluate([], [], T0 + H4)
        h1_empty = self.engine.evaluate(
            [snap(T0, "BULLISH")], [], T0 + H4
        )

        self.assertEqual(both_empty.alignment_state, "H4_MISSING")
        self.assertEqual(h1_empty.alignment_state, "H1_MISSING")
        for result in (both_empty, h1_empty):
            self.assertFalse(result.directional_permission)
            self.assertIsNone(result.permitted_direction)

    def test_h1_missing_is_reported_before_h4_mixed(self):
        # Records current check order: missing data wins over H4 state.
        result = self.engine.evaluate([snap(T0, "MIXED")], [], T0 + H4)

        self.assertEqual(result.alignment_state, "H1_MISSING")


if __name__ == "__main__":
    unittest.main()