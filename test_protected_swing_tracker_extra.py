import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from research.swing_detection.break_models import StructureBreak
from research.swing_detection.protected_swing_tracker import (
    ProtectedSwingTracker,
)
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.swing_models import SwingPoint

BASE = datetime(2026, 5, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def make_swing(swing_type, classification, price, swing_index,
               confirmation_index):
    swing = SwingPoint(
        swing_type=swing_type,
        price=price,
        swing_time=BASE + swing_index * HOUR,
        swing_index=swing_index,
        confirmation_time=BASE + confirmation_index * HOUR,
        confirmation_index=confirmation_index,
        left_bars=2,
        right_bars=2,
    )
    return ClassifiedSwing(
        swing=swing,
        classification=classification,
        previous_price=None,
        previous_swing_time=None,
    )


def make_bos(direction, break_index):
    bullish = direction == "BULLISH"
    level = make_swing(
        "HIGH" if bullish else "LOW",
        "HH" if bullish else "LL",
        1.1200 if bullish else 1.0800,
        1,
        3,
    )
    return StructureBreak(
        event_type="BOS",
        direction=direction,
        break_time=BASE + break_index * HOUR,
        break_index=break_index,
        level_price=level.swing.price,
        level_swing=level,
        close_price=1.1210 if bullish else 1.0790,
        prior_trend=direction,
    )


class TestProtectedSwingTrackerExtra(unittest.TestCase):

    def setUp(self):
        self.tracker = ProtectedSwingTracker()

    def test_bullish_bos_protects_latest_confirmed_hl(self):
        old_hl = make_swing("LOW", "HL", 1.1000, 5, 7)
        new_hl = make_swing("LOW", "HL", 1.1100, 9, 11)
        lower_low = make_swing("LOW", "LL", 1.0900, 10, 12)
        late_hl = make_swing("LOW", "HL", 1.1150, 12, 14)
        lower_high = make_swing("HIGH", "LH", 1.1300, 8, 10)

        result = self.tracker.track(
            [old_hl, new_hl, lower_low, late_hl, lower_high],
            [make_bos("BULLISH", 12)],
        )

        self.assertEqual(len(result), 1)
        self.assertIs(result[0].swing, new_hl)

    def test_bearish_bos_protects_latest_confirmed_lh(self):
        old_lh = make_swing("HIGH", "LH", 1.1200, 5, 7)
        new_lh = make_swing("HIGH", "LH", 1.1150, 9, 11)
        higher_high = make_swing("HIGH", "HH", 1.1300, 10, 12)
        late_lh = make_swing("HIGH", "LH", 1.1100, 12, 14)
        higher_low = make_swing("LOW", "HL", 1.0900, 8, 10)

        result = self.tracker.track(
            [old_lh, new_lh, higher_high, late_lh, higher_low],
            [make_bos("BEARISH", 12)],
        )

        self.assertEqual(len(result), 1)
        self.assertIs(result[0].swing, new_lh)

    def test_protected_swing_fields(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)
        lh = make_swing("HIGH", "LH", 1.1200, 5, 7)
        bull_event = make_bos("BULLISH", 12)
        bear_event = make_bos("BEARISH", 12)

        bull = self.tracker.track([hl], [bull_event])[0]
        bear = self.tracker.track([lh], [bear_event])[0]

        self.assertEqual(bull.trend, "BULLISH")
        self.assertIs(bull.swing, hl)
        self.assertIs(bull.established_by, bull_event)
        self.assertEqual(bull.activation_index, 13)
        self.assertEqual(bull.price, 1.1000)
        self.assertEqual(bear.trend, "BEARISH")
        self.assertIs(bear.swing, lh)
        self.assertEqual(bear.activation_index, 13)
        self.assertEqual(bear.price, 1.1200)

    def test_each_bos_gets_its_own_record_in_break_order(self):
        hl_a = make_swing("LOW", "HL", 1.1000, 5, 7)
        hl_b = make_swing("LOW", "HL", 1.1100, 13, 15)
        first = make_bos("BULLISH", 12)
        second = make_bos("BULLISH", 20)

        result = self.tracker.track([hl_b, hl_a], [second, first])

        self.assertEqual(
            [p.established_by.break_index for p in result], [12, 20]
        )
        self.assertIs(result[0].swing, hl_a)
        self.assertIs(result[1].swing, hl_b)

    def test_repeated_bos_can_protect_the_same_swing(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)

        result = self.tracker.track(
            [hl], [make_bos("BULLISH", 12), make_bos("BULLISH", 14)]
        )

        self.assertEqual(len(result), 2)
        self.assertIs(result[0].swing, hl)
        self.assertIs(result[1].swing, hl)
        self.assertEqual([p.activation_index for p in result], [13, 15])

    def test_no_swings_raises(self):
        with self.assertRaisesRegex(ValueError, "No eligible HL"):
            self.tracker.track([], [make_bos("BULLISH", 12)])

        with self.assertRaisesRegex(ValueError, "No eligible LH"):
            self.tracker.track([], [make_bos("BEARISH", 12)])

    def test_swing_confirmed_on_break_candle_is_not_eligible(self):
        hl = make_swing("LOW", "HL", 1.1000, 10, 12)

        with self.assertRaisesRegex(ValueError, "No eligible HL"):
            self.tracker.track([hl], [make_bos("BULLISH", 12)])

        result = self.tracker.track([hl], [make_bos("BULLISH", 13)])
        self.assertIs(result[0].swing, hl)

    def test_wrong_classification_is_not_used(self):
        only_ll = make_swing("LOW", "LL", 1.0900, 5, 7)

        with self.assertRaisesRegex(ValueError, "No eligible HL"):
            self.tracker.track([only_ll], [make_bos("BULLISH", 12)])

    def test_empty_events_returns_empty(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)

        self.assertEqual(self.tracker.track([hl], []), [])

    def test_accepts_generators(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)

        result = self.tracker.track(
            iter([hl]), iter([make_bos("BULLISH", 12)])
        )

        self.assertEqual(len(result), 1)
        self.assertIs(result[0].swing, hl)

    def test_non_classified_swing_rejected(self):
        with self.assertRaises(TypeError):
            self.tracker.track(["nope"], [make_bos("BULLISH", 12)])

    def test_non_break_event_rejected(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)

        with self.assertRaises(TypeError):
            self.tracker.track([hl], ["nope"])

    def test_choch_event_rejected(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)
        choch = replace(make_bos("BULLISH", 12), event_type="CHoCH")

        with self.assertRaisesRegex(ValueError, "BOS events only"):
            self.tracker.track([hl], [choch])

    def test_prior_trend_must_match_direction(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)
        event = replace(make_bos("BULLISH", 12), prior_trend="BEARISH")

        with self.assertRaisesRegex(ValueError, "prior trend"):
            self.tracker.track([hl], [event])

    def test_invalid_break_index_rejected(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)

        for bad in (-1, True):
            event = replace(make_bos("BULLISH", 12), break_index=bad)
            with self.assertRaisesRegex(ValueError, "break_index"):
                self.tracker.track([hl], [event])

    def test_invalid_direction_rejected(self):
        hl = make_swing("LOW", "HL", 1.1000, 5, 7)
        event = replace(
            make_bos("BULLISH", 12),
            direction="SIDEWAYS",
            prior_trend="SIDEWAYS",
        )

        with self.assertRaisesRegex(ValueError, "Invalid BOS direction"):
            self.tracker.track([hl], [event])


if __name__ == "__main__":
    unittest.main()