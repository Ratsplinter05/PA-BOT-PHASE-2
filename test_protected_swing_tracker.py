import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.break_models import StructureBreak
from research.swing_detection.protected_swing_tracker import (
    ProtectedSwingTracker,
)


BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_classified_swing(
    swing_type,
    price,
    classification,
    confirmation_index,
):
    swing_time = BASE_TIME + timedelta(hours=confirmation_index - 2)
    confirmation_time = BASE_TIME + timedelta(hours=confirmation_index)

    swing = SwingPoint(
        swing_type=swing_type,
        price=price,
        swing_time=swing_time,
        swing_index=confirmation_index - 2,
        confirmation_time=confirmation_time,
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
    break_time = BASE_TIME + timedelta(hours=break_index)

    if direction == "BULLISH":
        level_swing = make_classified_swing(
            "HIGH", 1.1200, "HH", break_index - 3
        )
        level_price = 1.1200
        close_price = 1.1210
    else:
        level_swing = make_classified_swing(
            "LOW", 1.1100, "LL", break_index - 3
        )
        level_price = 1.1100
        close_price = 1.1090

    return StructureBreak(
        event_type="BOS",
        direction=direction,
        break_time=break_time,
        break_index=break_index,
        level_price=level_price,
        level_swing=level_swing,
        close_price=close_price,
        prior_trend=direction,
    )


class TestProtectedSwingTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = ProtectedSwingTracker()

    def test_bullish_bos_protects_latest_higher_low(self):
        swings = [
            make_classified_swing("LOW", 1.1100, "HL", 5),
            make_classified_swing("LOW", 1.1150, "HL", 10),
            make_classified_swing("HIGH", 1.1200, "HH", 12),
        ]

        bos = [make_bos("BULLISH", 15)]

        result = self.tracker.track(swings, bos)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].trend, "BULLISH")
        self.assertEqual(result[0].price, 1.1150)
        self.assertEqual(result[0].activation_index, 16)

    def test_bearish_bos_protects_latest_lower_high(self):
        swings = [
            make_classified_swing("HIGH", 1.1300, "LH", 5),
            make_classified_swing("HIGH", 1.1250, "LH", 10),
            make_classified_swing("LOW", 1.1100, "LL", 12),
        ]

        bos = [make_bos("BEARISH", 15)]

        result = self.tracker.track(swings, bos)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].trend, "BEARISH")
        self.assertEqual(result[0].price, 1.1250)
        self.assertEqual(result[0].activation_index, 16)

    def test_swing_confirmed_on_bos_candle_is_excluded(self):
        swings = [
            make_classified_swing("LOW", 1.1100, "HL", 5),
            make_classified_swing("LOW", 1.1150, "HL", 15),
        ]

        bos = [make_bos("BULLISH", 15)]

        result = self.tracker.track(swings, bos)

        self.assertEqual(result[0].price, 1.1100)

    def test_unclassified_swing_is_not_protected(self):
        swings = [
            make_classified_swing("LOW", 1.1100, None, 5),
            make_classified_swing("LOW", 1.1150, "HL", 10),
        ]

        bos = [make_bos("BULLISH", 15)]

        result = self.tracker.track(swings, bos)

        self.assertEqual(result[0].price, 1.1150)

    def test_wrong_swing_classification_is_not_selected(self):
        swings = [
            make_classified_swing("LOW", 1.1100, "LL", 5),
            make_classified_swing("LOW", 1.1150, "HL", 10),
        ]

        bos = [make_bos("BULLISH", 15)]

        result = self.tracker.track(swings, bos)

        self.assertEqual(result[0].price, 1.1150)

    def test_missing_eligible_swing_raises_error(self):
        swings = [
            make_classified_swing("LOW", 1.1100, "LL", 5),
        ]

        bos = [make_bos("BULLISH", 15)]

        with self.assertRaises(ValueError):
            self.tracker.track(swings, bos)

    def test_multiple_bos_events_preserve_protected_history(self):
        swings = [
            make_classified_swing("LOW", 1.1100, "HL", 5),
            make_classified_swing("LOW", 1.1150, "HL", 10),
            make_classified_swing("HIGH", 1.1300, "LH", 20),
        ]

        bos = [
            make_bos("BULLISH", 15),
            make_bos("BEARISH", 25),
        ]

        result = self.tracker.track(swings, bos)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].trend, "BULLISH")
        self.assertEqual(result[0].price, 1.1150)
        self.assertEqual(result[1].trend, "BEARISH")
        self.assertEqual(result[1].price, 1.1300)

    def test_bos_events_are_processed_chronologically(self):
        swings = [
            make_classified_swing("LOW", 1.1100, "HL", 5),
            make_classified_swing("LOW", 1.1150, "HL", 10),
        ]

        bos = [
            make_bos("BULLISH", 15),
            make_bos("BULLISH", 20),
        ]

        result = self.tracker.track(
            swings,
            list(reversed(bos)),
        )

        self.assertEqual(
            [event.established_by.break_index for event in result],
            [15, 20],
        )

    def test_non_bos_event_is_rejected(self):
        bos = make_bos("BULLISH", 15)

        invalid_event = StructureBreak(
            event_type="CHoCH",
            direction=bos.direction,
            break_time=bos.break_time,
            break_index=bos.break_index,
            level_price=bos.level_price,
            level_swing=bos.level_swing,
            close_price=bos.close_price,
            prior_trend=bos.prior_trend,
        )

        swings = [
            make_classified_swing("LOW", 1.1100, "HL", 5),
        ]

        with self.assertRaises(ValueError):
            self.tracker.track(swings, [invalid_event])


if __name__ == "__main__":
    unittest.main()
