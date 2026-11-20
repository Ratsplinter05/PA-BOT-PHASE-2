import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.break_models import StructureBreak
from research.swing_detection.choch_detector import CHoCHDetector
from research.swing_detection.protected_level_registry import (
    ProtectedLevelRegistry,
)
from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.swing_models import SwingPoint

BASE_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def make_swing(swing_type, classification, price, swing_index,
               confirmation_index):
    swing = SwingPoint(
        swing_type=swing_type,
        price=price,
        swing_time=BASE_TIME + timedelta(hours=swing_index),
        swing_index=swing_index,
        confirmation_time=BASE_TIME + timedelta(hours=confirmation_index),
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


def make_bos(direction, break_index, swing):
    return StructureBreak(
        event_type="BOS",
        direction=direction,
        break_time=BASE_TIME + timedelta(hours=break_index),
        break_index=break_index,
        level_price=1.12000,
        level_swing=swing,
        close_price=1.12100,
        prior_trend=direction,
    )


def make_protected(direction="BULLISH", price=1.10000, break_index=0,
                   swing_index=5, confirmation_index=7):
    if direction == "BULLISH":
        swing = make_swing(
            "LOW", "HL", price, swing_index, confirmation_index
        )
    else:
        swing = make_swing(
            "HIGH", "LH", price, swing_index, confirmation_index
        )
    return ProtectedSwing(
        trend=direction,
        swing=swing,
        established_by=make_bos(direction, break_index, swing),
        activation_index=break_index + 1,
    )


def make_candles(closes):
    return pd.DataFrame(
        {
            "time": [
                BASE_TIME + timedelta(hours=i) for i in range(len(closes))
            ],
            "close": closes,
        }
    )


class TestCHoCHDetectorExtra(unittest.TestCase):

    def setUp(self):
        self.detector = CHoCHDetector()
        self.registry = ProtectedLevelRegistry()

    # ---- boundary behaviour ----

    def test_bullish_level_previous_close_on_level_still_triggers(self):
        self.registry.register([make_protected("BULLISH", 1.10000)])

        events = self.detector.detect(
            make_candles([1.10000, 1.09900]), self.registry
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, "BEARISH")
        self.assertEqual(events[0].close_price, 1.09900)

    def test_bullish_level_close_on_level_does_not_trigger(self):
        self.registry.register([make_protected("BULLISH", 1.10000)])

        events = self.detector.detect(
            make_candles([1.10500, 1.10000]), self.registry
        )

        self.assertEqual(events, [])

    def test_bearish_level_boundaries(self):
        self.registry.register([make_protected("BEARISH", 1.12000)])

        touches = self.detector.detect(
            make_candles([1.12000, 1.12100]), self.registry
        )
        equal_close = self.detector.detect(
            make_candles([1.11500, 1.12000]), self.registry
        )

        self.assertEqual(len(touches), 1)
        self.assertEqual(touches[0].direction, "BULLISH")
        self.assertEqual(equal_close, [])

    def test_can_trigger_on_the_activation_candle(self):
        # BOS on candle 2, so the level activates on candle 3.
        self.registry.register([make_protected(break_index=2)])

        events = self.detector.detect(
            make_candles([1.10500, 1.10500, 1.10500, 1.09900]),
            self.registry,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].break_index, 3)

    # ---- start_index ----

    def test_start_index_is_respected(self):
        self.registry.register([make_protected()])
        candles = make_candles([1.10500, 1.09900, 1.09800])

        from_one = self.detector.detect(candles, self.registry, start_index=1)
        from_two = self.detector.detect(candles, self.registry, start_index=2)

        self.assertEqual(len(from_one), 1)
        self.assertEqual(from_one[0].break_index, 1)
        self.assertEqual(
            from_one[0].break_time, BASE_TIME + timedelta(hours=1)
        )
        # Skipping the crossing candle means the break is never reported.
        self.assertEqual(from_two, [])

    def test_invalid_start_index_rejected(self):
        candles = make_candles([1.10500, 1.09900, 1.09800])

        for bad in (0, -1, 3, True, 1.5, "1"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.detector.detect(candles, self.registry, start_index=bad)

    def test_single_candle_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.detect(make_candles([1.10500]), self.registry)

    # ---- detect / retire handoff ----

    def test_repeated_calls_repeat_events_until_level_is_retired(self):
        # Records current behaviour: detect() never retires levels, so the
        # caller must call registry.retire() after each CHoCH.
        self.registry.register([make_protected()])
        candles = make_candles([1.10500, 1.09900, 1.09800])

        first = self.detector.detect(candles, self.registry)
        second = self.detector.detect(candles, self.registry)

        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)

        self.registry.retire(self.registry.levels[0], first[0])

        self.assertEqual(self.detector.detect(candles, self.registry), [])
        self.assertFalse(self.registry.levels[0].active)
        self.assertIs(self.registry.levels[0].retired_by, first[0])

    def test_events_follow_registry_order_not_price_order(self):
        low_price = make_protected("BULLISH", 1.10000)
        high_price = make_protected(
            "BULLISH", 1.10200, swing_index=8, confirmation_index=10
        )
        self.registry.register([low_price, high_price])

        events = self.detector.detect(
            make_candles([1.10500, 1.09900]), self.registry
        )

        self.assertEqual([e.level_price for e in events], [1.10000, 1.10200])

    def test_candles_are_not_modified(self):
        self.registry.register([make_protected()])
        candles = make_candles([1.10500, 1.09900])
        before = candles.copy()

        self.detector.detect(candles, self.registry)

        pd.testing.assert_frame_equal(candles, before)

    # ---- validation ----

    def test_wrong_argument_types_rejected(self):
        with self.assertRaises(TypeError):
            self.detector.detect("nope", self.registry)

        with self.assertRaises(TypeError):
            self.detector.detect(make_candles([1.1, 1.0]), "nope")

    def test_bullish_level_on_a_swing_high_rejected(self):
        # The registry accepts this record; the detector rejects it.
        swing = make_swing("HIGH", "LH", 1.10000, 5, 7)
        protected = ProtectedSwing(
            trend="BULLISH",
            swing=swing,
            established_by=make_bos("BULLISH", 0, swing),
            activation_index=1,
        )
        self.registry.register([protected])

        with self.assertRaisesRegex(ValueError, "swing low"):
            self.detector.detect(
                make_candles([1.10500, 1.09900]), self.registry
            )

    def test_duplicate_registry_levels_rejected(self):
        level = self.registry.register([make_protected()])[0]
        self.registry._levels["copy"] = level

        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.detector.detect(
                make_candles([1.10500, 1.09900]), self.registry
            )

    def test_non_level_registry_record_rejected(self):
        self.registry._levels["bad"] = "not a level"

        with self.assertRaises(TypeError):
            self.detector.detect(
                make_candles([1.10500, 1.09900]), self.registry
            )


if __name__ == "__main__":
    unittest.main()