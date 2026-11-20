
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.break_models import StructureBreak
from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.registry_models import ProtectedLevel
from research.swing_detection.protected_level_registry import ProtectedLevelRegistry
from research.swing_detection.choch_detector import CHoCHDetector


BASE_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def make_swing(
    swing_type="LOW",
    classification="HL",
    price=1.10000,
    swing_index=5,
    confirmation_index=7,
):

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


def make_protected(
    direction="BULLISH",
    price=1.10000,
    break_index=2,
    swing_index=5,
    confirmation_index=7,
):

    if direction == "BULLISH":
        swing = make_swing(
            "LOW", "HL",
            price=price,
            swing_index=swing_index,
            confirmation_index=confirmation_index,
        )
    else:
        swing = make_swing(
            "HIGH", "LH",
            price=price,
            swing_index=swing_index,
            confirmation_index=confirmation_index,
        )

    event = make_bos(direction, break_index, swing)

    return ProtectedSwing(
        trend=direction,
        swing=swing,
        established_by=event,
        activation_index=break_index + 1,
    )


def make_candles(closes):

    return pd.DataFrame({
        "time": [
            BASE_TIME + timedelta(hours=i)
            for i in range(len(closes))
        ],
        "close": closes,
    })


class TestCHoCHDetector(unittest.TestCase):

    def setUp(self):
        self.detector = CHoCHDetector()
        self.registry = ProtectedLevelRegistry()

    def test_detects_bearish_choch_below_protected_hl(self):

        protected = make_protected(
            direction="BULLISH",
            price=1.10000,
            break_index=0,
        )

        self.registry.register([protected])

        candles = make_candles([
            1.10500,
            1.10400,
            1.09900,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "CHoCH")
        self.assertEqual(events[0].direction, "BEARISH")
        self.assertEqual(events[0].level_price, 1.10000)
        self.assertEqual(events[0].break_index, 2)
        self.assertEqual(events[0].prior_trend, "BULLISH")

    def test_detects_bullish_choch_above_protected_lh(self):

        protected = make_protected(
            direction="BEARISH",
            price=1.12000,
            break_index=0,
        )

        self.registry.register([protected])

        candles = make_candles([
            1.11500,
            1.11600,
            1.12100,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, "BULLISH")
        self.assertEqual(events[0].level_price, 1.12000)
        self.assertEqual(events[0].prior_trend, "BEARISH")

    def test_wick_only_breach_does_not_trigger(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([
            1.10500,
            1.10100,
        ])

        candles["low"] = [1.10400, 1.09500]

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(events, [])

    def test_level_not_active_before_activation_index(self):

        protected = make_protected(break_index=2)
        self.registry.register([protected])

        candles = make_candles([
            1.10500,
            1.10400,
            1.09900,
            1.09800,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(events, [])

    def test_detects_break_after_level_activation(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([
            1.10500,
            1.10400,
            1.09900,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(len(events), 1)

    def test_does_not_trigger_when_price_remains_beyond_level(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([
            1.09900,
            1.09800,
            1.09700,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(events, [])

    def test_retired_level_does_not_trigger(self):

        protected = make_protected(break_index=0)
        level = self.registry.register([protected])[0]

        retirement_event = StructureBreak(
            event_type="CHoCH",
            direction="BEARISH",
            break_time=BASE_TIME + timedelta(hours=1),
            break_index=1,
            level_price=protected.price,
            level_swing=protected.swing,
            close_price=1.09900,
            prior_trend="BULLISH",
        )

        self.registry.retire(level, retirement_event)

        candles = make_candles([
            1.10500,
            1.09900,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(events, [])

    def test_same_level_does_not_trigger_twice(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([
            1.10500,
            1.09900,
            1.10500,
            1.09800,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(len(events), 1)

    def test_multiple_active_levels_can_trigger(self):

        first = make_protected(
            direction="BULLISH",
            price=1.10000,
            break_index=0,
        )

        second = make_protected(
            direction="BULLISH",
            price=1.10200,
            break_index=0,
            swing_index=8,
            confirmation_index=10,
        )

        self.registry.register([first, second])

        candles = make_candles([
            1.10500,
            1.09900,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(len(events), 2)

    def test_empty_dataframe_is_rejected(self):

        with self.assertRaises(ValueError):
            self.detector.detect(
                pd.DataFrame(columns=["time", "close"]),
                self.registry,
            )

    def test_missing_column_is_rejected(self):

        candles = pd.DataFrame({
            "time": [BASE_TIME],
        })

        with self.assertRaises(ValueError):
            self.detector.detect(candles, self.registry)

    def test_naive_timestamp_is_rejected(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([1.10500, 1.09900])
        candles["time"] = candles["time"].astype(object)
        candles.at[0, "time"] = datetime(2026, 5, 1)

        with self.assertRaises(ValueError):
            self.detector.detect(candles, self.registry)

    def test_non_chronological_timestamps_are_rejected(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([1.10500, 1.09900])
        candles.loc[1, "time"] = candles.loc[0, "time"]

        with self.assertRaises(ValueError):
            self.detector.detect(candles, self.registry)

    def test_invalid_close_price_is_rejected(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([1.10500, float("nan")])

        with self.assertRaises(ValueError):
            self.detector.detect(candles, self.registry)

    def test_forming_candle_is_rejected(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([1.10500, 1.09900])
        candles["is_forming_candle"] = [False, True]

        with self.assertRaises(ValueError):
            self.detector.detect(candles, self.registry)

    def test_string_forming_flags_are_rejected(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([1.10500, 1.09900])
        candles["is_forming_candle"] = ["False", "False"]

        with self.assertRaises(TypeError):
            self.detector.detect(candles, self.registry)

    def test_numeric_forming_flags_are_rejected(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        candles = make_candles([1.10500, 1.09900])
        candles["is_forming_candle"] = [0, 0]

        with self.assertRaises(TypeError):
            self.detector.detect(candles, self.registry)

    def test_invalid_protected_level_price_is_rejected(self):

        protected = make_protected(
            price=float("nan"),
            break_index=0,
        )

        self.registry.register([protected])

        candles = make_candles([1.10500, 1.09900])

        with self.assertRaises(ValueError):
            self.detector.detect(candles, self.registry)

    def test_invalid_protected_level_activation_index_is_rejected(self):

        protected = make_protected(break_index=0)

        invalid_protected = replace(
            protected,
            activation_index=True,
        )

        invalid_level = ProtectedLevel(
            protected_swing=invalid_protected,
            bos_events=(protected.established_by,),
        )

        self.registry._levels[
            ("BULLISH", "LOW", 5, 7)
        ] = invalid_level

        candles = make_candles([1.10500, 1.09900])

        with self.assertRaises(ValueError):
            self.detector.detect(candles, self.registry)

    def test_malformed_protected_level_active_flag_is_rejected(self):

        protected = make_protected(break_index=0)
        level = self.registry.register([protected])[0]

        invalid_level = replace(
            level,
            active="True",
        )

        self.registry._levels[
            ("BULLISH", "LOW", 5, 7)
        ] = invalid_level

        candles = make_candles([1.10500, 1.09900])

        with self.assertRaises(TypeError):
            self.detector.detect(candles, self.registry)

    def test_detection_does_not_mutate_registry(self):

        protected = make_protected(break_index=0)
        self.registry.register([protected])

        original_levels = self.registry.levels

        candles = make_candles([
            1.10500,
            1.09900,
        ])

        events = self.detector.detect(candles, self.registry)

        self.assertEqual(len(events), 1)
        self.assertEqual(self.registry.levels, original_levels)
        self.assertTrue(self.registry.levels[0].active)


if __name__ == "__main__":
    unittest.main()
