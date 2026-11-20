
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.break_models import StructureBreak
from research.swing_detection.market_structure_engine import MarketStructureEngine
from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing


BASE_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def candle_time(index):
    return BASE_TIME + timedelta(hours=index)


def make_candles(closes):
    return pd.DataFrame({
        "time": [candle_time(i) for i in range(len(closes))],
        "open": closes,
        "high": [price + 1.0 for price in closes],
        "low": [price - 1.0 for price in closes],
        "close": closes,
    })


def make_protected_swing(
    price,
    swing_index,
    break_index,
    trend="BULLISH",
):
    swing_type = "LOW" if trend == "BULLISH" else "HIGH"
    classification = "HL" if trend == "BULLISH" else "LH"

    confirmation_index = swing_index + 1

    swing = SwingPoint(
        swing_type=swing_type,
        price=price,
        swing_time=candle_time(swing_index),
        swing_index=swing_index,
        confirmation_time=candle_time(confirmation_index),
        confirmation_index=confirmation_index,
        left_bars=1,
        right_bars=1,
    )

    classified = ClassifiedSwing(
        swing=swing,
        classification=classification,
        previous_price=price - 5.0 if trend == "BULLISH" else price + 5.0,
        previous_swing_time=candle_time(max(0, swing_index - 1)),
    )

    bos = StructureBreak(
        event_type="BOS",
        direction=trend,
        break_time=candle_time(break_index),
        break_index=break_index,
        level_price=price + 10.0 if trend == "BULLISH" else price - 10.0,
        level_swing=classified,
        close_price=price + 10.0 if trend == "BULLISH" else price - 10.0,
        prior_trend=trend,
    )

    return ProtectedSwing(
        trend=trend,
        swing=classified,
        established_by=bos,
        activation_index=break_index + 1,
    )


def make_choch_event(protected, break_index, close_price):
    trend = protected.trend

    direction = "BEARISH" if trend == "BULLISH" else "BULLISH"

    return StructureBreak(
        event_type="CHoCH",
        direction=direction,
        break_time=candle_time(break_index),
        break_index=break_index,
        level_price=protected.price,
        level_swing=protected.swing,
        close_price=close_price,
        prior_trend=trend,
    )


class TestMarketStructureLifecycle(unittest.TestCase):

    def setUp(self):
        self.engine = MarketStructureEngine()

    def test_two_protected_levels_are_broken_independently(self):
        candles = make_candles([
            110, 110, 110, 99, 94
        ])

        level_100 = make_protected_swing(
            price=100,
            swing_index=0,
            break_index=2,
        )

        level_95 = make_protected_swing(
            price=95,
            swing_index=1,
            break_index=2,
        )

        events = self.engine._detect_choch_events(
            candles,
            (level_100, level_95),
        )

        self.assertEqual(len(events), 2)

        self.assertEqual(
            [(event.break_index, event.level_price) for event in events],
            [(3, 100.0), (4, 95.0)],
        )

        registry = self.engine._replay_registry(
            (level_100, level_95),
            events,
        )

        self.assertEqual(len(registry.get_retired_levels()), 2)
        self.assertEqual(len(registry.get_active_levels()), 0)

    def test_only_broken_level_is_retired(self):
        candles = make_candles([
            110, 110, 110, 99, 98
        ])

        level_100 = make_protected_swing(
            price=100,
            swing_index=0,
            break_index=2,
        )

        level_90 = make_protected_swing(
            price=90,
            swing_index=1,
            break_index=2,
        )

        events = self.engine._detect_choch_events(
            candles,
            (level_100, level_90),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].level_price, 100.0)

        registry = self.engine._replay_registry(
            (level_100, level_90),
            events,
        )

        active = registry.get_active_levels()
        retired = registry.get_retired_levels()

        self.assertEqual(len(active), 1)
        self.assertEqual(len(retired), 1)

        self.assertEqual(active[0].price, 90.0)
        self.assertEqual(retired[0].price, 100.0)

    def test_two_levels_broken_on_same_candle(self):
        candles = make_candles([
            110, 110, 110, 90
        ])

        level_100 = make_protected_swing(
            price=100,
            swing_index=0,
            break_index=2,
        )

        level_95 = make_protected_swing(
            price=95,
            swing_index=1,
            break_index=2,
        )

        events = self.engine._detect_choch_events(
            candles,
            (level_100, level_95),
        )

        self.assertEqual(len(events), 2)

        self.assertEqual(
            {event.level_price for event in events},
            {100.0, 95.0},
        )

        self.assertEqual(
            {event.break_index for event in events},
            {3},
        )

        registry = self.engine._replay_registry(
            (level_100, level_95),
            events,
        )

        self.assertEqual(len(registry.get_retired_levels()), 2)
        self.assertEqual(len(registry.get_active_levels()), 0)

    def test_one_level_generates_only_one_event_after_repeated_crossings(self):
        candles = make_candles([
            110, 110, 110, 99, 105, 95
        ])

        protected = make_protected_swing(
            price=100,
            swing_index=0,
            break_index=2,
        )

        events = self.engine._detect_choch_events(
            candles,
            (protected,),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].break_index, 3)
        self.assertEqual(events[0].level_price, 100.0)

        registry = self.engine._replay_registry(
            (protected,),
            events,
        )

        self.assertEqual(len(registry.get_retired_levels()), 1)
        self.assertEqual(len(registry.get_active_levels()), 0)

    def test_late_activating_level_cannot_trigger_before_activation(self):
        candles = make_candles([
            110, 110, 90, 105, 95
        ])

        protected = make_protected_swing(
            price=100,
            swing_index=1,
            break_index=2,
        )

        self.assertEqual(protected.activation_index, 3)

        events = self.engine._detect_choch_events(
            candles,
            (protected,),
        )

        self.assertEqual(len(events), 1)

        # The earlier downward crossing at index 2 must be ignored.
        # The valid crossing after activation occurs at index 4.
        self.assertEqual(events[0].break_index, 4)

        registry = self.engine._replay_registry(
            (protected,),
            events,
        )

        self.assertEqual(len(registry.get_retired_levels()), 1)

    def test_replay_registers_levels_and_retires_only_matching_identity(self):
        level_100 = make_protected_swing(
            price=100,
            swing_index=0,
            break_index=2,
        )

        level_95 = make_protected_swing(
            price=95,
            swing_index=1,
            break_index=2,
        )

        event = make_choch_event(
            protected=level_100,
            break_index=4,
            close_price=99,
        )

        registry = self.engine._replay_registry(
            (level_100, level_95),
            (event,),
        )

        active = registry.get_active_levels()
        retired = registry.get_retired_levels()

        self.assertEqual(len(active), 1)
        self.assertEqual(len(retired), 1)

        self.assertEqual(active[0].price, 95.0)
        self.assertEqual(retired[0].price, 100.0)

        self.assertEqual(
            retired[0].retired_by,
            event,
        )


if __name__ == "__main__":
    unittest.main()
