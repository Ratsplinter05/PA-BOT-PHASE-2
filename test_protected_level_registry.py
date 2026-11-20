
import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.break_models import StructureBreak
from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.protected_level_registry import ProtectedLevelRegistry


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


def make_event(
    event_type="BOS",
    direction="BULLISH",
    break_index=10,
    level_swing=None,
    level_price=1.12000,
    close_price=1.12100,
    prior_trend=None,
):

    if level_swing is None:
        level_swing = make_swing()

    if prior_trend is None:
        if event_type == "BOS":
            prior_trend = direction
        else:
            prior_trend = (
                "BULLISH" if direction == "BEARISH"
                else "BEARISH"
            )

    return StructureBreak(
        event_type=event_type,
        direction=direction,
        break_time=BASE_TIME + timedelta(hours=break_index),
        break_index=break_index,
        level_price=level_price,
        level_swing=level_swing,
        close_price=close_price,
        prior_trend=prior_trend,
    )


def make_protected(
    swing=None,
    direction="BULLISH",
    break_index=10,
):

    if swing is None:
        if direction == "BULLISH":
            swing = make_swing("LOW", "HL")
        else:
            swing = make_swing(
                "HIGH",
                "LH",
                price=1.12000,
            )

    event = make_event(
        event_type="BOS",
        direction=direction,
        break_index=break_index,
    )

    return ProtectedSwing(
        trend=direction,
        swing=swing,
        established_by=event,
        activation_index=break_index + 1,
    )


def make_choch(
    protected,
    direction,
    break_index=15,
    level_swing=None,
    level_price=None,
    close_price=None,
    prior_trend=None,
):

    if level_swing is None:
        level_swing = protected.swing

    if level_price is None:
        level_price = protected.price

    if close_price is None:
        if direction == "BEARISH":
            close_price = level_price - 0.00100
        else:
            close_price = level_price + 0.00100

    return make_event(
        event_type="CHoCH",
        direction=direction,
        break_index=break_index,
        level_swing=level_swing,
        level_price=level_price,
        close_price=close_price,
        prior_trend=prior_trend,
    )


class TestProtectedLevelRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ProtectedLevelRegistry()

    def test_registers_new_protected_level(self):

        protected = make_protected()

        result = self.registry.register([protected])

        self.assertEqual(len(result), 1)
        self.assertEqual(len(self.registry.levels), 1)
        self.assertEqual(result[0].price, 1.10000)
        self.assertTrue(result[0].active)

    def test_repeated_bos_uses_existing_level(self):

        swing = make_swing()

        first = make_protected(
            swing=swing,
            break_index=10,
        )

        second = make_protected(
            swing=swing,
            break_index=15,
        )

        self.registry.register([first])
        result = self.registry.register([second])

        self.assertEqual(len(self.registry.levels), 1)
        self.assertEqual(len(result[0].bos_events), 2)

    def test_repeated_bos_preserves_original_activation(self):

        swing = make_swing()

        first = make_protected(
            swing=swing,
            break_index=10,
        )

        second = make_protected(
            swing=swing,
            break_index=15,
        )

        self.registry.register([first])
        updated = self.registry.register([second])[0]

        self.assertEqual(updated.activation_index, 11)
        self.assertEqual(updated.established_by.break_index, 10)

    def test_unique_swing_creates_separate_level(self):

        first = make_protected(
            swing=make_swing(swing_index=5, confirmation_index=7),
            break_index=10,
        )

        second = make_protected(
            swing=make_swing(
                swing_index=12,
                confirmation_index=14,
                price=1.10500,
            ),
            break_index=18,
        )

        self.registry.register([first, second])

        self.assertEqual(len(self.registry.levels), 2)

    def test_level_is_not_active_before_activation_index(self):

        protected = make_protected(break_index=10)
        self.registry.register([protected])

        self.assertEqual(
            len(self.registry.get_active_levels(as_of_index=10)),
            0,
        )

        self.assertEqual(
            len(self.registry.get_active_levels(as_of_index=11)),
            1,
        )

    def test_bearish_choch_retires_bullish_level(self):

        protected = make_protected(direction="BULLISH")
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BEARISH",
        )

        retired = self.registry.retire(level, choch)

        self.assertFalse(retired.active)
        self.assertEqual(retired.retired_by, choch)
        self.assertEqual(len(self.registry.get_active_levels()), 0)
        self.assertEqual(len(self.registry.get_retired_levels()), 1)

    def test_bullish_choch_retires_bearish_level(self):

        protected = make_protected(direction="BEARISH")
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BULLISH",
        )

        retired = self.registry.retire(level, choch)

        self.assertFalse(retired.active)

    def test_same_direction_choch_is_rejected(self):

        protected = make_protected(direction="BULLISH")
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BULLISH",
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, choch)

    def test_bos_cannot_retire_protected_level(self):

        protected = make_protected()
        level = self.registry.register([protected])[0]

        bos = make_event(
            event_type="BOS",
            direction="BEARISH",
            break_index=15,
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, bos)

    def test_choch_before_activation_is_rejected(self):

        protected = make_protected(break_index=10)
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BEARISH",
            break_index=10,
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, choch)

    def test_retired_level_cannot_be_registered_again(self):

        protected = make_protected()
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BEARISH",
        )

        self.registry.retire(level, choch)

        repeated = make_protected(
            swing=protected.swing,
            direction="BULLISH",
            break_index=18,
        )

        with self.assertRaises(ValueError):
            self.registry.register([repeated])

    def test_retired_level_cannot_be_retired_twice(self):

        protected = make_protected()
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BEARISH",
        )

        retired = self.registry.retire(level, choch)

        with self.assertRaises(ValueError):
            self.registry.retire(retired, choch)

    def test_invalid_protected_item_is_rejected(self):

        with self.assertRaises(TypeError):
            self.registry.register(["not a protected swing"])

    def test_invalid_activation_index_is_rejected(self):

        protected = make_protected()

        invalid = ProtectedSwing(
            trend=protected.trend,
            swing=protected.swing,
            established_by=protected.established_by,
            activation_index=protected.activation_index + 1,
        )

        with self.assertRaises(ValueError):
            self.registry.register([invalid])

    def test_unrelated_swing_cannot_retire_protected_level(self):

        protected = make_protected()
        level = self.registry.register([protected])[0]

        unrelated_swing = make_swing(
            swing_type="LOW",
            classification="HL",
            price=1.09500,
            swing_index=12,
            confirmation_index=14,
        )

        choch = make_choch(
            protected,
            direction="BEARISH",
            level_swing=unrelated_swing,
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, choch)

        self.assertTrue(self.registry.levels[0].active)

    def test_incorrect_choch_level_price_is_rejected(self):

        protected = make_protected()
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BEARISH",
            level_price=1.09500,
            close_price=1.09400,
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, choch)

    def test_incorrect_prior_trend_is_rejected(self):

        protected = make_protected(direction="BULLISH")
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BEARISH",
            prior_trend="BEARISH",
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, choch)

    def test_close_must_confirm_break_below_bullish_level(self):

        protected = make_protected(direction="BULLISH")
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BEARISH",
            close_price=protected.price + 0.00100,
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, choch)

    def test_close_must_confirm_break_above_bearish_level(self):

        protected = make_protected(direction="BEARISH")
        level = self.registry.register([protected])[0]

        choch = make_choch(
            protected,
            direction="BULLISH",
            close_price=protected.price - 0.00100,
        )

        with self.assertRaises(ValueError):
            self.registry.retire(level, choch)


if __name__ == "__main__":
    unittest.main()
