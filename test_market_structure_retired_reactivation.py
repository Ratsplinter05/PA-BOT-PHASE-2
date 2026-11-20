import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.break_models import StructureBreak
from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
)
from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.swing_models import SwingPoint


BASE_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def candle_time(index):
    return BASE_TIME + timedelta(minutes=index)


def make_protected(
    swing_index,
    break_index,
    price,
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
        previous_price=None,
        previous_swing_time=None,
    )

    if trend == "BULLISH":
        bos_level = price + 0.01
        bos_close = price + 0.02
    else:
        bos_level = price - 0.01
        bos_close = price - 0.02

    bos = StructureBreak(
        event_type="BOS",
        direction=trend,
        break_time=candle_time(break_index),
        break_index=break_index,
        level_price=bos_level,
        level_swing=classified,
        close_price=bos_close,
        prior_trend=trend,
    )

    return ProtectedSwing(
        trend=trend,
        swing=classified,
        established_by=bos,
        activation_index=break_index + 1,
    )


def make_choch(protected, break_index):
    if protected.trend == "BULLISH":
        direction = "BEARISH"
        close_price = protected.price - 0.001
    else:
        direction = "BULLISH"
        close_price = protected.price + 0.001

    return StructureBreak(
        event_type="CHoCH",
        direction=direction,
        break_time=candle_time(break_index),
        break_index=break_index,
        level_price=protected.price,
        level_swing=protected.swing,
        close_price=close_price,
        prior_trend=protected.trend,
    )


class TestRetiredProtectedLevelReactivation(unittest.TestCase):

    def setUp(self):
        self.engine = MarketStructureEngine()

    def test_later_bos_for_retired_identity_is_excluded(self):
        first_bos = make_protected(
            swing_index=10,
            break_index=15,
            price=1.1000,
        )
        later_bos = make_protected(
            swing_index=10,
            break_index=25,
            price=1.1000,
        )
        retirement = make_choch(first_bos, break_index=20)

        result = self.engine._exclude_retired_reactivations(
            protected_swings=(first_bos, later_bos),
            choch_events=(retirement,),
        )

        self.assertEqual(result, (first_bos,))

    def test_repeated_bos_before_retirement_remains_eligible(self):
        first_bos = make_protected(
            swing_index=10,
            break_index=15,
            price=1.1000,
        )
        repeated_bos = make_protected(
            swing_index=10,
            break_index=18,
            price=1.1000,
        )
        retirement = make_choch(first_bos, break_index=20)

        result = self.engine._exclude_retired_reactivations(
            protected_swings=(first_bos, repeated_bos),
            choch_events=(retirement,),
        )

        self.assertEqual(result, (first_bos, repeated_bos))

    def test_same_candle_bos_is_registered_before_choch_retirement(self):
        first_bos = make_protected(
            swing_index=10,
            break_index=15,
            price=1.1000,
        )
        same_candle_bos = make_protected(
            swing_index=10,
            break_index=20,
            price=1.1000,
        )
        retirement = make_choch(first_bos, break_index=20)

        result = self.engine._exclude_retired_reactivations(
            protected_swings=(first_bos, same_candle_bos),
            choch_events=(retirement,),
        )

        self.assertEqual(result, (first_bos, same_candle_bos))

    def test_no_fallback_to_older_swing_after_latest_identity_retires(self):
        older_identity = make_protected(
            swing_index=5,
            break_index=12,
            price=1.0900,
        )
        latest_first_bos = make_protected(
            swing_index=10,
            break_index=15,
            price=1.1000,
        )
        latest_later_bos = make_protected(
            swing_index=10,
            break_index=25,
            price=1.1000,
        )
        retirement = make_choch(latest_first_bos, break_index=20)

        result = self.engine._exclude_retired_reactivations(
            protected_swings=(
                older_identity,
                latest_first_bos,
                latest_later_bos,
            ),
            choch_events=(retirement,),
        )

        self.assertEqual(
            result,
            (older_identity, latest_first_bos),
        )
        self.assertNotIn(latest_later_bos, result)

    def test_bearish_level_reactivation_is_also_excluded(self):
        first_bos = make_protected(
            swing_index=10,
            break_index=15,
            price=1.1200,
            trend="BEARISH",
        )
        later_bos = make_protected(
            swing_index=10,
            break_index=25,
            price=1.1200,
            trend="BEARISH",
        )
        retirement = make_choch(first_bos, break_index=20)

        result = self.engine._exclude_retired_reactivations(
            protected_swings=(first_bos, later_bos),
            choch_events=(retirement,),
        )

        self.assertEqual(result, (first_bos,))


if __name__ == "__main__":
    unittest.main()
