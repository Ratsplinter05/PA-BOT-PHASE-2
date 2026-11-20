
import unittest
from datetime import datetime, timedelta, timezone
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
    MarketStructureResult,
)
from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.trend_models import TrendSnapshot
from research.swing_detection.break_models import StructureBreak
from research.swing_detection.protected_models import ProtectedSwing


BASE_TIME = datetime(
    2026, 5, 1, tzinfo=timezone.utc
)


class TestMarketStructureEngine(unittest.TestCase):

    def setUp(self):
        self.engine = MarketStructureEngine()

    @staticmethod
    def make_candles(closes):

        return pd.DataFrame({
            "time": [
                BASE_TIME + timedelta(hours=index)
                for index in range(len(closes))
            ],
            "open": closes,
            "high": [
                price + 0.002
                for price in closes
            ],
            "low": [
                price - 0.002
                for price in closes
            ],
            "close": closes,
        })

    @staticmethod
    def make_swing():

        swing = SwingPoint(
            swing_type="LOW",
            price=1.10000,
            swing_time=BASE_TIME,
            swing_index=0,
            confirmation_time=BASE_TIME + timedelta(hours=1),
            confirmation_index=1,
            left_bars=1,
            right_bars=1,
        )

        return ClassifiedSwing(
            swing=swing,
            classification="HL",
            previous_price=None,
            previous_swing_time=None,
        )

    @staticmethod
    def make_bos(classified_swing):

        return StructureBreak(
            event_type="BOS",
            direction="BULLISH",
            break_time=BASE_TIME + timedelta(hours=2),
            break_index=2,
            level_price=1.11000,
            level_swing=classified_swing,
            close_price=1.11100,
            prior_trend="BULLISH",
        )

    @classmethod
    def make_protected(cls):

        classified_swing = cls.make_swing()
        bos = cls.make_bos(classified_swing)

        return ProtectedSwing(
            trend="BULLISH",
            swing=classified_swing,
            established_by=bos,
            activation_index=3,
        )

    def run_controlled_analysis(self, candles, protected):

        classified_swing = protected.swing
        swing = classified_swing.swing

        trend = TrendSnapshot(
            as_of_time=BASE_TIME + timedelta(hours=1),
            trend="BULLISH",
            latest_high=None,
            latest_low=classified_swing,
            reason="Controlled integration test.",
        )

        bos = protected.established_by

        with ExitStack() as stack:

            stack.enter_context(
                patch.object(
                    self.engine.fractal_detector,
                    "detect",
                    return_value=[swing],
                )
            )

            stack.enter_context(
                patch.object(
                    self.engine.structure_classifier,
                    "classify",
                    return_value=[classified_swing],
                )
            )

            stack.enter_context(
                patch.object(
                    self.engine.trend_classifier,
                    "classify",
                    return_value=[trend],
                )
            )

            stack.enter_context(
                patch.object(
                    self.engine.bos_detector,
                    "detect",
                    return_value=[bos],
                )
            )

            stack.enter_context(
                patch.object(
                    self.engine.protected_swing_tracker,
                    "track",
                    return_value=[protected],
                )
            )

            return self.engine.analyze(
                candles=candles,
                timeframe="H1",
            )

    def test_complete_analysis_detects_and_retires_protected_level(self):

        protected = self.make_protected()

        candles = self.make_candles([
            1.10500,
            1.10400,
            1.10300,
            1.09900,
        ])

        result = self.run_controlled_analysis(
            candles,
            protected,
        )

        self.assertEqual(result.timeframe, "H1")
        self.assertEqual(result.candle_count, 4)

        self.assertEqual(len(result.swings), 1)
        self.assertEqual(len(result.classified_swings), 1)
        self.assertEqual(len(result.trend_snapshots), 1)

        self.assertEqual(len(result.bos_events), 1)
        self.assertEqual(len(result.protected_swings), 1)

        self.assertEqual(len(result.choch_events), 1)
        self.assertEqual(
            result.choch_events[0].direction,
            "BEARISH",
        )

        self.assertEqual(len(result.protected_levels), 1)
        self.assertEqual(len(result.active_protected_levels), 0)
        self.assertEqual(len(result.retired_protected_levels), 1)

        self.assertFalse(
            result.retired_protected_levels[0].active
        )

    def test_level_cannot_trigger_before_activation(self):

        protected = self.make_protected()

        # The price breaks the protected level on the BOS candle.
        # The protected level only becomes active on candle index 3.
        # It must not trigger on the following candle because price
        # is already below the level.
        candles = self.make_candles([
            1.10500,
            1.10400,
            1.09900,
            1.09800,
        ])

        result = self.run_controlled_analysis(
            candles,
            protected,
        )

        self.assertEqual(result.choch_events, ())
        self.assertEqual(len(result.active_protected_levels), 1)
        self.assertEqual(result.retired_protected_levels, ())

    def test_level_remains_active_when_no_choch_occurs(self):

        protected = self.make_protected()

        candles = self.make_candles([
            1.10500,
            1.10400,
            1.10300,
            1.10200,
        ])

        result = self.run_controlled_analysis(
            candles,
            protected,
        )

        self.assertEqual(result.choch_events, ())
        self.assertEqual(len(result.active_protected_levels), 1)
        self.assertEqual(result.retired_protected_levels, ())

    def test_empty_candles_are_rejected(self):

        candles = pd.DataFrame(
            columns=[
                "time",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        with self.assertRaises(ValueError):
            self.engine.analyze(candles)

    def test_single_candle_is_rejected(self):

        candles = self.make_candles([1.10500])

        with self.assertRaises(ValueError):
            self.engine.analyze(candles)

    def test_invalid_timeframe_is_rejected(self):

        candles = self.make_candles([
            1.10500,
            1.10400,
        ])

        with self.assertRaises(ValueError):
            self.engine.analyze(
                candles,
                timeframe="M30",
            )

    def test_non_dataframe_input_is_rejected(self):

        with self.assertRaises(TypeError):
            self.engine.analyze([1.10500, 1.10400])

    def test_multi_timeframe_result_model(self):

        h4_result = MarketStructureResult(
            timeframe="H4",
            candle_count=10,
            swings=(),
            classified_swings=(),
            trend_snapshots=(),
            bos_events=(),
            protected_swings=(),
            choch_events=(),
            protected_levels=(),
            active_protected_levels=(),
            retired_protected_levels=(),
        )

        h1_result = MarketStructureResult(
            timeframe="H1",
            candle_count=10,
            swings=(),
            classified_swings=(),
            trend_snapshots=(),
            bos_events=(),
            protected_swings=(),
            choch_events=(),
            protected_levels=(),
            active_protected_levels=(),
            retired_protected_levels=(),
        )

        self.assertEqual(h4_result.timeframe, "H4")
        self.assertEqual(h1_result.timeframe, "H1")


if __name__ == "__main__":
    unittest.main()
