
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.trend_models import TrendSnapshot
from research.swing_detection.bos_detector import BOSDetector


class TestBOSDetector(unittest.TestCase):

    def setUp(self):
        self.detector = BOSDetector()

        self.base_time = datetime(
            2026, 1, 1, 0, 0, tzinfo=timezone.utc
        )

    def get_duration(self, timeframe):
        return self.detector.TIMEFRAME_DURATIONS[timeframe]

    def make_candles(
        self,
        closes,
        custom_highs=None,
        timeframe="H1",
    ):
        rows = []
        duration = self.get_duration(timeframe)

        for index, close in enumerate(closes):
            high = (
                custom_highs[index]
                if custom_highs is not None
                else close + 0.002
            )

            rows.append({
                "time": self.base_time + duration * index,
                "open": close,
                "high": high,
                "low": close - 0.002,
                "close": close,
                "is_forming_candle": False,
            })

        return pd.DataFrame(rows)

    def make_swing(
        self,
        swing_type,
        price,
        confirmation_index=1,
        timeframe="H1",
    ):
        duration = self.get_duration(timeframe)
        swing_index = confirmation_index - 1

        swing = SwingPoint(
            swing_type=swing_type,
            price=price,
            swing_time=self.base_time + duration * swing_index,
            swing_index=swing_index,
            confirmation_time=(
                self.base_time + duration * confirmation_index
            ),
            confirmation_index=confirmation_index,
            left_bars=2,
            right_bars=2,
        )

        classification = (
            "HH" if swing_type == "HIGH" else "HL"
        )

        return ClassifiedSwing(
            swing=swing,
            classification=classification,
            previous_price=None,
            previous_swing_time=None,
        )

    def make_trend(
        self,
        trend,
        offset=2,
        timeframe="H1",
    ):
        duration = self.get_duration(timeframe)

        return TrendSnapshot(
            as_of_time=self.base_time + duration * offset,
            trend=trend,
            latest_high=None,
            latest_low=None,
            reason="Test fixture",
        )

    def test_bullish_bos_detected(self):
        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
            1.1010,
            1.1020,
        ])

        swing = self.make_swing("HIGH", 1.1000)
        trend = self.make_trend("BULLISH")

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "BOS")
        self.assertEqual(events[0].direction, "BULLISH")
        self.assertEqual(events[0].break_index, 3)
        self.assertEqual(events[0].level_price, 1.1000)

    def test_bearish_bos_detected(self):
        candles = self.make_candles([
            1.1050,
            1.1040,
            1.1010,
            1.0990,
            1.0980,
        ])

        swing = self.make_swing("LOW", 1.1000)
        trend = self.make_trend("BEARISH")

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "BOS")
        self.assertEqual(events[0].direction, "BEARISH")
        self.assertEqual(events[0].break_index, 3)
        self.assertEqual(events[0].level_price, 1.1000)

    def test_wick_only_break_does_not_trigger_bos(self):
        candles = self.make_candles(
            [
                1.0950,
                1.0960,
                1.0990,
                1.0995,
                1.0990,
            ],
            custom_highs=[
                1.0970,
                1.0980,
                1.1000,
                1.1050,
                1.1010,
            ],
        )

        swing = self.make_swing("HIGH", 1.1000)
        trend = self.make_trend("BULLISH")

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(events, [])

    def test_mixed_trend_does_not_trigger_bos(self):
        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
            1.1010,
            1.1020,
        ])

        swing = self.make_swing("HIGH", 1.1000)
        trend = self.make_trend("MIXED")

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(events, [])

    def test_unconfirmed_trend_does_not_trigger_bos(self):
        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
            1.1010,
            1.1020,
        ])

        swing = self.make_swing("HIGH", 1.1000)
        trend = self.make_trend("UNCONFIRMED")

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(events, [])

    def test_same_level_does_not_generate_duplicate_bos(self):
        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
            1.1010,
            1.0990,
            1.1020,
        ])

        swing = self.make_swing("HIGH", 1.1000)
        trend = self.make_trend("BULLISH")

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].break_index, 3)

    def test_same_candle_trend_snapshot_cannot_trigger_bos(self):
        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
            1.1010,
            1.1020,
        ])

        swing = self.make_swing("HIGH", 1.1000)

        trend = self.make_trend(
            "BULLISH",
            offset=3,
        )

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(events, [])

    def test_same_candle_swing_confirmation_cannot_trigger_bos(self):
        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
            1.1010,
            1.1020,
        ])

        swing = self.make_swing(
            "HIGH",
            1.1000,
            confirmation_index=3,
        )

        trend = self.make_trend("BULLISH")

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(events, [])

    def test_old_break_is_not_reported_after_trend_confirmation(self):
        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
            1.1010,
            1.1020,
        ])

        swing = self.make_swing("HIGH", 1.1000)

        trend = self.make_trend(
            "BULLISH",
            offset=3,
        )

        events = self.detector.detect(
            candles,
            [swing],
            [trend],
            timeframe="H1",
        )

        self.assertEqual(events, [])

    def test_all_timeframes_allow_trend_known_at_candle_open(self):

        for timeframe in ("M5", "M15", "H1", "H4"):

            with self.subTest(timeframe=timeframe):

                candles = self.make_candles(
                    [
                        1.0950,
                        1.0990,
                        1.1010,
                        1.1020,
                    ],
                    timeframe=timeframe,
                )

                swing = self.make_swing(
                    "HIGH",
                    1.1000,
                    confirmation_index=1,
                    timeframe=timeframe,
                )

                trend = self.make_trend(
                    "BULLISH",
                    offset=1,
                    timeframe=timeframe,
                )

                events = self.detector.detect(
                    candles,
                    [swing],
                    [trend],
                    timeframe=timeframe,
                )

                self.assertEqual(len(events), 1)
                self.assertEqual(events[0].break_index, 2)

    def test_all_timeframes_reject_trend_known_after_bos_candle_opens(self):

        for timeframe in ("M5", "M15", "H1", "H4"):

            with self.subTest(timeframe=timeframe):

                candles = self.make_candles(
                    [
                        1.0950,
                        1.0990,
                        1.1010,
                        1.1020,
                    ],
                    timeframe=timeframe,
                )

                swing = self.make_swing(
                    "HIGH",
                    1.1000,
                    confirmation_index=1,
                    timeframe=timeframe,
                )

                trend = self.make_trend(
                    "BULLISH",
                    offset=2,
                    timeframe=timeframe,
                )

                events = self.detector.detect(
                    candles,
                    [swing],
                    [trend],
                    timeframe=timeframe,
                )

                self.assertEqual(events, [])

    def test_unsupported_timeframe_is_rejected(self):

        candles = self.make_candles([
            1.0950,
            1.0960,
            1.0990,
        ])

        with self.assertRaises(ValueError):
            self.detector.detect(
                candles,
                [],
                [],
                timeframe="D1",
            )


if __name__ == "__main__":
    unittest.main()
