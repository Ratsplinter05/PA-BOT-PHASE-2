import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.bos_detector import BOSDetector
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.trend_models import TrendSnapshot

T0 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)

BULL_CLOSES = [1.1000, 1.1010, 1.1040, 1.1020, 1.1030, 1.1060, 1.1070]
BEAR_CLOSES = [1.1000, 1.0990, 1.0960, 1.0980, 1.0970, 1.0940, 1.0930]


def make_candles(closes, wicks=None):
    wicks = wicks or {}
    rows = []
    previous = closes[0]
    for i, close in enumerate(closes):
        open_price = previous
        high = max(open_price, close) + 0.0002
        low = min(open_price, close) - 0.0002
        if i in wicks:
            high = max(high, wicks[i])
        rows.append(
            {
                "time": T0 + i * HOUR,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
        )
        previous = close
    return pd.DataFrame(rows)


def make_item(swing_type, price, swing_index, confirmation_index,
              classification):
    swing = SwingPoint(
        swing_type=swing_type,
        price=price,
        swing_time=T0 + swing_index * HOUR,
        swing_index=swing_index,
        confirmation_time=T0 + confirmation_index * HOUR,
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


def make_snapshot(as_of_index, trend):
    return TrendSnapshot(
        as_of_time=T0 + as_of_index * HOUR,
        trend=trend,
        latest_high=None,
        latest_low=None,
        reason="test",
    )


class TestBOSDetectorExtra(unittest.TestCase):

    def setUp(self):
        self.detector = BOSDetector()

    def detect(self, closes, swings, snapshots, wicks=None,
               timeframe="H1"):
        return self.detector.detect(
            make_candles(closes, wicks), swings, snapshots, timeframe
        )

    def valid_inputs(self):
        return (
            make_candles(BULL_CLOSES),
            [make_item("HIGH", 1.1050, 2, 4, "HH")],
            [make_snapshot(4, "BULLISH")],
        )

    # ---- detection behaviour ----

    def test_bullish_bos_event_fields(self):
        high = make_item("HIGH", 1.1050, 2, 4, "HH")

        events = self.detect(BULL_CLOSES, [high], [make_snapshot(4, "BULLISH")])

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_type, "BOS")
        self.assertEqual(event.direction, "BULLISH")
        self.assertEqual(event.break_index, 5)
        self.assertEqual(event.break_time, T0 + 5 * HOUR)
        self.assertEqual(event.level_price, 1.1050)
        self.assertIs(event.level_swing, high)
        self.assertEqual(event.close_price, 1.1060)
        self.assertEqual(event.prior_trend, "BULLISH")

    def test_bearish_bos_event(self):
        low = make_item("LOW", 1.0950, 2, 4, "LL")

        events = self.detect(BEAR_CLOSES, [low], [make_snapshot(4, "BEARISH")])

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.direction, "BEARISH")
        self.assertEqual(event.break_index, 5)
        self.assertEqual(event.level_price, 1.0950)
        self.assertEqual(event.close_price, 1.0940)
        self.assertEqual(event.prior_trend, "BEARISH")

    def test_each_level_breaks_only_once(self):
        high = make_item("HIGH", 1.1050, 2, 4, "HH")
        closes = BULL_CLOSES[:6] + [1.1040, 1.1065]

        events = self.detect(closes, [high], [make_snapshot(4, "BULLISH")])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].break_index, 5)

    def test_wick_above_level_is_not_a_break(self):
        high = make_item("HIGH", 1.1050, 2, 4, "HH")
        closes = [1.1000, 1.1010, 1.1040, 1.1020, 1.1030, 1.1040, 1.1035]

        events = self.detect(
            closes, [high], [make_snapshot(4, "BULLISH")], wicks={5: 1.1080}
        )

        self.assertEqual(events, [])

    def test_only_matching_trend_produces_bos(self):
        high = make_item("HIGH", 1.1050, 2, 4, "HH")

        for trend in ("MIXED", "UNCONFIRMED", "BEARISH"):
            events = self.detect(BULL_CLOSES, [high], [make_snapshot(4, trend)])
            self.assertEqual(events, [], msg=trend)

    def test_bullish_trend_ignores_swing_lows(self):
        low = make_item("LOW", 1.0950, 2, 4, "HL")

        events = self.detect(BEAR_CLOSES, [low], [make_snapshot(4, "BULLISH")])

        self.assertEqual(events, [])

    def test_late_trend_does_not_create_retroactive_bos(self):
        # Trend becomes usable only after price already closed above the
        # level, so the break is not reported afterwards.
        high = make_item("HIGH", 1.1050, 2, 4, "HH")

        events = self.detect(BULL_CLOSES, [high], [make_snapshot(5, "BULLISH")])

        self.assertEqual(events, [])

    def test_timeframe_controls_when_trend_is_usable(self):
        # Records current behaviour: the timeframe argument is trusted.
        high = make_item("HIGH", 1.1050, 2, 4, "HH")
        closes = BULL_CLOSES + [1.1080, 1.1090]
        snapshots = [make_snapshot(4, "BULLISH")]

        h1_events = self.detect(closes, [high], snapshots, timeframe="H1")
        h4_events = self.detect(closes, [high], snapshots, timeframe="H4")

        self.assertEqual(len(h1_events), 1)
        self.assertEqual(h1_events[0].break_index, 5)
        self.assertEqual(h4_events, [])

    def test_newer_swing_replaces_older_level(self):
        first = make_item("HIGH", 1.1050, 2, 4, "HH")
        second = make_item("HIGH", 1.1100, 5, 7, "HH")
        closes = [
            1.1000, 1.1010, 1.1040, 1.1020, 1.1030,
            1.1045, 1.1035, 1.1040, 1.1070, 1.1110,
        ]
        snapshots = [make_snapshot(4, "BULLISH")]

        events = self.detect(closes, [first, second], snapshots)
        reversed_events = self.detect(closes, [second, first], snapshots)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].break_index, 9)
        self.assertEqual(events[0].level_price, 1.1100)
        self.assertIs(events[0].level_swing, second)
        self.assertEqual(events, reversed_events)

    def test_close_above_level_before_swing_is_usable_gives_no_break(self):
        high = make_item("HIGH", 1.1050, 2, 4, "HH")
        closes = [1.1000, 1.1010, 1.1040, 1.1020, 1.1060, 1.1070, 1.1080]

        events = self.detect(closes, [high], [make_snapshot(4, "BULLISH")])

        self.assertEqual(events, [])

    # ---- input validation ----

    def test_unsupported_timeframe_rejected(self):
        candles, swings, snapshots = self.valid_inputs()

        with self.assertRaisesRegex(ValueError, "Unsupported timeframe"):
            self.detector.detect(candles, swings, snapshots, "D1")

    def test_empty_candles_rejected(self):
        candles = pd.DataFrame(
            columns=["time", "open", "high", "low", "close"]
        )

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.detector.detect(candles, [], [], "H1")

    def test_missing_column_rejected(self):
        candles, swings, snapshots = self.valid_inputs()

        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.detector.detect(
                candles.drop(columns=["close"]), swings, snapshots, "H1"
            )

    def test_non_dataframe_rejected(self):
        with self.assertRaises(TypeError):
            self.detector.detect("nope", [], [], "H1")

    def test_naive_timestamps_rejected(self):
        candles, swings, snapshots = self.valid_inputs()
        candles["time"] = candles["time"].dt.tz_localize(None)

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.detector.detect(candles, swings, snapshots, "H1")

    def test_forming_candle_rejected(self):
        candles, swings, snapshots = self.valid_inputs()
        candles["is_forming_candle"] = (
            [False] * (len(candles) - 1) + [True]
        )

        with self.assertRaisesRegex(ValueError, "Forming candles"):
            self.detector.detect(candles, swings, snapshots, "H1")

    def test_non_chronological_timestamps_rejected(self):
        candles, swings, snapshots = self.valid_inputs()
        candles.loc[3, "time"] = candles.loc[2, "time"]

        with self.assertRaisesRegex(ValueError, "chronological"):
            self.detector.detect(candles, swings, snapshots, "H1")

    def test_invalid_ohlc_rejected(self):
        candles, swings, snapshots = self.valid_inputs()
        candles.loc[2, "high"] = 1.0

        with self.assertRaisesRegex(ValueError, "Candle high"):
            self.detector.detect(candles, swings, snapshots, "H1")

    def test_swing_timestamp_mismatch_rejected(self):
        candles, _, snapshots = self.valid_inputs()
        swing = SwingPoint(
            swing_type="HIGH",
            price=1.1050,
            swing_time=T0 + 3 * HOUR,
            swing_index=2,
            confirmation_time=T0 + 4 * HOUR,
            confirmation_index=4,
            left_bars=2,
            right_bars=2,
        )
        item = ClassifiedSwing(
            swing=swing,
            classification="HH",
            previous_price=None,
            previous_swing_time=None,
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            self.detector.detect(candles, [item], snapshots, "H1")

    def test_swing_index_out_of_range_rejected(self):
        candles, _, snapshots = self.valid_inputs()
        item = make_item("HIGH", 1.1050, 2, 50, "HH")

        with self.assertRaisesRegex(ValueError, "indices"):
            self.detector.detect(candles, [item], snapshots, "H1")

    def test_duplicate_swing_rejected(self):
        candles, swings, snapshots = self.valid_inputs()

        with self.assertRaisesRegex(ValueError, "Duplicate swing"):
            self.detector.detect(
                candles, [swings[0], swings[0]], snapshots, "H1"
            )

    def test_invalid_classification_for_swing_type_rejected(self):
        candles, _, snapshots = self.valid_inputs()
        item = make_item("HIGH", 1.1050, 2, 4, "HL")

        with self.assertRaisesRegex(ValueError, "Invalid classification"):
            self.detector.detect(candles, [item], snapshots, "H1")

    def test_non_classified_swing_rejected(self):
        candles, swings, snapshots = self.valid_inputs()

        with self.assertRaises(TypeError):
            self.detector.detect(candles, [swings[0].swing], snapshots, "H1")

    def test_duplicate_trend_timestamp_rejected(self):
        candles, swings, _ = self.valid_inputs()
        snapshots = [make_snapshot(4, "BULLISH"), make_snapshot(4, "MIXED")]

        with self.assertRaisesRegex(ValueError, "Duplicate trend snapshot"):
            self.detector.detect(candles, swings, snapshots, "H1")

    def test_invalid_trend_state_rejected(self):
        candles, swings, _ = self.valid_inputs()

        with self.assertRaisesRegex(ValueError, "Invalid trend state"):
            self.detector.detect(
                candles, swings, [make_snapshot(4, "SIDEWAYS")], "H1"
            )


if __name__ == "__main__":
    unittest.main()