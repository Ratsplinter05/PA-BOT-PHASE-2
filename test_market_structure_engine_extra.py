import math
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
)

T0 = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)


def wave_candles(bars=240, flip=120, step_hours=1):
    """Rising waves for `flip` bars, then falling waves.

    Candles are dojis (open == close) with a fixed 0.0002 wick, so each
    swing price is the close plus or minus 0.0002.
    """
    closes = []
    for i in range(bars):
        if i < flip:
            phase = i
            level = 0.0004 * i
        else:
            phase = i - flip
            level = 0.0004 * flip - 0.0004 * phase
        closes.append(1.1000 + level + 0.003 * math.sin(math.pi * phase / 6))

    return pd.DataFrame(
        {
            "time": pd.date_range(
                T0, periods=bars, freq=pd.Timedelta(hours=step_hours)
            ),
            "open": closes,
            "high": [c + 0.0002 for c in closes],
            "low": [c - 0.0002 for c in closes],
            "close": closes,
        }
    )


class TestMarketStructureEngineExtra(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.candles = wave_candles()
        cls.engine = MarketStructureEngine()
        cls.full = cls.engine.analyze(cls.candles, "H1")

    def test_hand_traced_scenario(self):
        full = self.full

        self.assertEqual(
            [(s.swing_type, s.swing_index) for s in full.swings[:4]],
            [("HIGH", 3), ("LOW", 9), ("HIGH", 15), ("LOW", 21)],
        )
        self.assertAlmostEqual(full.swings[0].price, 1.1044, places=6)
        self.assertEqual(
            [c.classification for c in full.classified_swings[:4]],
            [None, None, "HH", "HL"],
        )
        self.assertEqual(
            [t.trend for t in full.trend_snapshots[:4]],
            ["UNCONFIRMED", "UNCONFIRMED", "UNCONFIRMED", "BULLISH"],
        )

        first_bos = full.bos_events[0]
        self.assertEqual(first_bos.direction, "BULLISH")
        self.assertEqual(first_bos.break_index, 24)
        self.assertAlmostEqual(first_bos.level_price, 1.1092, places=6)

        first_protected = full.protected_swings[0]
        self.assertEqual(first_protected.trend, "BULLISH")
        self.assertEqual(first_protected.activation_index, 25)
        self.assertAlmostEqual(first_protected.price, 1.1052, places=6)

        first_choch = full.choch_events[0]
        self.assertEqual(first_choch.direction, "BEARISH")
        self.assertEqual(first_choch.break_index, 128)
        self.assertAlmostEqual(first_choch.level_price, 1.1436, places=6)

    def test_registry_invariants(self):
        full = self.full
        retired = full.retired_protected_levels

        self.assertGreater(len(full.choch_events), 0)
        self.assertEqual(len(retired), len(full.choch_events))
        self.assertEqual(
            len(full.active_protected_levels) + len(retired),
            len(full.protected_levels),
        )
        self.assertCountEqual(
            [level.retired_by for level in retired],
            list(full.choch_events),
        )
        for level in retired:
            self.assertGreaterEqual(
                level.retired_by.break_index, level.activation_index
            )

    def test_no_look_ahead_prefix_stability(self):
        # Analysing the first n candles must give exactly the results the
        # full analysis reports up to candle n. Any difference means some
        # stage used information from the future.
        candles = self.candles
        full = self.full

        for n in list(range(10, len(candles), 7)) + [len(candles)]:
            prefix = self.engine.analyze(candles.iloc[:n], "H1")
            last_time = candles["time"].iloc[n - 1].to_pydatetime()
            label = f"prefix length {n}"

            self.assertEqual(
                prefix.swings,
                tuple(s for s in full.swings if s.confirmation_index < n),
                msg=label,
            )
            self.assertEqual(
                prefix.classified_swings,
                tuple(
                    c
                    for c in full.classified_swings
                    if c.swing.confirmation_index < n
                ),
                msg=label,
            )
            self.assertEqual(
                prefix.trend_snapshots,
                tuple(
                    t for t in full.trend_snapshots if t.as_of_time <= last_time
                ),
                msg=label,
            )
            self.assertEqual(
                prefix.bos_events,
                tuple(e for e in full.bos_events if e.break_index < n),
                msg=label,
            )
            self.assertEqual(
                prefix.protected_swings,
                tuple(
                    p
                    for p in full.protected_swings
                    if p.established_by.break_index < n
                ),
                msg=label,
            )
            self.assertEqual(
                prefix.choch_events,
                tuple(e for e in full.choch_events if e.break_index < n),
                msg=label,
            )

    def test_analysis_is_deterministic_and_does_not_modify_input(self):
        before = self.candles.copy()

        again = self.engine.analyze(self.candles, "H1")

        self.assertEqual(again, self.full)
        pd.testing.assert_frame_equal(self.candles, before)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.analyze("nope")

        with self.assertRaises(ValueError):
            self.engine.analyze(self.candles.iloc[:0])

        with self.assertRaisesRegex(ValueError, "At least two"):
            self.engine.analyze(self.candles.iloc[:1])

        with self.assertRaisesRegex(ValueError, "Unsupported timeframe"):
            self.engine.analyze(self.candles, "D1")

        with self.assertRaises(TypeError):
            self.engine.analyze(self.candles, 60)

    def test_multi_timeframe_only_uses_closed_snapshots(self):
        h1 = self.candles
        h4 = wave_candles(step_hours=4)

        for hours in (0, 30, 100, 200, 400, 900):
            when = T0 + timedelta(hours=hours)
            alignment = self.engine.analyze_multi_timeframe(
                h4, h1, when
            ).alignment

            if alignment.h4_snapshot is not None:
                self.assertLessEqual(alignment.h4_known_at_time, when)
            if alignment.h1_snapshot is not None:
                self.assertLessEqual(alignment.h1_known_at_time, when)
            self.assertEqual(
                alignment.directional_permission,
                alignment.alignment_state.startswith("ALIGNED"),
            )

        at_start = self.engine.analyze_multi_timeframe(h4, h1, T0)
        self.assertEqual(at_start.alignment.alignment_state, "H4_MISSING")


if __name__ == "__main__":
    unittest.main()