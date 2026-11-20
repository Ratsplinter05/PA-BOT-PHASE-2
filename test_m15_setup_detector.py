
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from research.swing_detection.market_structure_engine import (
    MarketStructureResult,
)
from strategy.m15_setup_detector import M15SetupDetector


UTC = timezone.utc
START = datetime(2026, 1, 1, tzinfo=UTC)
BOS_INDEX = 2
BROKEN_LEVEL = 100.0


class FakeAlignmentEngine:
    """Controlled alignment responses for isolated detector tests."""

    def __init__(
        self,
        deny_from=None,
        permitted_direction="BULLISH",
    ):
        self.deny_from = deny_from
        self.permitted_direction = permitted_direction
        self.calls = []

    def evaluate(
        self,
        h4_snapshots,
        h1_snapshots,
        assessment_time,
    ):
        self.calls.append(assessment_time)

        permitted = (
            self.deny_from is None
            or assessment_time < self.deny_from
        )

        return SimpleNamespace(
            directional_permission=permitted,
            permitted_direction=(
                self.permitted_direction if permitted else None
            ),
            h4_known_at_time=assessment_time,
            h1_known_at_time=assessment_time,
            reason=(
                "Test alignment permitted."
                if permitted
                else "Test alignment permission denied."
            ),
        )


def make_candles(count=30, direction="BULLISH"):
    """Create valid, completed M15 OHLC candles."""

    rows = []

    for index in range(count):
        if direction == "BULLISH":
            rows.append({
                "time": START + timedelta(minutes=15 * index),
                "open": 101.0,
                "high": 102.0,
                "low": 100.5,
                "close": 101.0,
                "is_forming": False,
            })
        else:
            rows.append({
                "time": START + timedelta(minutes=15 * index),
                "open": 99.0,
                "high": 99.5,
                "low": 98.0,
                "close": 99.0,
                "is_forming": False,
            })

    return pd.DataFrame(rows)


def make_event(
    candles,
    direction="BULLISH",
    break_index=BOS_INDEX,
    event_type="BOS",
    level=BROKEN_LEVEL,
    break_time=None,
):
    if break_time is None:
        break_time = candles.iloc[break_index]["time"]

    return SimpleNamespace(
        event_type=event_type,
        direction=direction,
        break_time=break_time,
        break_index=break_index,
        level_price=level,
    )


def make_market_structure(candles, events=None):
    return MarketStructureResult(
        timeframe="M15",
        candle_count=len(candles),
        swings=(),
        classified_swings=(),
        trend_snapshots=(),
        bos_events=tuple(events or ()),
        protected_swings=(),
        choch_events=(),
        protected_levels=(),
        active_protected_levels=(),
        retired_protected_levels=(),
    )


def make_detector(
    alignment_engine=None,
    expiry_candles=20,
):
    return M15SetupDetector(
        atr_period=2,
        retest_tolerance_atr=0.10,
        expiry_candles=expiry_candles,
        alignment_engine=(
            alignment_engine or FakeAlignmentEngine()
        ),
    )


def run_detector(
    candles,
    events,
    detector=None,
    h4_snapshots=(),
    h1_snapshots=(),
):
    detector = detector or make_detector()

    return detector.detect(
        candles=candles,
        market_structure=make_market_structure(
            candles,
            events,
        ),
        h4_snapshots=h4_snapshots,
        h1_snapshots=h1_snapshots,
    )


def set_bullish_retest(candles, index):
    candles.loc[index, [
        "open", "high", "low", "close"
    ]] = [101.0, 101.1, 99.9, 100.1]


def set_bullish_rejection(candles, index):
    candles.loc[index, [
        "open", "high", "low", "close"
    ]] = [100.1, 101.0, 100.0, 100.5]


def set_bearish_retest(candles, index):
    candles.loc[index, [
        "open", "high", "low", "close"
    ]] = [99.0, 100.1, 98.9, 99.9]


def set_bearish_rejection(candles, index):
    candles.loc[index, [
        "open", "high", "low", "close"
    ]] = [99.9, 100.0, 99.0, 99.5]


class TestM15SetupDetector:

    def test_bullish_setup_qualifies_after_retest_and_rejection(self):
        candles = make_candles()
        set_bullish_retest(candles, 3)
        set_bullish_rejection(candles, 4)

        event = make_event(candles)

        result = run_detector(candles, [event])
        setup = result.setups[0]

        assert setup.state == "QUALIFIED"
        assert setup.direction == "BULLISH"
        assert setup.retest_time == (
            START + timedelta(minutes=60)
        )
        assert setup.qualification_time == (
            START + timedelta(minutes=75)
        )
        assert setup.atr_value is not None
        assert setup.retest_tolerance is not None
        assert len(result.qualified_setups) == 1

    def test_bearish_setup_qualifies_after_retest_and_rejection(self):
        candles = make_candles(direction="BEARISH")
        set_bearish_retest(candles, 3)
        set_bearish_rejection(candles, 4)

        event = make_event(
            candles,
            direction="BEARISH",
        )

        detector = make_detector(
            alignment_engine=FakeAlignmentEngine(
                permitted_direction="BEARISH",
            ),
        )

        result = run_detector(
            candles,
            [event],
            detector=detector,
        )

        setup = result.setups[0]

        assert setup.state == "QUALIFIED"
        assert setup.direction == "BEARISH"
        assert setup.retest_time is not None
        assert setup.qualification_time is not None

    def test_retest_candle_cannot_confirm_its_own_rejection(self):
        candles = make_candles(count=5)
        set_bullish_retest(candles, 3)

        # The final candle remains neutral and cannot confirm rejection.
        candles.loc[4, [
            "open", "high", "low", "close"
        ]] = [100.0, 100.5, 99.9, 100.0]

        result = run_detector(
            candles,
            [make_event(candles)],
        )

        setup = result.setups[0]

        assert setup.state == "RETEST_DETECTED"
        assert setup.retest_time is not None
        assert setup.qualification_time is None

    def test_waits_when_price_never_retests_zone(self):
        candles = make_candles()

        result = run_detector(
            candles,
            [make_event(candles)],
        )

        setup = result.setups[0]

        assert setup.state == "EXPIRED"
        assert setup.retest_time is None
        assert setup.expiry_time is not None
        assert setup.candles_elapsed == 20

    def test_invalidation_when_bullish_candle_closes_below_zone(self):
        candles = make_candles()

        candles.loc[3, [
            "open", "high", "low", "close"
        ]] = [100.5, 100.5, 99.0, 99.5]

        result = run_detector(
            candles,
            [make_event(candles)],
        )

        assert result.setups[0].state == "INVALIDATED"

    def test_invalidation_when_bearish_candle_closes_above_zone(self):
        candles = make_candles(direction="BEARISH")

        candles.loc[3, [
            "open", "high", "low", "close"
        ]] = [99.5, 101.0, 99.5, 100.5]

        detector = make_detector(
            alignment_engine=FakeAlignmentEngine(
                permitted_direction="BEARISH",
            ),
        )

        result = run_detector(
            candles,
            [make_event(candles, direction="BEARISH")],
            detector=detector,
        )

        assert result.setups[0].state == "INVALIDATED"

    def test_alignment_loss_invalidates_setup(self):
        candles = make_candles()
        set_bullish_retest(candles, 3)

        denial_time = (
            START + timedelta(minutes=75)
        )

        alignment = FakeAlignmentEngine(
            deny_from=denial_time,
        )

        detector = make_detector(
            alignment_engine=alignment,
        )

        result = run_detector(
            candles,
            [make_event(candles)],
            detector=detector,
        )

        setup = result.setups[0]

        assert setup.state == "INVALIDATED"
        assert "permission" in setup.reason.lower()

    def test_initial_alignment_denial_rejects_bos(self):
        candles = make_candles()

        alignment = FakeAlignmentEngine(
            deny_from=START + timedelta(minutes=30),
        )

        detector = make_detector(
            alignment_engine=alignment,
        )

        result = run_detector(
            candles,
            [make_event(candles)],
            detector=detector,
        )

        assert result.setups[0].state == "REJECTED"
        assert result.setups[0].atr_value is None

    def test_twentieth_candle_can_qualify_setup(self):
        candles = make_candles()

        # Candle 19 after BOS detects the retest.
        set_bullish_retest(candles, BOS_INDEX + 19)

        # Candle 20 after BOS confirms rejection.
        set_bullish_rejection(candles, BOS_INDEX + 20)

        result = run_detector(
            candles,
            [make_event(candles)],
        )

        setup = result.setups[0]

        assert setup.state == "QUALIFIED"
        assert setup.candles_elapsed == 20

    def test_retest_first_detected_on_twentieth_candle_expires(self):
        candles = make_candles()

        set_bullish_retest(candles, BOS_INDEX + 20)

        result = run_detector(
            candles,
            [make_event(candles)],
        )

        setup = result.setups[0]

        assert setup.state == "EXPIRED"
        assert setup.retest_time is not None
        assert setup.qualification_time is None
        assert setup.candles_elapsed == 20

    def test_non_bos_event_is_rejected_as_invalid_input(self):
        candles = make_candles()

        event = make_event(
            candles,
            event_type="CHoCH",
        )

        with pytest.raises(ValueError):
            run_detector(candles, [event])

    def test_break_time_must_match_break_index(self):
        candles = make_candles()

        incorrect_time = (
            candles.iloc[BOS_INDEX]["time"]
            + timedelta(minutes=15)
        )

        event = make_event(
            candles,
            break_time=incorrect_time,
        )

        with pytest.raises(ValueError):
            run_detector(candles, [event])

    def test_naive_break_timestamp_is_rejected(self):
        candles = make_candles()

        event = make_event(
            candles,
            break_time=datetime(2026, 1, 1, 0, 30),
        )

        with pytest.raises(ValueError):
            run_detector(candles, [event])

    def test_market_structure_candle_count_must_match(self):
        candles = make_candles()
        event = make_event(candles)

        structure = make_market_structure(
            candles.iloc[:-1].copy(),
            [event],
        )

        detector = make_detector()

        with pytest.raises(ValueError):
            detector.detect(
                candles=candles,
                market_structure=structure,
                h4_snapshots=(),
                h1_snapshots=(),
            )

    def test_non_m15_market_structure_is_rejected(self):
        candles = make_candles()
        structure = make_market_structure(candles)

        incorrect_structure = MarketStructureResult(
            timeframe="H1",
            candle_count=structure.candle_count,
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

        with pytest.raises(ValueError):
            make_detector().detect(
                candles=candles,
                market_structure=incorrect_structure,
                h4_snapshots=(),
                h1_snapshots=(),
            )

    def test_multiple_bos_events_are_processed(self):
        candles = make_candles()

        first_event = make_event(
            candles,
            break_index=2,
        )

        second_event = make_event(
            candles,
            break_index=3,
        )

        result = run_detector(
            candles,
            [second_event, first_event],
        )

        assert len(result.setups) == 2
        assert result.setups[0].break_index == 2
        assert result.setups[1].break_index == 3

    def test_snapshot_generators_are_materialized(self):
        candles = make_candles()

        alignment = FakeAlignmentEngine()
        detector = make_detector(
            alignment_engine=alignment,
        )

        h4 = (item for item in ())
        h1 = (item for item in ())

        result = run_detector(
            candles,
            [
                make_event(candles, break_index=2),
                make_event(candles, break_index=3),
            ],
            detector=detector,
            h4_snapshots=h4,
            h1_snapshots=h1,
        )

        assert len(result.setups) == 2
        assert len(alignment.calls) > 2
