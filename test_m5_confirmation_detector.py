
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from research.swing_detection.break_models import StructureBreak
from strategy.m5_confirmation_detector import (
    M5ConfirmationDetector,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def make_candles(count, start_time=BASE_TIME):
    return pd.DataFrame(
        {
            "time": [
                start_time + timedelta(minutes=5 * index)
                for index in range(count)
            ],
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.5] * count,
        }
    )


def make_setup(
    direction="BULLISH",
    state="QUALIFIED",
    qualification_time=BASE_TIME,
    break_time=None,
    break_index=1,
    broken_level=100.0,
):
    if break_time is None:
        break_time = BASE_TIME - timedelta(minutes=15)

    return SimpleNamespace(
        direction=direction,
        state=state,
        break_time=break_time,
        break_index=break_index,
        broken_level=broken_level,
        qualification_time=qualification_time,
    )


def make_bos(
    direction="BULLISH",
    break_index=0,
    break_time=None,
    event_type="BOS",
):
    if break_time is None:
        break_time = (
            BASE_TIME + timedelta(minutes=5 * break_index)
        )

    return StructureBreak(
        event_type=event_type,
        direction=direction,
        break_time=break_time,
        break_index=break_index,
        level_price=100.0,
        level_swing=None,
        close_price=101.0,
        prior_trend="BULLISH",
    )


def make_market_structure(
    candle_count,
    bos_events=None,
    timeframe="M5",
):
    return SimpleNamespace(
        timeframe=timeframe,
        candle_count=candle_count,
        bos_events=tuple(bos_events or []),
    )


def run_detector(
    setup,
    candles,
    bos_events=None,
    window=12,
):
    detector = M5ConfirmationDetector(
        confirmation_window_candles=window,
    )

    m15_result = SimpleNamespace(setups=(setup,))

    market_structure = make_market_structure(
        candle_count=len(candles),
        bos_events=bos_events,
    )

    return detector.detect(
        m15_setups=m15_result,
        m5_candles=candles,
        m5_market_structure=market_structure,
    )


class TestM5ConfirmationDetector:

    def test_bullish_setup_confirmed_by_bullish_bos(self):
        setup = make_setup(direction="BULLISH")
        candles = make_candles(5)
        bos = make_bos(
            direction="BULLISH",
            break_index=2,
        )

        result = run_detector(setup, candles, [bos])

        assert len(result.confirmed) == 1
        assert result.confirmed[0].state == "CONFIRMED"
        assert result.confirmed[0].direction == "BULLISH"
        assert result.confirmed[0].confirmation_break_index == 2

    def test_bearish_setup_confirmed_by_bearish_bos(self):
        setup = make_setup(direction="BEARISH")
        candles = make_candles(5)
        bos = make_bos(
            direction="BEARISH",
            break_index=1,
        )

        result = run_detector(setup, candles, [bos])

        assert len(result.confirmed) == 1
        assert result.confirmed[0].state == "CONFIRMED"
        assert result.confirmed[0].direction == "BEARISH"

    def test_bos_before_qualification_is_ignored(self):
        setup = make_setup(
            qualification_time=BASE_TIME,
        )

        candles = make_candles(
            12,
            start_time=BASE_TIME - timedelta(minutes=5),
        )

        bos = make_bos(
            direction="BULLISH",
            break_index=0,
            break_time=BASE_TIME - timedelta(minutes=5),
        )

        result = run_detector(setup, candles, [bos])

        assert len(result.confirmed) == 0
        assert len(result.waiting) == 1
        assert result.waiting[0].candles_elapsed == 11

    def test_opposite_direction_bos_is_ignored(self):
        setup = make_setup(direction="BULLISH")
        candles = make_candles(5)

        bos = make_bos(
            direction="BEARISH",
            break_index=2,
        )

        result = run_detector(setup, candles, [bos])

        assert len(result.confirmed) == 0
        assert len(result.waiting) == 1

    def test_incomplete_confirmation_window_remains_waiting(self):
        setup = make_setup()
        candles = make_candles(6)

        result = run_detector(setup, candles)

        assert len(result.waiting) == 1
        assert result.waiting[0].candles_elapsed == 6

    def test_setup_expires_after_twelve_eligible_candles(self):
        setup = make_setup()
        candles = make_candles(12)

        result = run_detector(setup, candles)

        assert len(result.expired) == 1
        assert result.expired[0].candles_elapsed == 12

    def test_bos_on_twelfth_candle_confirms_setup(self):
        setup = make_setup()
        candles = make_candles(12)

        bos = make_bos(
            direction="BULLISH",
            break_index=11,
        )

        result = run_detector(setup, candles, [bos])

        assert len(result.confirmed) == 1
        assert result.confirmed[0].confirmation_break_index == 11

    def test_bos_after_twelfth_candle_does_not_confirm(self):
        setup = make_setup()
        candles = make_candles(13)

        bos = make_bos(
            direction="BULLISH",
            break_index=12,
        )

        result = run_detector(setup, candles, [bos])

        assert len(result.confirmed) == 0
        assert len(result.expired) == 1

    def test_non_qualified_m15_setup_is_rejected(self):
        setup = make_setup(state="WAITING_FOR_RETEST")
        candles = make_candles(5)

        result = run_detector(setup, candles)

        assert len(result.rejected) == 1
        assert result.rejected[0].state == "REJECTED"

    def test_earliest_matching_bos_is_selected(self):
        setup = make_setup()
        candles = make_candles(8)

        later_bos = make_bos(
            direction="BULLISH",
            break_index=5,
        )

        earlier_bos = make_bos(
            direction="BULLISH",
            break_index=2,
        )

        result = run_detector(
            setup,
            candles,
            [later_bos, earlier_bos],
        )

        assert len(result.confirmed) == 1
        assert result.confirmed[0].confirmation_break_index == 2

    def test_setup_identity_is_stable(self):
        setup = make_setup()
        candles = make_candles(5)

        first_result = run_detector(setup, candles)
        second_result = run_detector(setup, candles)

        assert (
            first_result.waiting[0].setup_id
            == second_result.waiting[0].setup_id
        )

    def test_candle_count_mismatch_is_rejected(self):
        setup = make_setup()
        candles = make_candles(5)

        detector = M5ConfirmationDetector()

        m15_result = SimpleNamespace(setups=(setup,))

        market_structure = make_market_structure(
            candle_count=4,
        )

        with pytest.raises(ValueError, match="candle count"):
            detector.detect(
                m15_result,
                candles,
                market_structure,
            )

    def test_non_m5_market_structure_is_rejected(self):
        setup = make_setup()
        candles = make_candles(5)

        detector = M5ConfirmationDetector()

        m15_result = SimpleNamespace(setups=(setup,))

        market_structure = make_market_structure(
            candle_count=5,
            timeframe="M15",
        )

        with pytest.raises(ValueError, match="M5 timeframe"):
            detector.detect(
                m15_result,
                candles,
                market_structure,
            )

    def test_naive_candle_timestamp_is_rejected(self):
        setup = make_setup()
        candles = make_candles(5)

        candles["time"] = candles["time"].astype(object)
        candles.loc[0, "time"] = datetime(2026, 1, 1, 10, 0)

        with pytest.raises(ValueError, match="timezone-aware"):
            run_detector(setup, candles)

    def test_bos_time_must_match_its_candle_index(self):
        setup = make_setup()
        candles = make_candles(5)

        bos = make_bos(
            direction="BULLISH",
            break_index=2,
            break_time=BASE_TIME + timedelta(minutes=5),
        )

        with pytest.raises(ValueError, match="must match"):
            run_detector(setup, candles, [bos])

    def test_duplicate_candle_timestamps_are_rejected(self):
        setup = make_setup()
        candles = make_candles(5)

        candles.loc[2, "time"] = candles.loc[1, "time"]

        with pytest.raises(ValueError, match="strictly increasing"):
            run_detector(setup, candles)

    def test_result_properties_filter_confirmation_states(self):
        qualified_setup = make_setup(
            direction="BULLISH",
            qualification_time=BASE_TIME,
        )

        waiting_setup = make_setup(
            direction="BEARISH",
            qualification_time=BASE_TIME,
        )

        rejected_setup = make_setup(
            state="WAITING_FOR_RETEST",
        )

        candles = make_candles(5)

        detector = M5ConfirmationDetector()

        m15_result = SimpleNamespace(
            setups=(
                qualified_setup,
                waiting_setup,
                rejected_setup,
            )
        )

        market_structure = make_market_structure(
            candle_count=5,
        )

        result = detector.detect(
            m15_result,
            candles,
            market_structure,
        )

        assert len(result.waiting) == 2
        assert len(result.rejected) == 1
        assert len(result.confirmed) == 0
        assert len(result.expired) == 0
