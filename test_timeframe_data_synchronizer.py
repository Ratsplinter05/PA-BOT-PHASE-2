
"""Tests for timeframe data synchronisation and validation."""

import pandas as pd
import pytest

from strategy.timeframe_data_synchronizer import (
    SynchronizedTimeframeData,
    TimeframeDataSynchronizer,
)


@pytest.fixture
def assessment_time():
    return pd.Timestamp(
        "2026-01-01 12:00:00",
        tz="UTC",
    )


@pytest.fixture
def candle_data():
    """Create four datasets containing completed and forming candles."""

    return {
        "h4": pd.DataFrame({
            "time": pd.to_datetime([
                "2026-01-01 00:00:00+00:00",
                "2026-01-01 04:00:00+00:00",
                "2026-01-01 08:00:00+00:00",
                "2026-01-01 12:00:00+00:00",
            ]),
            "close": [100.0, 101.0, 102.0, 103.0],
        }),
        "h1": pd.DataFrame({
            "time": pd.to_datetime([
                "2026-01-01 09:00:00+00:00",
                "2026-01-01 10:00:00+00:00",
                "2026-01-01 11:00:00+00:00",
                "2026-01-01 12:00:00+00:00",
            ]),
            "close": [100.0, 101.0, 102.0, 103.0],
        }),
        "m15": pd.DataFrame({
            "time": pd.to_datetime([
                "2026-01-01 11:15:00+00:00",
                "2026-01-01 11:30:00+00:00",
                "2026-01-01 11:45:00+00:00",
                "2026-01-01 12:00:00+00:00",
            ]),
            "close": [100.0, 101.0, 102.0, 103.0],
        }),
        "m5": pd.DataFrame({
            "time": pd.to_datetime([
                "2026-01-01 11:45:00+00:00",
                "2026-01-01 11:50:00+00:00",
                "2026-01-01 11:55:00+00:00",
                "2026-01-01 12:00:00+00:00",
            ]),
            "close": [100.0, 101.0, 102.0, 103.0],
        }),
    }


@pytest.fixture
def synchronizer():
    return TimeframeDataSynchronizer()


def run_synchronization(
    synchronizer,
    candles,
    assessment_time,
):
    return synchronizer.synchronize(
        h4_candles=candles["h4"],
        h1_candles=candles["h1"],
        m15_candles=candles["m15"],
        m5_candles=candles["m5"],
        assessment_time=assessment_time,
    )


def test_successful_synchronization_returns_result(
    synchronizer,
    candle_data,
    assessment_time,
):
    result = run_synchronization(
        synchronizer,
        candle_data,
        assessment_time,
    )

    assert isinstance(result, SynchronizedTimeframeData)
    assert result.assessment_time == assessment_time


def test_incomplete_candles_are_excluded(
    synchronizer,
    candle_data,
    assessment_time,
):
    result = run_synchronization(
        synchronizer,
        candle_data,
        assessment_time,
    )

    assert len(result.h4_candles) == 3
    assert len(result.h1_candles) == 3
    assert len(result.m15_candles) == 3
    assert len(result.m5_candles) == 3


def test_latest_completed_candles_close_at_assessment_time(
    synchronizer,
    candle_data,
    assessment_time,
):
    result = run_synchronization(
        synchronizer,
        candle_data,
        assessment_time,
    )

    for close_time in result.latest_candle_close_times.values():
        assert close_time == assessment_time


def test_all_returned_candle_times_are_utc(
    synchronizer,
    candle_data,
    assessment_time,
):
    result = run_synchronization(
        synchronizer,
        candle_data,
        assessment_time,
    )

    datasets = [
        result.h4_candles,
        result.h1_candles,
        result.m15_candles,
        result.m5_candles,
    ]

    for dataset in datasets:
        assert str(dataset["time"].dt.tz) == "UTC"


def test_input_dataframes_are_not_modified(
    synchronizer,
    candle_data,
    assessment_time,
):
    original_data = {
        key: value.copy(deep=True)
        for key, value in candle_data.items()
    }

    run_synchronization(
        synchronizer,
        candle_data,
        assessment_time,
    )

    for key in candle_data:
        pd.testing.assert_frame_equal(
            candle_data[key],
            original_data[key],
        )


def test_timezone_naive_assessment_time_is_rejected(
    synchronizer,
    candle_data,
):
    naive_time = pd.Timestamp("2026-01-01 12:00:00")

    with pytest.raises(ValueError, match="timezone-aware"):
        run_synchronization(
            synchronizer,
            candle_data,
            naive_time,
        )


@pytest.mark.parametrize(
    "timeframe",
    ["h4", "h1", "m15", "m5"],
)
def test_timezone_naive_candle_timestamp_is_rejected(
    synchronizer,
    candle_data,
    assessment_time,
    timeframe,
):
    candles = candle_data[timeframe].copy()

    candles["time"] = candles["time"].astype(object)
    candles.loc[0, "time"] = pd.Timestamp(
        "2026-01-01 00:00:00"
    )

    candle_data[timeframe] = candles

    with pytest.raises(ValueError, match="timezone-naive"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment_time,
        )


@pytest.mark.parametrize(
    "timeframe",
    ["h4", "h1", "m15", "m5"],
)
def test_duplicate_candle_timestamps_are_rejected(
    synchronizer,
    candle_data,
    assessment_time,
    timeframe,
):
    candles = candle_data[timeframe].copy()

    candles.loc[1, "time"] = candles.loc[0, "time"]

    candle_data[timeframe] = candles

    with pytest.raises(ValueError, match="duplicate"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment_time,
        )


@pytest.mark.parametrize(
    "timeframe",
    ["h4", "h1", "m15", "m5"],
)
def test_unordered_candle_timestamps_are_rejected(
    synchronizer,
    candle_data,
    assessment_time,
    timeframe,
):
    candle_data[timeframe] = (
        candle_data[timeframe].iloc[::-1].reset_index(drop=True)
    )

    with pytest.raises(ValueError, match="chronological"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment_time,
        )


@pytest.mark.parametrize(
    "timeframe",
    ["h4", "h1", "m15", "m5"],
)
def test_empty_candle_dataset_is_rejected(
    synchronizer,
    candle_data,
    assessment_time,
    timeframe,
):
    candle_data[timeframe] = pd.DataFrame()

    with pytest.raises(ValueError, match="cannot be empty"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment_time,
        )


@pytest.mark.parametrize(
    "timeframe",
    ["h4", "h1", "m15", "m5"],
)
def test_missing_time_column_is_rejected(
    synchronizer,
    candle_data,
    assessment_time,
    timeframe,
):
    candle_data[timeframe] = pd.DataFrame({
        "close": [100.0, 101.0],
    })

    with pytest.raises(ValueError, match="'time' column"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment_time,
        )


@pytest.mark.parametrize(
    "timeframe",
    ["h4", "h1", "m15", "m5"],
)
def test_non_dataframe_input_is_rejected(
    synchronizer,
    candle_data,
    assessment_time,
    timeframe,
):
    candle_data[timeframe] = [100.0, 101.0]

    with pytest.raises(TypeError, match="pandas DataFrame"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment_time,
        )


def test_dataset_with_no_completed_candles_is_rejected(
    synchronizer,
    candle_data,
):
    assessment = pd.Timestamp(
        "2025-12-31 00:00:00",
        tz="UTC",
    )

    with pytest.raises(ValueError, match="no completed candles"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment,
        )


def test_stale_candle_data_is_rejected(
    synchronizer,
    candle_data,
):
    assessment = pd.Timestamp(
        "2026-01-01 20:00:00",
        tz="UTC",
    )

    with pytest.raises(ValueError, match="stale"):
        run_synchronization(
            synchronizer,
            candle_data,
            assessment,
        )


def test_custom_freshness_limits_are_respected(
    candle_data,
):
    custom_limits = {
        "H4": pd.Timedelta(hours=12),
        "H1": pd.Timedelta(hours=8),
        "M15": pd.Timedelta(hours=9),
        "M5": pd.Timedelta(hours=9),
    }

    synchronizer = TimeframeDataSynchronizer(
        max_age_by_timeframe=custom_limits,
    )

    assessment = pd.Timestamp(
        "2026-01-01 20:00:00",
        tz="UTC",
    )

    result = run_synchronization(
        synchronizer,
        candle_data,
        assessment,
    )

    assert isinstance(result, SynchronizedTimeframeData)


def test_unknown_custom_timeframe_is_rejected():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        TimeframeDataSynchronizer(
            max_age_by_timeframe={
                "M1": pd.Timedelta(minutes=1),
            }
        )


def test_non_positive_freshness_limit_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        TimeframeDataSynchronizer(
            max_age_by_timeframe={
                "M5": pd.Timedelta(0),
            }
        )


def test_assessment_time_is_normalised_to_utc(
    synchronizer,
    candle_data,
):
    assessment = pd.Timestamp(
        "2026-01-01 14:00:00",
        tz="Africa/Blantyre",
    )

    result = run_synchronization(
        synchronizer,
        candle_data,
        assessment,
    )

    assert result.assessment_time == pd.Timestamp(
        "2026-01-01 12:00:00",
        tz="UTC",
    )
