from types import SimpleNamespace

import pandas as pd
import pytest

from research.historical_strategy_replay import (
    HistoricalStrategyReplay,
    ReplayAssessment,
)


class FakeStorage:
    def __init__(self, datasets):
        self.datasets = datasets
        self.loaded = []

    def load_dataset(self, symbol, timeframe):
        self.loaded.append((symbol, timeframe))
        return self.datasets[timeframe].copy()


class FakeOrchestrator:
    def __init__(self):
        self.calls = []

    def analyze(
        self,
        h4_candles,
        h1_candles,
        m15_candles,
        m5_candles,
        assessment_time,
    ):
        self.calls.append({
            "h4": h4_candles.copy(),
            "h1": h1_candles.copy(),
            "m15": m15_candles.copy(),
            "m5": m5_candles.copy(),
            "assessment_time": assessment_time,
        })

        return SimpleNamespace(
            alignment="BULLISH",
            qualified_setups=[object()],
            confirmed_setups=[object()],
            waiting_setups=[],
            expired_setups=[],
            rejected_setups=[],
        )


def make_dataset(timeframe, latest_close, count=30):
    durations = {
        "H4": pd.Timedelta(hours=4),
        "H1": pd.Timedelta(hours=1),
        "M15": pd.Timedelta(minutes=15),
        "M5": pd.Timedelta(minutes=5),
    }

    duration = durations[timeframe]

    opening_times = [
        latest_close - duration * (count - index)
        for index in range(count)
    ]

    return pd.DataFrame({
        "time": opening_times,
        "open": [1.1000] * count,
        "high": [1.1010] * count,
        "low": [1.0990] * count,
        "close": [1.1005] * count,
        "tick_volume": [100] * count,
        "spread": [10] * count,
        "real_volume": [0] * count,
        "is_forming_candle": [False] * count,
    })


def make_datasets(latest_close, count=30):
    return {
        timeframe: make_dataset(
            timeframe,
            latest_close,
            count,
        )
        for timeframe in ("H4", "H1", "M15", "M5")
    }


def make_replay(datasets, orchestrator=None):
    storage = FakeStorage(datasets)
    strategy = (
        orchestrator
        if orchestrator is not None
        else FakeOrchestrator()
    )

    replay = HistoricalStrategyReplay(
        storage=storage,
        strategy_orchestrator=strategy,
    )

    return replay, storage, strategy


def test_loads_all_four_timeframe_datasets():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)

    replay, storage, _ = make_replay(datasets)

    loaded = replay.load_datasets("EURUSD")

    assert set(loaded) == {"H4", "H1", "M15", "M5"}
    assert len(storage.loaded) == 4
    assert all(symbol == "EURUSD" for symbol, _ in storage.loaded)


def test_dataset_timestamps_are_normalized_to_utc():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)

    replay, _, _ = make_replay(datasets)

    loaded = replay.load_datasets("EURUSD")

    for data in loaded.values():
        assert str(data["time"].dt.tz) == "UTC"


def test_rejects_empty_dataset():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)
    datasets["H1"] = pd.DataFrame()

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="empty"):
        replay.load_datasets("EURUSD")


def test_rejects_dataset_with_missing_required_columns():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)
    datasets["M15"] = datasets["M15"].drop(columns=["close"])

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="missing columns"):
        replay.load_datasets("EURUSD")


def test_rejects_duplicate_timestamps():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)
    datasets["M5"].loc[1, "time"] = datasets["M5"].loc[0, "time"]

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="duplicate"):
        replay.load_datasets("EURUSD")


def test_rejects_non_chronological_dataset():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)
    datasets["H4"] = datasets["H4"].iloc[::-1].reset_index(drop=True)

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="chronologically"):
        replay.load_datasets("EURUSD")


def test_rejects_timezone_naive_assessment_time():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="timezone-aware"):
        replay.run(
            start_time=pd.Timestamp("2026-06-01 12:00:00"),
        )


def test_rejects_invalid_assessment_interval():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="assessment_interval"):
        replay.run(assessment_interval=0)


def test_rejects_invalid_symbol():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="Symbol"):
        replay.load_datasets("")


def test_completed_candle_filter_excludes_forming_candles():
    assessment_time = pd.Timestamp("2026-06-01T12:00:00Z")
    data = make_dataset("M5", assessment_time, count=3)

    data.loc[data.index[-1], "is_forming_candle"] = True

    completed = HistoricalStrategyReplay._completed_candles(
        data,
        "M5",
        assessment_time,
    )

    assert len(completed) == 2
    assert not completed["is_forming_candle"].any()


def test_completed_candle_filter_excludes_future_closes():
    assessment_time = pd.Timestamp("2026-06-01T12:00:00Z")
    data = make_dataset("M5", assessment_time, count=3)

    future = pd.DataFrame([{
        "time": assessment_time,
        "open": 1.1000,
        "high": 1.1010,
        "low": 1.0990,
        "close": 1.1005,
        "tick_volume": 100,
        "spread": 10,
        "real_volume": 0,
        "is_forming_candle": False,
    }])

    data = pd.concat([data, future], ignore_index=True)

    completed = HistoricalStrategyReplay._completed_candles(
        data,
        "M5",
        assessment_time,
    )

    assert (completed["time"] + pd.Timedelta(minutes=5) <= assessment_time).all()


def test_replay_records_strategy_pipeline_results():
    assessment_time = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(assessment_time)

    replay, _, orchestrator = make_replay(datasets)

    result = replay.run(
        symbol="EURUSD",
        start_time=assessment_time,
        end_time=assessment_time,
    )

    assert result.assessment_count == 1
    assert len(orchestrator.calls) == 1
    assert result.total_qualified_setups == 1
    assert result.total_confirmed_setups == 1
    assert result.total_waiting_setups == 0
    assert result.total_expired_setups == 0
    assert result.total_rejected_setups == 0

    assessment = result.assessments[0]

    assert isinstance(assessment, ReplayAssessment)
    assert assessment.alignment == "BULLISH"
    assert assessment.assessment_time == assessment_time


def test_replay_does_not_pass_future_candles_to_orchestrator():
    assessment_time = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(assessment_time)

    for timeframe, minutes in {
        "H4": 240,
        "H1": 60,
        "M15": 15,
        "M5": 5,
    }.items():
        future_row = datasets[timeframe].iloc[[-1]].copy()
        future_row["time"] = assessment_time
        future_row["is_forming_candle"] = False

        datasets[timeframe] = pd.concat(
            [datasets[timeframe], future_row],
            ignore_index=True,
        )

    replay, _, orchestrator = make_replay(datasets)

    replay.run(
        start_time=assessment_time,
        end_time=assessment_time,
    )

    assert len(orchestrator.calls) == 1

    call = orchestrator.calls[0]

    for timeframe in ("h4", "h1", "m15", "m5"):
        duration = {
            "h4": pd.Timedelta(hours=4),
            "h1": pd.Timedelta(hours=1),
            "m15": pd.Timedelta(minutes=15),
            "m5": pd.Timedelta(minutes=5),
        }[timeframe]

        close_times = call[timeframe]["time"] + duration

        assert (close_times <= assessment_time).all()


def test_replay_skips_insufficient_history():
    assessment_time = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(assessment_time, count=5)

    replay, _, orchestrator = make_replay(datasets)

    result = replay.run(
        start_time=assessment_time,
        end_time=assessment_time,
    )

    assert result.assessment_count == 0
    assert result.skipped_insufficient_history == 1
    assert len(orchestrator.calls) == 0


def test_replay_skips_stale_timeframe_data():
    assessment_time = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(assessment_time)

    datasets["H4"] = make_dataset(
        "H4",
        assessment_time - pd.Timedelta(hours=8),
    )

    replay, _, orchestrator = make_replay(datasets)

    result = replay.run(
        start_time=assessment_time,
        end_time=assessment_time,
    )

    assert result.assessment_count == 0
    assert result.skipped_stale_data == 1
    assert len(orchestrator.calls) == 0


def test_replay_rejects_end_time_before_start_time():
    latest_close = pd.Timestamp("2026-06-01T12:00:00Z")
    datasets = make_datasets(latest_close)

    replay, _, _ = make_replay(datasets)

    with pytest.raises(ValueError, match="later than"):
        replay.run(
            start_time="2026-06-02T12:00:00Z",
            end_time="2026-06-01T12:00:00Z",
        )


def test_replay_result_totals_are_zero_when_no_assessments():
    result = HistoricalStrategyReplay.__new__(
        HistoricalStrategyReplay
    )

    from research.historical_strategy_replay import HistoricalReplayResult

    empty = HistoricalReplayResult(
        symbol="EURUSD",
        start_time=None,
        end_time=None,
        assessment_count=0,
        skipped_insufficient_history=0,
        skipped_stale_data=0,
        assessments=(),
    )

    assert empty.total_qualified_setups == 0
    assert empty.total_confirmed_setups == 0
    assert empty.total_waiting_setups == 0
    assert empty.total_expired_setups == 0
    assert empty.total_rejected_setups == 0

