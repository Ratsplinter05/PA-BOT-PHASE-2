
"""Tests for the strategy orchestrator and synchronizer integration."""

from types import SimpleNamespace

import pandas as pd
import pytest

from strategy.strategy_orchestrator import (
    StrategyOrchestrator,
    StrategyPipelineResult,
)


class FakeTimeframeDataSynchronizer:
    def __init__(self, synchronized_data=None):
        self.calls = []
        self.synchronized_data = synchronized_data

    def synchronize(
        self,
        h4_candles,
        h1_candles,
        m15_candles,
        m5_candles,
        assessment_time,
    ):
        self.calls.append(
            {
                "h4_candles": h4_candles,
                "h1_candles": h1_candles,
                "m15_candles": m15_candles,
                "m5_candles": m5_candles,
                "assessment_time": assessment_time,
            }
        )

        datasets = {
            "H4": h4_candles,
            "H1": h1_candles,
            "M15": m15_candles,
            "M5": m5_candles,
        }

        for timeframe, candles in datasets.items():
            if not isinstance(candles, pd.DataFrame):
                raise TypeError(
                    f"{timeframe} candles must be a pandas DataFrame."
                )

            if candles.empty:
                raise ValueError(
                    f"{timeframe} candles cannot be empty."
                )

        if self.synchronized_data is None:
            self.synchronized_data = SimpleNamespace(
                assessment_time=assessment_time,
                h4_candles=h4_candles.copy(),
                h1_candles=h1_candles.copy(),
                m15_candles=m15_candles.copy(),
                m5_candles=m5_candles.copy(),
                latest_candle_close_times={},
            )

        return self.synchronized_data


class FakeMarketStructureEngine:
    def __init__(self):
        self.multi_timeframe_calls = []
        self.analyze_calls = []

        self.h4_result = SimpleNamespace(
            trend_snapshots=("H4_SNAPSHOT",)
        )

        self.h1_result = SimpleNamespace(
            trend_snapshots=("H1_SNAPSHOT",)
        )

        self.multi_timeframe_result = SimpleNamespace(
            h4_analysis=self.h4_result,
            h1_analysis=self.h1_result,
            alignment="ALIGNED",
        )

    def analyze_multi_timeframe(
        self,
        h4_candles,
        h1_candles,
        assessment_time,
    ):
        self.multi_timeframe_calls.append(
            {
                "h4_candles": h4_candles,
                "h1_candles": h1_candles,
                "assessment_time": assessment_time,
            }
        )

        return self.multi_timeframe_result

    def analyze(self, candles, timeframe="H1"):
        result = SimpleNamespace(
            timeframe=timeframe,
            candle_data=candles,
        )

        self.analyze_calls.append(
            {
                "candles": candles,
                "timeframe": timeframe,
                "result": result,
            }
        )

        return result


class FakeM15SetupDetector:
    def __init__(self):
        self.calls = []
        self.result = SimpleNamespace(
            qualified_setups=("QUALIFIED_SETUP",)
        )

    def detect(
        self,
        candles,
        market_structure,
        h4_snapshots,
        h1_snapshots,
    ):
        self.calls.append(
            {
                "candles": candles,
                "market_structure": market_structure,
                "h4_snapshots": h4_snapshots,
                "h1_snapshots": h1_snapshots,
            }
        )

        return self.result


class FakeM5ConfirmationDetector:
    def __init__(self):
        self.calls = []

        self.result = SimpleNamespace(
            confirmed=("CONFIRMED_SETUP",),
            waiting=("WAITING_SETUP",),
            expired=("EXPIRED_SETUP",),
            rejected=("REJECTED_SETUP",),
        )

    def detect(
        self,
        m15_setups,
        m5_candles,
        m5_market_structure,
    ):
        self.calls.append(
            {
                "m15_setups": m15_setups,
                "m5_candles": m5_candles,
                "m5_market_structure": m5_market_structure,
            }
        )

        return self.result


@pytest.fixture
def candle_data():
    return {
        "H4": pd.DataFrame({"close": [100, 101]}),
        "H1": pd.DataFrame({"close": [100, 101]}),
        "M15": pd.DataFrame({"close": [100, 101]}),
        "M5": pd.DataFrame({"close": [100, 101]}),
    }


@pytest.fixture
def assessment_time():
    return pd.Timestamp("2026-01-01 12:00:00", tz="UTC")


@pytest.fixture
def pipeline(candle_data):
    engine = FakeMarketStructureEngine()
    m15_detector = FakeM15SetupDetector()
    m5_detector = FakeM5ConfirmationDetector()
    synchronizer = FakeTimeframeDataSynchronizer()

    orchestrator = StrategyOrchestrator(
        market_structure_engine=engine,
        m15_setup_detector=m15_detector,
        m5_confirmation_detector=m5_detector,
        timeframe_data_synchronizer=synchronizer,
    )

    return {
        "orchestrator": orchestrator,
        "engine": engine,
        "m15_detector": m15_detector,
        "m5_detector": m5_detector,
        "synchronizer": synchronizer,
        "candles": candle_data,
    }


def run_pipeline(pipeline, assessment_time):
    candles = pipeline["candles"]

    return pipeline["orchestrator"].analyze(
        h4_candles=candles["H4"],
        h1_candles=candles["H1"],
        m15_candles=candles["M15"],
        m5_candles=candles["M5"],
        assessment_time=assessment_time,
    )


def test_complete_pipeline_returns_consolidated_result(
    pipeline, assessment_time
):
    result = run_pipeline(pipeline, assessment_time)

    assert isinstance(result, StrategyPipelineResult)
    assert result.assessment_time == assessment_time
    assert result.alignment == "ALIGNED"
    assert result.qualified_setups == ("QUALIFIED_SETUP",)
    assert result.confirmed_setups == ("CONFIRMED_SETUP",)
    assert result.waiting_setups == ("WAITING_SETUP",)
    assert result.expired_setups == ("EXPIRED_SETUP",)
    assert result.rejected_setups == ("REJECTED_SETUP",)
    assert result.synchronized_data is not None


def test_synchronizer_receives_all_original_inputs(
    pipeline, assessment_time
):
    candles = pipeline["candles"]

    run_pipeline(pipeline, assessment_time)

    assert len(pipeline["synchronizer"].calls) == 1

    call = pipeline["synchronizer"].calls[0]

    assert call["h4_candles"] is candles["H4"]
    assert call["h1_candles"] is candles["H1"]
    assert call["m15_candles"] is candles["M15"]
    assert call["m5_candles"] is candles["M5"]
    assert call["assessment_time"] is assessment_time


def test_h4_h1_analysis_uses_synchronized_inputs(
    pipeline, assessment_time
):
    result = run_pipeline(pipeline, assessment_time)

    call = pipeline["engine"].multi_timeframe_calls[0]

    assert call["h4_candles"] is result.synchronized_data.h4_candles
    assert call["h1_candles"] is result.synchronized_data.h1_candles
    assert call["assessment_time"] == result.assessment_time


def test_m15_detector_receives_h4_h1_snapshots(
    pipeline, assessment_time
):
    result = run_pipeline(pipeline, assessment_time)

    call = pipeline["m15_detector"].calls[0]

    assert call["h4_snapshots"] == ("H4_SNAPSHOT",)
    assert call["h1_snapshots"] == ("H1_SNAPSHOT",)
    assert call["market_structure"] is result.m15_market_structure
    assert call["candles"] is result.synchronized_data.m15_candles


def test_m5_detector_receives_m15_result_and_synchronized_m5(
    pipeline, assessment_time
):
    result = run_pipeline(pipeline, assessment_time)

    call = pipeline["m5_detector"].calls[0]

    assert call["m15_setups"] is result.m15_setups
    assert call["m5_candles"] is result.synchronized_data.m5_candles
    assert call["m5_market_structure"] is result.m5_market_structure


def test_market_structure_engine_analyses_m15_and_m5(
    pipeline, assessment_time
):
    result = run_pipeline(pipeline, assessment_time)

    calls = pipeline["engine"].analyze_calls

    assert len(calls) == 2
    assert calls[0]["timeframe"] == "M15"
    assert calls[0]["candles"] is result.synchronized_data.m15_candles
    assert calls[1]["timeframe"] == "M5"
    assert calls[1]["candles"] is result.synchronized_data.m5_candles


def test_pipeline_result_retains_synchronization_result(
    pipeline, assessment_time
):
    result = run_pipeline(pipeline, assessment_time)

    assert (
        result.synchronized_data
        is pipeline["synchronizer"].synchronized_data
    )


@pytest.mark.parametrize("timeframe", ["H4", "H1", "M15", "M5"])
def test_empty_candle_input_is_rejected(
    pipeline, assessment_time, timeframe
):
    pipeline["candles"][timeframe] = pd.DataFrame()

    with pytest.raises(ValueError):
        run_pipeline(pipeline, assessment_time)

    assert pipeline["engine"].multi_timeframe_calls == []


@pytest.mark.parametrize("timeframe", ["H4", "H1", "M15", "M5"])
def test_non_dataframe_input_is_rejected(
    pipeline, assessment_time, timeframe
):
    pipeline["candles"][timeframe] = [100, 101]

    with pytest.raises(TypeError):
        run_pipeline(pipeline, assessment_time)

    assert pipeline["engine"].multi_timeframe_calls == []


def test_synchronization_failure_stops_the_pipeline(
    pipeline, assessment_time
):
    pipeline["candles"]["M5"] = pd.DataFrame()

    with pytest.raises(ValueError):
        run_pipeline(pipeline, assessment_time)

    assert pipeline["engine"].multi_timeframe_calls == []
    assert pipeline["engine"].analyze_calls == []
    assert pipeline["m15_detector"].calls == []
    assert pipeline["m5_detector"].calls == []


def test_pipeline_result_exposes_setup_categories(
    pipeline, assessment_time
):
    result = run_pipeline(pipeline, assessment_time)

    assert result.qualified_setups == ("QUALIFIED_SETUP",)
    assert result.confirmed_setups == ("CONFIRMED_SETUP",)
    assert result.waiting_setups == ("WAITING_SETUP",)
    assert result.expired_setups == ("EXPIRED_SETUP",)
    assert result.rejected_setups == ("REJECTED_SETUP",)
