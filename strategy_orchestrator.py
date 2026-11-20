
"""Coordinates the H4, H1, M15 and M5 strategy analysis pipeline.

The orchestrator synchronizes and validates candle data before passing it
through the existing market structure, M15 setup and M5 confirmation
components.

It is analysis-only. It does not calculate trade execution parameters,
place orders, modify positions or manage open trades.
"""

from dataclasses import dataclass
from typing import Any

from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
)
from strategy.m15_setup_detector import (
    M15SetupDetector,
    M15SetupResult,
)
from strategy.m5_confirmation_detector import (
    M5ConfirmationDetector,
    M5ConfirmationResult,
)
from strategy.timeframe_data_synchronizer import (
    SynchronizedTimeframeData,
    TimeframeDataSynchronizer,
)


@dataclass(frozen=True)
class StrategyPipelineResult:
    """Consolidated output from the multi-timeframe strategy pipeline."""

    assessment_time: Any
    multi_timeframe_analysis: Any
    m15_market_structure: Any
    m15_setups: M15SetupResult
    m5_market_structure: Any
    m5_confirmations: M5ConfirmationResult
    synchronized_data: SynchronizedTimeframeData

    @property
    def alignment(self):
        return self.multi_timeframe_analysis.alignment

    @property
    def qualified_setups(self):
        return self.m15_setups.qualified_setups

    @property
    def confirmed_setups(self):
        return self.m5_confirmations.confirmed

    @property
    def waiting_setups(self):
        return self.m5_confirmations.waiting

    @property
    def expired_setups(self):
        return self.m5_confirmations.expired

    @property
    def rejected_setups(self):
        return self.m5_confirmations.rejected


class StrategyOrchestrator:
    """Coordinates synchronized multi-timeframe strategy analysis."""

    def __init__(
        self,
        market_structure_engine=None,
        m15_setup_detector=None,
        m5_confirmation_detector=None,
        timeframe_data_synchronizer=None,
    ):
        self.market_structure_engine = (
            market_structure_engine
            if market_structure_engine is not None
            else MarketStructureEngine()
        )

        self.m15_setup_detector = (
            m15_setup_detector
            if m15_setup_detector is not None
            else M15SetupDetector()
        )

        self.m5_confirmation_detector = (
            m5_confirmation_detector
            if m5_confirmation_detector is not None
            else M5ConfirmationDetector()
        )

        self.timeframe_data_synchronizer = (
            timeframe_data_synchronizer
            if timeframe_data_synchronizer is not None
            else TimeframeDataSynchronizer()
        )

    def analyze(
        self,
        h4_candles,
        h1_candles,
        m15_candles,
        m5_candles,
        assessment_time,
    ) -> StrategyPipelineResult:
        """Synchronize candle data and execute the complete strategy pipeline."""

        synchronized_data = (
            self.timeframe_data_synchronizer.synchronize(
                h4_candles=h4_candles,
                h1_candles=h1_candles,
                m15_candles=m15_candles,
                m5_candles=m5_candles,
                assessment_time=assessment_time,
            )
        )

        synchronized_assessment_time = (
            synchronized_data.assessment_time
        )

        synchronized_h4 = synchronized_data.h4_candles
        synchronized_h1 = synchronized_data.h1_candles
        synchronized_m15 = synchronized_data.m15_candles
        synchronized_m5 = synchronized_data.m5_candles

        multi_timeframe_analysis = (
            self.market_structure_engine.analyze_multi_timeframe(
                h4_candles=synchronized_h4,
                h1_candles=synchronized_h1,
                assessment_time=synchronized_assessment_time,
            )
        )

        m15_market_structure = (
            self.market_structure_engine.analyze(
                synchronized_m15,
                timeframe="M15",
            )
        )

        m15_setups = self.m15_setup_detector.detect(
            candles=synchronized_m15,
            market_structure=m15_market_structure,
            h4_snapshots=(
                multi_timeframe_analysis.h4_analysis.trend_snapshots
            ),
            h1_snapshots=(
                multi_timeframe_analysis.h1_analysis.trend_snapshots
            ),
        )

        m5_market_structure = (
            self.market_structure_engine.analyze(
                synchronized_m5,
                timeframe="M5",
            )
        )

        m5_confirmations = self.m5_confirmation_detector.detect(
            m15_setups=m15_setups,
            m5_candles=synchronized_m5,
            m5_market_structure=m5_market_structure,
        )

        return StrategyPipelineResult(
            assessment_time=synchronized_assessment_time,
            multi_timeframe_analysis=multi_timeframe_analysis,
            m15_market_structure=m15_market_structure,
            m15_setups=m15_setups,
            m5_market_structure=m5_market_structure,
            m5_confirmations=m5_confirmations,
            synchronized_data=synchronized_data,
        )
