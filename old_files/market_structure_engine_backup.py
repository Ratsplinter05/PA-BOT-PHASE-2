
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from research.swing_detection.alignment_engine import AlignmentEngine
from research.swing_detection.alignment_models import AlignmentResult
from research.swing_detection.bos_detector import BOSDetector
from research.swing_detection.choch_detector import CHoCHDetector
from research.swing_detection.fractal_detector import FractalSwingDetector
from research.swing_detection.protected_level_registry import (
    ProtectedLevelRegistry,
)
from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.registry_models import ProtectedLevel
from research.swing_detection.protected_swing_tracker import (
    ProtectedSwingTracker,
)
from research.swing_detection.structure_classifier import (
    StructureClassifier,
)
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.trend_classifier import TrendClassifier
from research.swing_detection.trend_models import TrendSnapshot
from research.swing_detection.break_models import StructureBreak


@dataclass(frozen=True)
class MarketStructureResult:
    """Complete market-structure analysis for one timeframe."""

    timeframe: str
    candle_count: int

    swings: tuple[SwingPoint, ...]
    classified_swings: tuple[ClassifiedSwing, ...]
    trend_snapshots: tuple[TrendSnapshot, ...]

    bos_events: tuple[StructureBreak, ...]
    protected_swings: tuple[ProtectedSwing, ...]
    choch_events: tuple[StructureBreak, ...]

    protected_levels: tuple[ProtectedLevel, ...]
    active_protected_levels: tuple[ProtectedLevel, ...]
    retired_protected_levels: tuple[ProtectedLevel, ...]


@dataclass(frozen=True)
class MultiTimeframeMarketStructureResult:
    """Combined H4/H1 structural analysis and directional alignment."""

    h4_analysis: MarketStructureResult
    h1_analysis: MarketStructureResult
    alignment: AlignmentResult


class MarketStructureEngine:
    """
    Coordinates the existing market-structure components.

    Processing stages:

    1. Fractal swing detection.
    2. Swing classification.
    3. Trend classification.
    4. BOS detection.
    5. Protected swing tracking.
    6. Protected level registration.
    7. CHoCH detection.
    8. Chronological protected-level retirement.

    Detection remains separate from registry management.

    No trading orders are created or executed.
    """

    def __init__(
        self,
        left_bars: int = 2,
        right_bars: int = 2,
        price_tolerance: float = 0.0,
    ) -> None:

        self.fractal_detector = FractalSwingDetector(
            left_bars=left_bars,
            right_bars=right_bars,
        )

        self.structure_classifier = StructureClassifier(
            price_tolerance=price_tolerance,
        )

        self.trend_classifier = TrendClassifier()
        self.bos_detector = BOSDetector()
        self.protected_swing_tracker = ProtectedSwingTracker()
        self.choch_detector = CHoCHDetector()
        self.alignment_engine = AlignmentEngine()

    @staticmethod
    def _protected_identity(protected: ProtectedSwing) -> tuple:
        swing = protected.swing.swing

        return (
            protected.trend,
            swing.swing_type,
            swing.swing_index,
            swing.confirmation_index,
        )

    @staticmethod
    def _level_identity(level: ProtectedLevel) -> tuple:
        swing = level.swing.swing

        return (
            level.trend,
            swing.swing_type,
            swing.swing_index,
            swing.confirmation_index,
        )

    @staticmethod
    def _event_identity(event: StructureBreak) -> tuple:
        swing = event.level_swing.swing

        return (
            event.prior_trend,
            swing.swing_type,
            swing.swing_index,
            swing.confirmation_index,
        )

    @staticmethod
    def _protected_sort_key(protected: ProtectedSwing) -> tuple:
        return (
            protected.established_by.break_index,
            protected.swing.swing.confirmation_index,
            protected.swing.swing.swing_index,
            protected.trend,
        )

    @staticmethod
    def _event_sort_key(event: StructureBreak) -> tuple:
        return (
            event.break_index,
            event.break_time,
            event.direction,
        )

    def _detect_choch_events(
        self,
        candles: pd.DataFrame,
        protected_swings: tuple[ProtectedSwing, ...],
    ) -> tuple[StructureBreak, ...]:
        """
        Detect CHoCH events against all structurally established levels.

        This is a detection pass only. It does not mutate the final
        chronological registry.

        CHoCHDetector checks each level's activation_index, so a
        protected level cannot trigger before its BOS activation.
        """

        detection_registry = ProtectedLevelRegistry()

        ordered_protected = sorted(
            protected_swings,
            key=self._protected_sort_key,
        )

        for protected in ordered_protected:
            detection_registry.register([protected])

        events = self.choch_detector.detect(
            candles=candles,
            registry=detection_registry,
        )

        return tuple(
            sorted(events, key=self._event_sort_key)
        )

    def _replay_registry(
        self,
        protected_swings: tuple[ProtectedSwing, ...],
        choch_events: tuple[StructureBreak, ...],
    ) -> ProtectedLevelRegistry:
        """
        Reconstruct registry state in chronological candle order.

        BOS events register protected levels on their break candles.
        Their activation_index determines when they become usable.

        CHoCH events retire matching active levels on their own
        confirmation candles.

        BOS registration is processed before CHoCH retirement when
        both events occur on the same candle. A newly registered
        level cannot trigger on that candle because activation occurs
        on the following candle.
        """

        registry = ProtectedLevelRegistry()

        protected_by_index = defaultdict(list)
        choch_by_index = defaultdict(list)

        for protected in protected_swings:
            break_index = protected.established_by.break_index

            protected_by_index[break_index].append(protected)

        for event in choch_events:
            choch_by_index[event.break_index].append(event)

        timeline = sorted(
            set(protected_by_index) | set(choch_by_index)
        )

        for candle_index in timeline:

            current_protected = sorted(
                protected_by_index.get(candle_index, []),
                key=self._protected_sort_key,
            )

            # Register BOS-established protected levels first.
            for protected in current_protected:
                registry.register([protected])

            current_choch_events = sorted(
                choch_by_index.get(candle_index, []),
                key=self._event_sort_key,
            )

            # Retire the exact level broken by each CHoCH event.
            for event in current_choch_events:

                identity = self._event_identity(event)

                matching_levels = [
                    level
                    for level in registry.get_active_levels(
                        as_of_index=candle_index
                    )
                    if self._level_identity(level) == identity
                ]

                if len(matching_levels) != 1:
                    raise RuntimeError(
                        "CHoCH event could not be matched to exactly "
                        "one active protected level. "
                        f"Identity: {identity}; "
                        f"candle index: {candle_index}."
                    )

                registry.retire(
                    matching_levels[0],
                    event,
                )

        return registry

    def analyze(
        self,
        candles: pd.DataFrame,
        timeframe: str = "H1",
    ) -> MarketStructureResult:
        """
        Perform complete market-structure analysis for one timeframe.

        Supported timeframes are defined by BOSDetector.
        Candles must contain completed historical OHLC data.
        """

        if not isinstance(candles, pd.DataFrame):
            raise TypeError(
                "candles must be a pandas DataFrame."
            )

        if candles.empty:
            raise ValueError(
                "Candle dataframe cannot be empty."
            )

        if len(candles) < 2:
            raise ValueError(
                "At least two candles are required for "
                "market-structure analysis."
            )

        if not isinstance(timeframe, str):
            raise TypeError(
                "timeframe must be a string."
            )

        if timeframe not in BOSDetector.TIMEFRAME_DURATIONS:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}."
            )

        # Stage 1: Detect confirmed fractal swings.
        swings = tuple(
            self.fractal_detector.detect(candles)
        )

        # Stage 2: Classify confirmed swings.
        classified_swings = tuple(
            self.structure_classifier.classify(swings)
        )

        # Stage 3: Generate chronological trend snapshots.
        trend_snapshots = tuple(
            self.trend_classifier.classify(classified_swings)
        )

        # Stage 4: Detect confirmed structural breaks.
        bos_events = tuple(
            sorted(
                self.bos_detector.detect(
                    candles=candles,
                    swings=classified_swings,
                    trend_snapshots=trend_snapshots,
                    timeframe=timeframe,
                ),
                key=self._event_sort_key,
            )
        )

        # Stage 5: Identify protected swings established by BOS.
        protected_swings = tuple(
            sorted(
                self.protected_swing_tracker.track(
                    swings=classified_swings,
                    bos_events=bos_events,
                ),
                key=self._protected_sort_key,
            )
        )

        # Stage 6 and 7: Detect CHoCH against established levels.
        choch_events = self._detect_choch_events(
            candles=candles,
            protected_swings=protected_swings,
        )

        # Stage 8: Reconstruct the final registry chronologically.
        registry = self._replay_registry(
            protected_swings=protected_swings,
            choch_events=choch_events,
        )

        return MarketStructureResult(
            timeframe=timeframe,
            candle_count=len(candles),

            swings=swings,
            classified_swings=classified_swings,
            trend_snapshots=trend_snapshots,

            bos_events=bos_events,
            protected_swings=protected_swings,
            choch_events=choch_events,

            protected_levels=registry.levels,
            active_protected_levels=tuple(
                registry.get_active_levels()
            ),
            retired_protected_levels=tuple(
                registry.get_retired_levels()
            ),
        )

    def analyze_multi_timeframe(
        self,
        h4_candles: pd.DataFrame,
        h1_candles: pd.DataFrame,
        assessment_time,
    ) -> MultiTimeframeMarketStructureResult:
        """
        Analyze H4 and H1 independently, then evaluate alignment.

        Directional permission is granted only when both timeframes
        have eligible, confirmed and matching directional trends.
        """

        h4_analysis = self.analyze(
            candles=h4_candles,
            timeframe="H4",
        )

        h1_analysis = self.analyze(
            candles=h1_candles,
            timeframe="H1",
        )

        alignment = self.alignment_engine.evaluate(
            h4_snapshots=h4_analysis.trend_snapshots,
            h1_snapshots=h1_analysis.trend_snapshots,
            assessment_time=assessment_time,
        )

        return MultiTimeframeMarketStructureResult(
            h4_analysis=h4_analysis,
            h1_analysis=h1_analysis,
            alignment=alignment,
        )
