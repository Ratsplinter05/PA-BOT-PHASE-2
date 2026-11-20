
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from research.historical_storage import HistoricalDataStorage
from strategy.strategy_orchestrator import StrategyOrchestrator


@dataclass(frozen=True)
class ReplayAssessment:
    """Compact record of one historical strategy assessment."""

    assessment_time: pd.Timestamp
    alignment: object
    qualified_setup_count: int
    confirmed_setup_count: int
    waiting_setup_count: int
    expired_setup_count: int
    rejected_setup_count: int


@dataclass(frozen=True)
class HistoricalReplayResult:
    """Summary and assessment records from a historical replay."""

    symbol: str
    start_time: Optional[pd.Timestamp]
    end_time: Optional[pd.Timestamp]
    assessment_count: int
    skipped_insufficient_history: int
    skipped_stale_data: int
    assessments: tuple[ReplayAssessment, ...]

    @property
    def total_qualified_setups(self) -> int:
        return sum(item.qualified_setup_count for item in self.assessments)

    @property
    def total_confirmed_setups(self) -> int:
        return sum(item.confirmed_setup_count for item in self.assessments)

    @property
    def total_waiting_setups(self) -> int:
        return sum(item.waiting_setup_count for item in self.assessments)

    @property
    def total_expired_setups(self) -> int:
        return sum(item.expired_setup_count for item in self.assessments)

    @property
    def total_rejected_setups(self) -> int:
        return sum(item.rejected_setup_count for item in self.assessments)


class HistoricalStrategyReplay:
    """
    Replays historical candles through the existing strategy pipeline.

    Each assessment uses only candles whose closing times are at or before
    the assessment timestamp. No future candles are passed to the strategy.

    This component evaluates strategy signals. It does not simulate orders,
    entries, stop losses, take profits, or trade execution.
    """

    TIMEFRAME_MINUTES = {
        "H4": 240,
        "H1": 60,
        "M15": 15,
        "M5": 5,
    }

    def __init__(
        self,
        storage=None,
        strategy_orchestrator=None,
        minimum_candles=None,
    ):
        self.storage = (
            storage if storage is not None else HistoricalDataStorage()
        )
        self.strategy_orchestrator = (
            strategy_orchestrator
            if strategy_orchestrator is not None
            else StrategyOrchestrator()
        )

        defaults = {
            "H4": 20,
            "H1": 20,
            "M15": 20,
            "M5": 20,
        }

        if minimum_candles is not None:
            unknown = set(minimum_candles) - set(defaults)
            if unknown:
                raise ValueError(
                    f"Unsupported timeframe(s): {sorted(unknown)}"
                )

            for timeframe, count in minimum_candles.items():
                if not isinstance(count, int) or count < 1:
                    raise ValueError(
                        f"Minimum candle count for {timeframe} "
                        "must be a positive integer."
                    )
                defaults[timeframe] = count

        self.minimum_candles = defaults

    @staticmethod
    def _normalize_datetime(value, field_name):
        if value is None:
            return None

        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            raise ValueError(
                f"{field_name} must be timezone-aware."
            )

        return timestamp.tz_convert("UTC")

    @staticmethod
    def _validate_dataset(data, timeframe):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"{timeframe} dataset must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                f"{timeframe} historical dataset is empty."
            )

        required_columns = {
            "time", "open", "high", "low", "close"
        }

        missing = required_columns - set(data.columns)

        if missing:
            raise ValueError(
                f"{timeframe} dataset is missing columns: "
                f"{sorted(missing)}"
            )

        if data["time"].isna().any():
            raise ValueError(
                f"{timeframe} dataset contains missing timestamps."
            )

        timestamps = pd.to_datetime(data["time"], utc=True)

        if timestamps.duplicated().any():
            raise ValueError(
                f"{timeframe} dataset contains duplicate timestamps."
            )

        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"{timeframe} dataset is not chronologically ordered."
            )

        result = data.copy()
        result["time"] = timestamps

        return result

    @staticmethod
    def _latest_contiguous_m5_segment(completed_m5):
        """
        Return the latest uninterrupted sequence of completed M5 candles.

        A new segment begins whenever adjacent candle opening timestamps
        are not exactly five minutes apart. This handles weekend closures
        and missing historical candles without weakening detector validation.
        """

        if completed_m5.empty:
            return completed_m5.copy()

        result = completed_m5.reset_index(drop=True).copy()

        expected_interval = pd.Timedelta(minutes=5)
        differences = result["time"].diff()

        gap_positions = [
            position
            for position, difference in enumerate(differences)
            if position > 0 and difference != expected_interval
        ]

        if gap_positions:
            latest_segment_start = gap_positions[-1]
            result = result.iloc[latest_segment_start:].copy()

        return result.reset_index(drop=True)

    def load_datasets(self, symbol):
        """Load and validate all four historical timeframe datasets."""

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string.")

        datasets = {}

        for timeframe in self.TIMEFRAME_MINUTES:
            data = self.storage.load_dataset(
                symbol.strip(), timeframe
            )
            datasets[timeframe] = self._validate_dataset(
                data, timeframe
            )

        return datasets

    @classmethod
    def _completed_candles(
        cls,
        data,
        timeframe,
        assessment_time,
    ):
        duration = pd.Timedelta(
            minutes=cls.TIMEFRAME_MINUTES[timeframe]
        )

        closing_times = data["time"] + duration

        completed_mask = closing_times <= assessment_time

        if "is_forming_candle" in data.columns:
            completed_mask &= ~data["is_forming_candle"].fillna(
                True
            ).astype(bool)

        return data.loc[completed_mask].copy()

    def _assess(
        self,
        datasets,
        assessment_time,
    ):
        synchronized_inputs = {}

        for timeframe, data in datasets.items():
            completed = self._completed_candles(
                data,
                timeframe,
                assessment_time,
            )

            if timeframe == "M5":
                completed = self._latest_contiguous_m5_segment(
                    completed
                )

            if len(completed) < self.minimum_candles[timeframe]:
                return None, "insufficient_history"

            latest_close = (
                completed["time"].iloc[-1]
                + pd.Timedelta(
                    minutes=self.TIMEFRAME_MINUTES[timeframe]
                )
            )

            maximum_age = pd.Timedelta(
                minutes=self.TIMEFRAME_MINUTES[timeframe]
            )

            if assessment_time - latest_close > maximum_age:
                return None, "stale_data"

            synchronized_inputs[timeframe] = completed

        pipeline_result = self.strategy_orchestrator.analyze(
            h4_candles=synchronized_inputs["H4"],
            h1_candles=synchronized_inputs["H1"],
            m15_candles=synchronized_inputs["M15"],
            m5_candles=synchronized_inputs["M5"],
            assessment_time=assessment_time,
        )

        assessment = ReplayAssessment(
            assessment_time=assessment_time,
            alignment=pipeline_result.alignment,
            qualified_setup_count=len(
                pipeline_result.qualified_setups
            ),
            confirmed_setup_count=len(
                pipeline_result.confirmed_setups
            ),
            waiting_setup_count=len(
                pipeline_result.waiting_setups
            ),
            expired_setup_count=len(
                pipeline_result.expired_setups
            ),
            rejected_setup_count=len(
                pipeline_result.rejected_setups
            ),
        )

        return assessment, None

    def run(
        self,
        symbol="EURUSD",
        start_time=None,
        end_time=None,
        assessment_interval=1,
        max_assessments=None,
    ):
        """
        Replay the strategy chronologically using completed M5 candles.

        assessment_interval=1 evaluates every completed M5 candle.
        Larger values reduce the number of assessments for quicker trials.
        """

        if (
            not isinstance(assessment_interval, int)
            or assessment_interval < 1
        ):
            raise ValueError(
                "assessment_interval must be a positive integer."
            )

        if max_assessments is not None:
            if (
                not isinstance(max_assessments, int)
                or max_assessments < 1
            ):
                raise ValueError(
                    "max_assessments must be a positive integer."
                )

        start = self._normalize_datetime(
            start_time, "start_time"
        )
        end = self._normalize_datetime(
            end_time, "end_time"
        )

        if start is not None and end is not None and start > end:
            raise ValueError(
                "start_time cannot be later than end_time."
            )

        datasets = self.load_datasets(symbol)
        m5_data = datasets["M5"]

        m5_closes = (
            m5_data["time"]
            + pd.Timedelta(minutes=self.TIMEFRAME_MINUTES["M5"])
        )

        candidate_times = m5_closes.drop_duplicates().sort_values()

        if start is not None:
            candidate_times = candidate_times[
                candidate_times >= start
            ]

        if end is not None:
            candidate_times = candidate_times[
                candidate_times <= end
            ]

        assessments = []
        skipped_insufficient = 0
        skipped_stale = 0
        processed_candidates = 0

        for assessment_time in candidate_times:
            processed_candidates += 1

            if (
                processed_candidates - 1
            ) % assessment_interval != 0:
                continue

            assessment_time = pd.Timestamp(
                assessment_time
            ).tz_convert("UTC")

            assessment, skip_reason = self._assess(
                datasets,
                assessment_time,
            )

            if skip_reason == "insufficient_history":
                skipped_insufficient += 1
                continue

            if skip_reason == "stale_data":
                skipped_stale += 1
                continue

            assessments.append(assessment)

            if (
                max_assessments is not None
                and len(assessments) >= max_assessments
            ):
                break

        actual_start = (
            assessments[0].assessment_time
            if assessments else None
        )
        actual_end = (
            assessments[-1].assessment_time
            if assessments else None
        )

        return HistoricalReplayResult(
            symbol=symbol.strip(),
            start_time=actual_start,
            end_time=actual_end,
            assessment_count=len(assessments),
            skipped_insufficient_history=skipped_insufficient,
            skipped_stale_data=skipped_stale,
            assessments=tuple(assessments),
        )

