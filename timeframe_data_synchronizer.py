
"""Validates and synchronises H4, H1, M15 and M5 candle datasets.

The synchronizer:
- Requires timezone-aware candle timestamps.
- Converts timestamps to UTC.
- Rejects duplicate or chronologically unordered candles.
- Excludes candles that have not closed by assessment_time.
- Checks the age of the latest completed candle.
- Returns independent copies of the validated candle datasets.

This component does not fetch market data or execute trades.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class SynchronizedTimeframeData:
    """Validated candle datasets for the four strategy timeframes."""

    assessment_time: pd.Timestamp
    h4_candles: pd.DataFrame
    h1_candles: pd.DataFrame
    m15_candles: pd.DataFrame
    m5_candles: pd.DataFrame
    latest_candle_close_times: dict


class TimeframeDataSynchronizer:
    """Validates and synchronises four timeframe candle datasets."""

    TIMEFRAME_DURATIONS = {
        "H4": pd.Timedelta(hours=4),
        "H1": pd.Timedelta(hours=1),
        "M15": pd.Timedelta(minutes=15),
        "M5": pd.Timedelta(minutes=5),
    }

    def __init__(self, max_age_by_timeframe: Optional[dict] = None):
        """Initialise timeframe freshness limits.

        By default, the maximum permitted age of the latest completed
        candle is one timeframe duration.

        Custom limits can be supplied as a dictionary, for example:

            {
                "H4": pd.Timedelta(hours=8),
                "H1": pd.Timedelta(hours=2),
                "M15": pd.Timedelta(minutes=30),
                "M5": pd.Timedelta(minutes=10),
            }
        """

        self.max_age_by_timeframe = {}

        supplied_limits = max_age_by_timeframe or {}

        unknown_timeframes = (
            set(supplied_limits) - set(self.TIMEFRAME_DURATIONS)
        )

        if unknown_timeframes:
            raise ValueError(
                "Unsupported timeframe freshness limits: "
                f"{sorted(unknown_timeframes)}"
            )

        for timeframe, duration in self.TIMEFRAME_DURATIONS.items():
            maximum_age = supplied_limits.get(
                timeframe,
                duration,
            )

            try:
                maximum_age = pd.Timedelta(maximum_age)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid maximum age for {timeframe}."
                ) from exc

            if pd.isna(maximum_age) or maximum_age <= pd.Timedelta(0):
                raise ValueError(
                    f"Maximum age for {timeframe} must be positive."
                )

            self.max_age_by_timeframe[timeframe] = maximum_age

    @staticmethod
    def _normalise_assessment_time(assessment_time) -> pd.Timestamp:
        """Validate assessment_time and convert it to UTC."""

        try:
            timestamp = pd.Timestamp(assessment_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "assessment_time must be a valid timestamp."
            ) from exc

        if pd.isna(timestamp):
            raise ValueError(
                "assessment_time cannot be NaT."
            )

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                "assessment_time must be timezone-aware."
            )

        return timestamp.tz_convert("UTC")

    @staticmethod
    def _normalise_candle_times(
        candles: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """Validate candle timestamps and return a normalised copy."""

        if not isinstance(candles, pd.DataFrame):
            raise TypeError(
                f"{timeframe} candles must be a pandas DataFrame."
            )

        if candles.empty:
            raise ValueError(
                f"{timeframe} candle data cannot be empty."
            )

        if "time" not in candles.columns:
            raise ValueError(
                f"{timeframe} candles must contain a 'time' column."
            )

        normalised = candles.copy()

        timestamps = []

        for index, value in normalised["time"].items():
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{timeframe} contains an invalid timestamp "
                    f"at index {index}."
                ) from exc

            if pd.isna(timestamp):
                raise ValueError(
                    f"{timeframe} contains a missing timestamp "
                    f"at index {index}."
                )

            if (
                timestamp.tzinfo is None
                or timestamp.utcoffset() is None
            ):
                raise ValueError(
                    f"{timeframe} contains a timezone-naive timestamp "
                    f"at index {index}."
                )

            timestamps.append(timestamp.tz_convert("UTC"))

        normalised["time"] = pd.Series(
            pd.DatetimeIndex(timestamps),
            index=normalised.index,
        )

        if normalised["time"].duplicated().any():
            raise ValueError(
                f"{timeframe} contains duplicate candle timestamps."
            )

        if not normalised["time"].is_monotonic_increasing:
            raise ValueError(
                f"{timeframe} candle timestamps must be "
                "in chronological order."
            )

        return normalised

    def _process_timeframe(
        self,
        candles: pd.DataFrame,
        timeframe: str,
        assessment_time: pd.Timestamp,
    ):
        """Filter completed candles and validate their freshness."""

        if timeframe not in self.TIMEFRAME_DURATIONS:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        normalised = self._normalise_candle_times(
            candles=candles,
            timeframe=timeframe,
        )

        candle_duration = self.TIMEFRAME_DURATIONS[timeframe]

        # Candle timestamps represent opening times.
        # A candle is completed only when its closing time is
        # at or before the assessment time.
        candle_close_times = (
            normalised["time"] + candle_duration
        )

        completed_mask = candle_close_times <= assessment_time

        completed_candles = normalised.loc[
            completed_mask
        ].copy()

        if completed_candles.empty:
            raise ValueError(
                f"{timeframe} has no completed candles at "
                "the supplied assessment_time."
            )

        latest_candle_open = completed_candles["time"].iloc[-1]

        latest_candle_close = (
            latest_candle_open + candle_duration
        )

        candle_age = (
            assessment_time - latest_candle_close
        )

        maximum_age = self.max_age_by_timeframe[timeframe]

        if candle_age > maximum_age:
            raise ValueError(
                f"{timeframe} candle data is stale. "
                f"Latest candle closed at "
                f"{latest_candle_close.isoformat()}, "
                f"assessment time is {assessment_time.isoformat()}, "
                f"and maximum permitted age is {maximum_age}."
            )

        return completed_candles, latest_candle_close

    def synchronize(
        self,
        h4_candles: pd.DataFrame,
        h1_candles: pd.DataFrame,
        m15_candles: pd.DataFrame,
        m5_candles: pd.DataFrame,
        assessment_time,
    ) -> SynchronizedTimeframeData:
        """Validate and synchronise all four timeframe datasets."""

        normalised_assessment_time = (
            self._normalise_assessment_time(assessment_time)
        )

        candle_inputs = {
            "H4": h4_candles,
            "H1": h1_candles,
            "M15": m15_candles,
            "M5": m5_candles,
        }

        processed = {}
        latest_close_times = {}

        for timeframe, candles in candle_inputs.items():
            completed_candles, latest_close = (
                self._process_timeframe(
                    candles=candles,
                    timeframe=timeframe,
                    assessment_time=normalised_assessment_time,
                )
            )

            processed[timeframe] = completed_candles
            latest_close_times[timeframe] = latest_close

        return SynchronizedTimeframeData(
            assessment_time=normalised_assessment_time,
            h4_candles=processed["H4"],
            h1_candles=processed["H1"],
            m15_candles=processed["M15"],
            m5_candles=processed["M5"],
            latest_candle_close_times=latest_close_times,
        )

