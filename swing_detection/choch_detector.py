
import math
from datetime import datetime
from collections.abc import Iterable
from numbers import Integral, Real

import pandas as pd

from research.swing_detection.break_models import StructureBreak
from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.registry_models import ProtectedLevel
from research.swing_detection.protected_level_registry import ProtectedLevelRegistry


class CHoCHDetector:
    """
    Detects Change of Character events using active protected levels.

    Bullish protected level:
        A bearish CHoCH occurs when a candle closes below the
        protected Higher Low after crossing it from above.

    Bearish protected level:
        A bullish CHoCH occurs when a candle closes above the
        protected Lower High after crossing it from below.

    Detection does not mutate or retire registry levels.
    Retirement is handled separately by ProtectedLevelRegistry.
    """

    REQUIRED_COLUMNS = {"time", "close"}

    @staticmethod
    def _validate_timestamp(value, index):

        if not isinstance(value, datetime):
            raise TypeError(
                f"Timestamp at index {index} must be a datetime."
            )

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"Timestamp at index {index} must be timezone-aware."
            )

    @staticmethod
    def _identity(level: ProtectedLevel) -> tuple:

        swing = level.swing.swing

        return (
            level.trend,
            swing.swing_type,
            swing.swing_index,
            swing.confirmation_index,
        )

    @staticmethod
    def _validate_forming_flags(candles: pd.DataFrame) -> None:
        """
        Reject malformed forming-candle flags and all forming candles.

        Only actual Boolean columns are accepted. Values such as the
        string "False" must not be silently interpreted as True.
        """

        if "is_forming_candle" not in candles.columns:
            return

        forming = candles["is_forming_candle"]

        if forming.isna().any():
            raise ValueError(
                "Forming-candle flags cannot contain missing values."
            )

        if not pd.api.types.is_bool_dtype(forming.dtype):
            raise TypeError(
                "Forming-candle flags must contain Boolean values."
            )

        if forming.any():
            raise ValueError(
                "Forming candles must be excluded before CHoCH detection."
            )

    @staticmethod
    def _validate_registry_levels(levels) -> list:
        """
        Validate protected-level records before using them.

        This prevents malformed registry records from reaching the
        crossing logic.
        """

        validated_levels = []
        seen_identities = set()

        for level in levels:

            if not isinstance(level, ProtectedLevel):
                raise TypeError(
                    "Registry contains an invalid protected level."
                )

            protected = level.protected_swing

            if not isinstance(protected, ProtectedSwing):
                raise TypeError(
                    "Protected level contains an invalid protected swing."
                )

            classified_swing = protected.swing

            if not isinstance(classified_swing, ClassifiedSwing):
                raise TypeError(
                    "Protected swing must contain a ClassifiedSwing."
                )

            swing = classified_swing.swing

            if not isinstance(swing, SwingPoint):
                raise TypeError(
                    "Classified swing must contain a SwingPoint."
                )

            if level.trend not in ("BULLISH", "BEARISH"):
                raise ValueError(
                    "Protected level has an invalid trend."
                )

            if swing.swing_type not in ("HIGH", "LOW"):
                raise ValueError(
                    "Protected level has an invalid swing type."
                )

            if (
                not isinstance(level.price, Real)
                or isinstance(level.price, bool)
                or not math.isfinite(level.price)
                or level.price <= 0
            ):
                raise ValueError(
                    "Protected level price must be finite and positive."
                )

            if (
                not isinstance(level.activation_index, Integral)
                or isinstance(level.activation_index, bool)
                or level.activation_index < 0
            ):
                raise ValueError(
                    "Protected level activation index is invalid."
                )

            if not isinstance(level.active, bool):
                raise TypeError(
                    "Protected level active flag must be Boolean."
                )

            if (
                not isinstance(swing.swing_index, Integral)
                or isinstance(swing.swing_index, bool)
                or swing.swing_index < 0
            ):
                raise ValueError(
                    "Protected swing index must be a non-negative integer."
                )

            if (
                not isinstance(swing.confirmation_index, Integral)
                or isinstance(swing.confirmation_index, bool)
                or swing.confirmation_index <= swing.swing_index
            ):
                raise ValueError(
                    "Protected swing confirmation index is invalid."
                )

            if level.trend == "BULLISH" and swing.swing_type != "LOW":
                raise ValueError(
                    "A bullish protected level must reference a swing low."
                )

            if level.trend == "BEARISH" and swing.swing_type != "HIGH":
                raise ValueError(
                    "A bearish protected level must reference a swing high."
                )

            identity = CHoCHDetector._identity(level)

            if identity in seen_identities:
                raise ValueError(
                    "Registry contains duplicate protected levels."
                )

            seen_identities.add(identity)
            validated_levels.append(level)

        return validated_levels

    def detect(
        self,
        candles: pd.DataFrame,
        registry: ProtectedLevelRegistry,
        start_index: int = 1,
    ) -> list[StructureBreak]:

        if not isinstance(candles, pd.DataFrame):
            raise TypeError("candles must be a pandas DataFrame.")

        if not isinstance(registry, ProtectedLevelRegistry):
            raise TypeError(
                "registry must be a ProtectedLevelRegistry."
            )

        missing = self.REQUIRED_COLUMNS.difference(candles.columns)

        if missing:
            raise ValueError(
                f"Missing required candle columns: {sorted(missing)}"
            )

        if candles.empty:
            raise ValueError("Candle dataframe cannot be empty.")

        if (
            not isinstance(start_index, Integral)
            or isinstance(start_index, bool)
            or start_index < 1
            or start_index >= len(candles)
        ):
            raise ValueError(
                "start_index must be an integer between 1 "
                "and the final candle index."
            )

        self._validate_forming_flags(candles)

        times = candles["time"].tolist()
        closes = candles["close"].tolist()

        for index, candle_time in enumerate(times):

            self._validate_timestamp(candle_time, index)

            if index > 0 and candle_time <= times[index - 1]:
                raise ValueError(
                    "Candle timestamps must be strictly increasing."
                )

        validated_closes = []

        for index, value in enumerate(closes):

            try:
                close = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid close price at index {index}."
                ) from None

            if not math.isfinite(close) or close <= 0:
                raise ValueError(
                    f"Close price at index {index} must be finite and positive."
                )

            validated_closes.append(close)

        levels = self._validate_registry_levels(registry.levels)

        events = []
        triggered_levels = set()

        for candle_index in range(start_index, len(candles)):

            previous_close = validated_closes[candle_index - 1]
            current_close = validated_closes[candle_index]
            candle_time = times[candle_index]

            for level in levels:

                if not level.active:
                    continue

                if level.activation_index > candle_index:
                    continue

                identity = self._identity(level)

                if identity in triggered_levels:
                    continue

                level_price = float(level.price)

                if level.trend == "BULLISH":

                    crossed = (
                        previous_close >= level_price
                        and current_close < level_price
                    )

                    direction = "BEARISH"

                else:

                    crossed = (
                        previous_close <= level_price
                        and current_close > level_price
                    )

                    direction = "BULLISH"

                if not crossed:
                    continue

                event = StructureBreak(
                    event_type="CHoCH",
                    direction=direction,
                    break_time=candle_time,
                    break_index=candle_index,
                    level_price=level_price,
                    level_swing=level.swing,
                    close_price=current_close,
                    prior_trend=level.trend,
                )

                events.append(event)
                triggered_levels.add(identity)

        return events
