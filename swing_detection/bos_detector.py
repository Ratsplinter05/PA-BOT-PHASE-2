
import math
from datetime import datetime, timedelta, timezone
from numbers import Integral
from typing import Iterable

import pandas as pd

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.trend_models import TrendSnapshot
from research.swing_detection.break_models import StructureBreak


class BOSDetector:
    """
    Detects continuation Break of Structure (BOS) events.

    Bullish BOS:
        An established bullish trend must close above the latest
        previously confirmed swing high.

    Bearish BOS:
        An established bearish trend must close below the latest
        previously confirmed swing low.

    Trend snapshots become available only after their
    confirmation candle has completed.

    The trend must be available before the BOS candle opens.

    Each structural level can generate only one BOS event.

    Event timestamps identify the opening time of the BOS candle.
    A close-based event becomes knowable after that candle completes.
    """

    REQUIRED_COLUMNS = {
        "time",
        "open",
        "high",
        "low",
        "close",
    }

    TIMEFRAME_DURATIONS = {
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
    }

    VALID_TRENDS = {
        "BULLISH",
        "BEARISH",
        "MIXED",
        "UNCONFIRMED",
    }

    VALID_CLASSIFICATIONS = {
        "HIGH": {"HH", "LH", "EQH"},
        "LOW": {"HL", "LL", "EQL"},
    }

    def _validate_timestamp(
        self,
        value: datetime,
        field_name: str,
    ) -> datetime:

        if not isinstance(value, datetime):
            raise TypeError(
                f"{field_name} must be a datetime."
            )

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{field_name} must be timezone-aware."
            )

        return value.astimezone(timezone.utc)

    def _validate_candles(
        self,
        candles: pd.DataFrame,
    ) -> tuple[list[datetime], list[dict]]:

        if not isinstance(candles, pd.DataFrame):
            raise TypeError(
                "candles must be a pandas DataFrame."
            )

        missing = self.REQUIRED_COLUMNS - set(candles.columns)

        if missing:
            raise ValueError(
                f"Missing required candle columns: {sorted(missing)}"
            )

        if candles.empty:
            raise ValueError(
                "candles cannot be empty."
            )

        if "is_forming_candle" in candles.columns:

            forming_flags = candles["is_forming_candle"]

            if forming_flags.isna().any():
                raise ValueError(
                    "Forming candle flags cannot contain missing values."
                )

            if not pd.api.types.is_bool_dtype(forming_flags.dtype):
                raise TypeError(
                    "is_forming_candle must contain boolean values."
                )

            if forming_flags.any():
                raise ValueError(
                    "Forming candles are not permitted."
                )

        candle_times = []
        candle_data = []
        previous_time = None

        for position in range(len(candles)):

            candle = candles.iloc[position]

            candle_time = self._validate_timestamp(
                candle["time"],
                f"candles.time at position {position}",
            )

            if (
                previous_time is not None
                and candle_time <= previous_time
            ):
                raise ValueError(
                    "Candle timestamps must be strictly chronological."
                )

            previous_time = candle_time
            candle_times.append(candle_time)

            prices = {}

            for column in ("open", "high", "low", "close"):

                try:
                    price = float(candle[column])
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        f"Candle {column} prices must be numeric."
                    ) from None

                if not math.isfinite(price) or price <= 0:
                    raise ValueError(
                        f"Candle {column} prices must be finite and positive."
                    )

                prices[column] = price

            open_price = prices["open"]
            high_price = prices["high"]
            low_price = prices["low"]
            close_price = prices["close"]

            if high_price < low_price:
                raise ValueError(
                    "Candle high cannot be lower than candle low."
                )

            if high_price < max(open_price, close_price):
                raise ValueError(
                    "Candle high must contain the open and close prices."
                )

            if low_price > min(open_price, close_price):
                raise ValueError(
                    "Candle low must contain the open and close prices."
                )

            candle_data.append(prices)

        return candle_times, candle_data

    def _validate_swings(
        self,
        swings: Iterable[ClassifiedSwing],
        candle_times: list[datetime],
    ) -> list[ClassifiedSwing]:

        records = list(swings)

        seen_swing_ids = set()

        for item in records:

            if not isinstance(item, ClassifiedSwing):
                raise TypeError(
                    "All swings must be ClassifiedSwing instances."
                )

            swing = item.swing

            if not isinstance(swing, SwingPoint):
                raise TypeError(
                    "Classified swings must contain SwingPoint instances."
                )

            if swing.swing_type not in ("HIGH", "LOW"):
                raise ValueError(
                    "Invalid swing type."
                )

            try:
                price = float(swing.price)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "Swing prices must be numeric."
                ) from None

            if not math.isfinite(price):
                raise ValueError(
                    "Swing prices must be finite."
                )

            if price <= 0:
                raise ValueError(
                    "Swing prices must be positive."
                )

            if (
                not isinstance(swing.swing_index, Integral)
                or isinstance(swing.swing_index, bool)
            ):
                raise TypeError(
                    "Swing index must be an integer."
                )

            if (
                not isinstance(swing.confirmation_index, Integral)
                or isinstance(swing.confirmation_index, bool)
            ):
                raise TypeError(
                    "Swing confirmation index must be an integer."
                )

            if not (
                0 <= swing.swing_index
                < swing.confirmation_index
                < len(candle_times)
            ):
                raise ValueError(
                    "Swing and confirmation indices must be valid "
                    "and confirmation must follow the swing."
                )

            self._validate_timestamp(
                swing.swing_time,
                "swing.swing_time",
            )

            self._validate_timestamp(
                swing.confirmation_time,
                "swing.confirmation_time",
            )

            if (
                swing.swing_time.astimezone(timezone.utc)
                != candle_times[swing.swing_index]
            ):
                raise ValueError(
                    "Swing timestamp does not match its candle index."
                )

            if (
                swing.confirmation_time.astimezone(timezone.utc)
                != candle_times[swing.confirmation_index]
            ):
                raise ValueError(
                    "Swing confirmation timestamp does not match "
                    "its candle index."
                )

            if (
                not isinstance(swing.left_bars, Integral)
                or isinstance(swing.left_bars, bool)
                or swing.left_bars < 1
            ):
                raise ValueError(
                    "Swing left_bars must be a positive integer."
                )

            if (
                not isinstance(swing.right_bars, Integral)
                or isinstance(swing.right_bars, bool)
                or swing.right_bars < 1
            ):
                raise ValueError(
                    "Swing right_bars must be a positive integer."
                )

            if item.classification is not None:

                if item.classification not in self.VALID_CLASSIFICATIONS[
                    swing.swing_type
                ]:
                    raise ValueError(
                        f"Invalid classification "
                        f"'{item.classification}' for "
                        f"{swing.swing_type} swing."
                    )

            swing_id = (
                swing.swing_type,
                swing.swing_index,
                swing.confirmation_index,
            )

            if swing_id in seen_swing_ids:
                raise ValueError(
                    f"Duplicate swing record: {swing_id}"
                )

            seen_swing_ids.add(swing_id)

        records.sort(
            key=lambda item: (
                item.swing.confirmation_index,
                item.swing.swing_index,
                item.swing.swing_type,
            )
        )

        return records

    def _validate_trend_snapshots(
        self,
        trend_snapshots: Iterable[TrendSnapshot],
    ) -> list[TrendSnapshot]:

        snapshots = list(trend_snapshots)

        seen_timestamps = set()

        for snapshot in snapshots:

            if not isinstance(snapshot, TrendSnapshot):
                raise TypeError(
                    "All trend snapshots must be TrendSnapshot instances."
                )

            if (
                not isinstance(snapshot.trend, str)
                or snapshot.trend not in self.VALID_TRENDS
            ):
                raise ValueError(
                    f"Invalid trend state: {snapshot.trend}"
                )

            normalized_time = self._validate_timestamp(
                snapshot.as_of_time,
                "Trend snapshot timestamp",
            )

            if normalized_time in seen_timestamps:
                raise ValueError(
                    "Duplicate trend snapshot timestamp: "
                    f"{normalized_time.isoformat()}"
                )

            seen_timestamps.add(normalized_time)

        snapshots.sort(
            key=lambda item: item.as_of_time.astimezone(timezone.utc)
        )

        return snapshots

    def detect(
        self,
        candles: pd.DataFrame,
        swings: Iterable[ClassifiedSwing],
        trend_snapshots: Iterable[TrendSnapshot],
        timeframe: str = "H1",
    ) -> list[StructureBreak]:

        if timeframe not in self.TIMEFRAME_DURATIONS:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported values: "
                f"{sorted(self.TIMEFRAME_DURATIONS)}"
            )

        candle_times, candle_data = self._validate_candles(
            candles
        )

        records = self._validate_swings(
            swings,
            candle_times,
        )

        snapshots = self._validate_trend_snapshots(
            trend_snapshots
        )

        timeframe_duration = self.TIMEFRAME_DURATIONS[timeframe]

        latest_high = None
        latest_low = None
        current_trend = None

        swing_pointer = 0
        trend_pointer = 0

        broken_levels = set()
        events = []

        for candle_index in range(len(candle_data)):

            candle_time = candle_times[candle_index]
            close_price = candle_data[candle_index]["close"]

            # Activate only swings confirmed on an earlier candle.
            while (
                swing_pointer < len(records)
                and records[swing_pointer].swing.confirmation_index
                < candle_index
            ):

                item = records[swing_pointer]

                if item.swing.swing_type == "HIGH":
                    latest_high = item
                else:
                    latest_low = item

                swing_pointer += 1

            # A trend snapshot becomes available only after its
            # confirmation candle has completed.
            #
            # It must be available before the current BOS candle opens.

            while trend_pointer < len(snapshots):

                snapshot = snapshots[trend_pointer]

                known_at_time = (
                    snapshot.as_of_time.astimezone(timezone.utc)
                    + timeframe_duration
                )

                if known_at_time > candle_time:
                    break

                current_trend = snapshot.trend
                trend_pointer += 1

            # A structural break requires a close crossing the level.
            # Wick penetration alone cannot generate a BOS event.

            if candle_index > 0:

                previous_close = candle_data[candle_index - 1]["close"]

                # Bullish continuation BOS.
                if (
                    current_trend == "BULLISH"
                    and latest_high is not None
                ):

                    level_key = (
                        "HIGH",
                        latest_high.swing.swing_index,
                        latest_high.swing.confirmation_index,
                    )

                    crossed_level = (
                        previous_close <= latest_high.swing.price
                        and close_price > latest_high.swing.price
                    )

                    if (
                        level_key not in broken_levels
                        and crossed_level
                    ):

                        events.append(
                            StructureBreak(
                                event_type="BOS",
                                direction="BULLISH",
                                break_time=candle_times[candle_index],
                                break_index=candle_index,
                                level_price=latest_high.swing.price,
                                level_swing=latest_high,
                                close_price=close_price,
                                prior_trend=current_trend,
                            )
                        )

                        broken_levels.add(level_key)

                # Bearish continuation BOS.
                elif (
                    current_trend == "BEARISH"
                    and latest_low is not None
                ):

                    level_key = (
                        "LOW",
                        latest_low.swing.swing_index,
                        latest_low.swing.confirmation_index,
                    )

                    crossed_level = (
                        previous_close >= latest_low.swing.price
                        and close_price < latest_low.swing.price
                    )

                    if (
                        level_key not in broken_levels
                        and crossed_level
                    ):

                        events.append(
                            StructureBreak(
                                event_type="BOS",
                                direction="BEARISH",
                                break_time=candle_times[candle_index],
                                break_index=candle_index,
                                level_price=latest_low.swing.price,
                                level_swing=latest_low,
                                close_price=close_price,
                                prior_trend=current_trend,
                            )
                        )

                        broken_levels.add(level_key)

        return events
