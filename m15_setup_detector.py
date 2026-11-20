from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Literal, Optional

import pandas as pd

from research.swing_detection.atr_calculator import WilderATR
from research.swing_detection.alignment_engine import AlignmentEngine
from research.swing_detection.market_structure_engine import (
    MarketStructureResult,
)


SetupState = Literal[
    "REJECTED",
    "WAITING_FOR_RETEST",
    "RETEST_DETECTED",
    "QUALIFIED",
    "INVALIDATED",
    "EXPIRED",
]


@dataclass(frozen=True)
class M15Setup:
    """Auditable lifecycle record for one M15 BOS setup."""

    direction: str
    state: SetupState

    break_time: datetime
    decision_time: datetime
    break_index: int

    broken_level: float
    atr_value: Optional[float]
    retest_tolerance: Optional[float]

    retest_time: Optional[datetime]
    qualification_time: Optional[datetime]
    expiry_time: Optional[datetime]

    candles_elapsed: int
    reason: str


@dataclass(frozen=True)
class M15SetupResult:
    """Collection of M15 setup lifecycle records."""

    setups: tuple[M15Setup, ...]

    @property
    def qualified_setups(self) -> tuple[M15Setup, ...]:
        return tuple(
            setup
            for setup in self.setups
            if setup.state == "QUALIFIED"
        )


class M15SetupDetector:
    """
    Detect M15 retest and rejection setups from confirmed BOS events.

    Consumes MarketStructureResult produced by MarketStructureEngine.
    Does not independently detect swings or BOS events.

    This component does not place, modify, or manage trading orders.
    """

    TIMEFRAME_DURATION = timedelta(minutes=15)

    def __init__(
        self,
        atr_period: int = 14,
        retest_tolerance_atr: float = 0.10,
        expiry_candles: int = 20,
        alignment_engine: Optional[AlignmentEngine] = None,
    ) -> None:

        if isinstance(atr_period, bool) or not isinstance(
            atr_period, int
        ):
            raise TypeError("atr_period must be an integer.")

        if atr_period < 1:
            raise ValueError("atr_period must be at least 1.")

        if (
            isinstance(retest_tolerance_atr, bool)
            or not isinstance(retest_tolerance_atr, (int, float))
        ):
            raise TypeError(
                "retest_tolerance_atr must be numeric."
            )

        if (
            not math.isfinite(float(retest_tolerance_atr))
            or retest_tolerance_atr <= 0
        ):
            raise ValueError(
                "retest_tolerance_atr must be positive and finite."
            )

        if isinstance(expiry_candles, bool) or not isinstance(
            expiry_candles, int
        ):
            raise TypeError("expiry_candles must be an integer.")

        if expiry_candles < 1:
            raise ValueError(
                "expiry_candles must be at least 1."
            )

        self.atr_period = atr_period
        self.retest_tolerance_atr = float(
            retest_tolerance_atr
        )
        self.expiry_candles = expiry_candles

        self.atr_calculator = WilderATR(period=atr_period)
        self.alignment_engine = (
            alignment_engine or AlignmentEngine()
        )

    @staticmethod
    def _to_utc_datetime(value, field_name: str) -> datetime:
        """Convert an aware timestamp to a UTC Python datetime."""

        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field_name} must be a valid timezone-aware timestamp."
            ) from exc

        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError(
                f"{field_name} must be timezone-aware."
            )

        return timestamp.tz_convert("UTC").to_pydatetime()

    @staticmethod
    def _validate_inputs(
        candles: pd.DataFrame,
        market_structure: MarketStructureResult,
    ) -> None:

        if not isinstance(candles, pd.DataFrame):
            raise TypeError(
                "candles must be a pandas DataFrame."
            )

        if not isinstance(
            market_structure,
            MarketStructureResult,
        ):
            raise TypeError(
                "market_structure must be a MarketStructureResult."
            )

        if market_structure.timeframe != "M15":
            raise ValueError(
                "market_structure must contain M15 analysis."
            )

        if market_structure.candle_count != len(candles):
            raise ValueError(
                "Candle count does not match market-structure analysis."
            )

        if "time" not in candles.columns:
            raise ValueError(
                "candles must contain a time column."
            )

    @classmethod
    def _validate_event(
        cls,
        event,
        candles: pd.DataFrame,
    ) -> datetime:
        """Validate one BOS event and return its normalized break time."""

        if event.event_type != "BOS":
            raise ValueError(
                "Only BOS events can create M15 retest setups."
            )

        if event.direction not in ("BULLISH", "BEARISH"):
            raise ValueError(
                "Unsupported BOS direction."
            )

        if (
            isinstance(event.break_index, bool)
            or not isinstance(event.break_index, int)
        ):
            raise TypeError(
                "BOS break_index must be an integer."
            )

        if not 0 <= event.break_index < len(candles):
            raise ValueError(
                "BOS break_index is outside candle data."
            )

        if (
            isinstance(event.level_price, bool)
            or not isinstance(event.level_price, (int, float))
            or not math.isfinite(float(event.level_price))
            or event.level_price <= 0
        ):
            raise ValueError(
                "BOS level_price must be positive and finite."
            )

        break_time = cls._to_utc_datetime(
            event.break_time,
            "BOS break_time",
        )

        candle_time = cls._to_utc_datetime(
            candles.iloc[event.break_index]["time"],
            "BOS candle time",
        )

        if break_time != candle_time:
            raise ValueError(
                "BOS break_time does not match its candle index."
            )

        return break_time

    def _alignment_permitted(
        self,
        h4_snapshots: tuple,
        h1_snapshots: tuple,
        assessment_time: datetime,
        direction: str,
    ) -> tuple[bool, str]:

        alignment = self.alignment_engine.evaluate(
            h4_snapshots=h4_snapshots,
            h1_snapshots=h1_snapshots,
            assessment_time=assessment_time,
        )

        if (
            alignment.h4_known_at_time is not None
            and alignment.h4_known_at_time > assessment_time
        ):
            raise RuntimeError(
                "H4 alignment lookahead detected."
            )

        if (
            alignment.h1_known_at_time is not None
            and alignment.h1_known_at_time > assessment_time
        ):
            raise RuntimeError(
                "H1 alignment lookahead detected."
            )

        permitted = (
            alignment.directional_permission
            and alignment.permitted_direction == direction
        )

        return permitted, alignment.reason

    @staticmethod
    def _build_setup(
        *,
        event,
        break_time: datetime,
        decision_time: datetime,
        state: SetupState,
        reason: str,
        atr_value: Optional[float] = None,
        tolerance: Optional[float] = None,
        retest_time: Optional[datetime] = None,
        qualification_time: Optional[datetime] = None,
        expiry_time: Optional[datetime] = None,
        candles_elapsed: int = 0,
    ) -> M15Setup:

        return M15Setup(
            direction=event.direction,
            state=state,
            break_time=break_time,
            decision_time=decision_time,
            break_index=event.break_index,
            broken_level=float(event.level_price),
            atr_value=atr_value,
            retest_tolerance=tolerance,
            retest_time=retest_time,
            qualification_time=qualification_time,
            expiry_time=expiry_time,
            candles_elapsed=candles_elapsed,
            reason=reason,
        )

    def _process_event(
        self,
        event,
        candles: pd.DataFrame,
        atr_values: pd.Series,
        h4_snapshots: tuple,
        h1_snapshots: tuple,
    ) -> M15Setup:

        break_time = self._validate_event(event, candles)

        decision_time = (
            break_time + self.TIMEFRAME_DURATION
        )

        permitted, alignment_reason = (
            self._alignment_permitted(
                h4_snapshots,
                h1_snapshots,
                decision_time,
                event.direction,
            )
        )

        if not permitted:
            return self._build_setup(
                event=event,
                break_time=break_time,
                decision_time=decision_time,
                state="REJECTED",
                reason=(
                    "BOS rejected by H4/H1 alignment: "
                    f"{alignment_reason}"
                ),
            )

        atr_value = atr_values.iloc[event.break_index]

        if (
            pd.isna(atr_value)
            or not math.isfinite(float(atr_value))
            or float(atr_value) <= 0
        ):
            return self._build_setup(
                event=event,
                break_time=break_time,
                decision_time=decision_time,
                state="REJECTED",
                reason=(
                    "ATR was unavailable or invalid at BOS confirmation."
                ),
            )

        atr_value = float(atr_value)

        # Tolerance is frozen using ATR at BOS confirmation.
        tolerance = (
            atr_value * self.retest_tolerance_atr
        )

        level = float(event.level_price)

        lower_boundary = level - tolerance
        upper_boundary = level + tolerance

        retest_time = None

        for index in range(
            event.break_index + 1,
            len(candles),
        ):

            elapsed = index - event.break_index
            candle = candles.iloc[index]

            candle_time = candle["time"].to_pydatetime()

            candle_close_time = (
                candle_time + self.TIMEFRAME_DURATION
            )

            # Reassess directional permission at every completed
            # M15 candle close. The alignment engine filters out
            # snapshots that were not yet knowable at that time.
            still_permitted, current_reason = (
                self._alignment_permitted(
                    h4_snapshots,
                    h1_snapshots,
                    candle_close_time,
                    event.direction,
                )
            )

            if not still_permitted:
                return self._build_setup(
                    event=event,
                    break_time=break_time,
                    decision_time=decision_time,
                    state="INVALIDATED",
                    reason=(
                        "H4/H1 directional permission was lost or "
                        f"reversed: {current_reason}"
                    ),
                    atr_value=atr_value,
                    tolerance=tolerance,
                    retest_time=retest_time,
                    candles_elapsed=elapsed,
                )

            close = float(candle["close"])
            high = float(candle["high"])
            low = float(candle["low"])

            # Invalidation takes priority over retest/rejection.
            if event.direction == "BULLISH":
                invalidated = close < lower_boundary
            else:
                invalidated = close > upper_boundary

            if invalidated:
                return self._build_setup(
                    event=event,
                    break_time=break_time,
                    decision_time=decision_time,
                    state="INVALIDATED",
                    reason=(
                        "Completed M15 candle closed beyond the "
                        "invalidation boundary."
                    ),
                    atr_value=atr_value,
                    tolerance=tolerance,
                    retest_time=retest_time,
                    candles_elapsed=elapsed,
                )

            if retest_time is None:

                previous_close = float(
                    candles.iloc[index - 1]["close"]
                )

                intersects_zone = (
                    low <= upper_boundary
                    and high >= lower_boundary
                )

                approached_from_correct_side = (
                    previous_close > level
                    if event.direction == "BULLISH"
                    else previous_close < level
                )

                if (
                    intersects_zone
                    and approached_from_correct_side
                ):
                    # Record the completed retest candle.
                    # It cannot also confirm rejection.
                    retest_time = candle_close_time

            else:

                rejection_confirmed = (
                    close > level
                    if event.direction == "BULLISH"
                    else close < level
                )

                if rejection_confirmed:
                    return self._build_setup(
                        event=event,
                        break_time=break_time,
                        decision_time=decision_time,
                        state="QUALIFIED",
                        reason=(
                            "Retest detected and a subsequent completed "
                            "M15 candle confirmed rejection."
                        ),
                        atr_value=atr_value,
                        tolerance=tolerance,
                        retest_time=retest_time,
                        qualification_time=candle_close_time,
                        candles_elapsed=elapsed,
                    )

            # Expiry is checked AFTER processing the candle.
            # This allows the 20th candle to qualify, but prevents
            # a retest first detected on candle 20 from remaining active.
            if elapsed == self.expiry_candles:
                return self._build_setup(
                    event=event,
                    break_time=break_time,
                    decision_time=decision_time,
                    state="EXPIRED",
                    reason=(
                        "Setup reached its configured expiry without "
                        "confirmed rejection."
                    ),
                    atr_value=atr_value,
                    tolerance=tolerance,
                    retest_time=retest_time,
                    expiry_time=candle_close_time,
                    candles_elapsed=elapsed,
                )

        if retest_time is not None:
            state: SetupState = "RETEST_DETECTED"
            reason = (
                "Retest detected. A subsequent rejection candle "
                "has not yet been confirmed."
            )
        else:
            state = "WAITING_FOR_RETEST"
            reason = (
                "Waiting for price to return to the retest zone."
            )

        candles_elapsed = min(
            len(candles) - 1 - event.break_index,
            self.expiry_candles,
        )

        return self._build_setup(
            event=event,
            break_time=break_time,
            decision_time=decision_time,
            state=state,
            reason=reason,
            atr_value=atr_value,
            tolerance=tolerance,
            retest_time=retest_time,
            candles_elapsed=candles_elapsed,
        )

    def detect(
        self,
        candles: pd.DataFrame,
        market_structure: MarketStructureResult,
        h4_snapshots,
        h1_snapshots,
    ) -> M15SetupResult:
        """
        Evaluate all confirmed M15 BOS events.

        Candles must be completed M15 candles.
        H4/H1 snapshots are evaluated using their known-at timestamps.

        No trading orders are placed or managed.
        """

        self._validate_inputs(
            candles,
            market_structure,
        )

        # Public calculation performs the ATR calculator's full
        # candle validation. Do not call its private validator.
        atr_values = self.atr_calculator.calculate(candles)

        # Normalize a separate working copy for detector operations.
        working_candles = candles.copy(deep=True)

        working_candles["time"] = pd.to_datetime(
            working_candles["time"],
            utc=True,
            errors="coerce",
        )

        for column in ("high", "low", "close"):
            working_candles[column] = pd.to_numeric(
                working_candles[column],
                errors="coerce",
            )

        # Materialize iterables once. This prevents generators from
        # being exhausted when multiple BOS events are evaluated.
        try:
            h4_snapshot_list = tuple(h4_snapshots)
            h1_snapshot_list = tuple(h1_snapshots)
        except TypeError as exc:
            raise TypeError(
                "H4 and H1 snapshots must be iterable."
            ) from exc

        ordered_events = sorted(
            market_structure.bos_events,
            key=lambda event: (
                event.break_index,
                event.direction,
            ),
        )

        setups = []

        for event in ordered_events:
            setups.append(
                self._process_event(
                    event=event,
                    candles=working_candles,
                    atr_values=atr_values,
                    h4_snapshots=h4_snapshot_list,
                    h1_snapshots=h1_snapshot_list,
                )
            )

        return M15SetupResult(
            setups=tuple(setups)
        )