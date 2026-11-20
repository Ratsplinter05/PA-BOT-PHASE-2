
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Literal, Optional

import pandas as pd

from research.swing_detection.break_models import StructureBreak
from research.swing_detection.market_structure_engine import (
    MarketStructureResult,
)
from strategy.m15_setup_detector import (
    M15Setup,
    M15SetupResult,
)


M5ConfirmationState = Literal[
    "WAITING",
    "CONFIRMED",
    "EXPIRED",
    "REJECTED",
]


@dataclass(frozen=True)
class M5Confirmation:
    setup_id: str
    direction: str
    state: M5ConfirmationState
    qualification_time: Optional[datetime]
    confirmation_time: Optional[datetime]
    confirmation_break_time: Optional[datetime]
    confirmation_break_index: Optional[int]
    confirmation_event: Optional[StructureBreak]
    candles_elapsed: int
    reason: str


@dataclass(frozen=True)
class M5ConfirmationResult:
    confirmations: tuple[M5Confirmation, ...]

    @property
    def confirmed(self) -> tuple[M5Confirmation, ...]:
        return tuple(
            item for item in self.confirmations
            if item.state == "CONFIRMED"
        )

    @property
    def waiting(self) -> tuple[M5Confirmation, ...]:
        return tuple(
            item for item in self.confirmations
            if item.state == "WAITING"
        )

    @property
    def expired(self) -> tuple[M5Confirmation, ...]:
        return tuple(
            item for item in self.confirmations
            if item.state == "EXPIRED"
        )

    @property
    def rejected(self) -> tuple[M5Confirmation, ...]:
        return tuple(
            item for item in self.confirmations
            if item.state == "REJECTED"
        )


class M5ConfirmationDetector:
    """
    Confirms qualified M15 setups using existing M5 BOS events.

    The detector:
    - Accepts only QUALIFIED M15 setups.
    - Requires an M5 BOS in the same direction.
    - Ignores M5 BOS events before M15 qualification.
    - Uses a maximum window of 12 eligible M5 candles.
    - Selects the earliest matching BOS.
    - Does not calculate BOS independently.
    - Does not place or manage trades.

    All supplied M5 candles must be completed candles.
    """

    TIMEFRAME_DURATION = timedelta(minutes=5)

    def __init__(self, confirmation_window_candles: int = 12):
        if (
            not isinstance(confirmation_window_candles, int)
            or isinstance(confirmation_window_candles, bool)
            or confirmation_window_candles < 1
        ):
            raise ValueError(
                "confirmation_window_candles must be a positive integer."
            )

        self.confirmation_window_candles = confirmation_window_candles

    @staticmethod
    def _to_utc_datetime(value, field_name: str) -> datetime:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            raise ValueError(f"{field_name} cannot be NaT.")

        if timestamp.tzinfo is None:
            raise ValueError(
                f"{field_name} must be timezone-aware."
            )

        return timestamp.to_pydatetime().astimezone(timezone.utc)

    @classmethod
    def _setup_identity(
        cls,
        setup,
        qualification_time: Optional[datetime],
    ) -> str:
        break_time = cls._to_utc_datetime(
            setup.break_time,
            "M15 setup break_time",
        )

        identity_parts = (
            str(setup.direction),
            break_time.isoformat(),
            str(setup.break_index),
            str(float(setup.broken_level)),
            (
                qualification_time.isoformat()
                if qualification_time is not None
                else "NOT_QUALIFIED"
            ),
        )

        raw_identity = "|".join(identity_parts)
        return hashlib.sha256(
            raw_identity.encode("utf-8")
        ).hexdigest()[:20]

    @staticmethod
    def _validate_inputs(
        m15_setups: M15SetupResult,
        m5_candles: pd.DataFrame,
        m5_market_structure: MarketStructureResult,
    ) -> None:
        if not hasattr(m15_setups, "setups"):
            raise TypeError(
                "m15_setups must provide a setups collection."
            )

        if not isinstance(m5_candles, pd.DataFrame):
            raise TypeError("m5_candles must be a pandas DataFrame.")

        if "time" not in m5_candles.columns:
            raise ValueError(
                "m5_candles must contain a 'time' column."
            )

        if not hasattr(m5_market_structure, "bos_events"):
            raise TypeError(
                "m5_market_structure must provide bos_events."
            )

        timeframe = str(
            getattr(m5_market_structure, "timeframe", "")
        ).upper()

        if timeframe not in {"M5", "5M"} and not timeframe.endswith(".M5"):
            raise ValueError(
                "m5_market_structure must represent the M5 timeframe."
            )

        if len(m5_candles) != getattr(
            m5_market_structure, "candle_count", len(m5_candles)
        ):
            raise ValueError(
                "M5 candle count does not match market structure candle_count."
            )

        candle_times = [
            M5ConfirmationDetector._to_utc_datetime(
                value,
                f"M5 candle time at index {index}",
            )
            for index, value in enumerate(m5_candles["time"])
        ]

        if any(
            current <= previous
            for previous, current in zip(
                candle_times,
                candle_times[1:],
            )
        ):
            raise ValueError(
                "M5 candle timestamps must be strictly increasing."
            )

        if len(set(candle_times)) != len(candle_times):
            raise ValueError("Duplicate M5 candle timestamps detected.")

        for previous, current in zip(
            candle_times,
            candle_times[1:],
        ):
            if current - previous != timedelta(minutes=5):
                raise ValueError(
                    "M5 candle timestamps must be spaced exactly 5 minutes apart."
                )

    @staticmethod
    def _validate_bos_events(
        bos_events,
        candle_times: list[datetime],
    ) -> None:
        for event in bos_events:
            if event.event_type != "BOS":
                raise ValueError(
                    "m5_market_structure.bos_events must contain BOS events only."
                )

            if event.direction not in {"BULLISH", "BEARISH"}:
                raise ValueError(
                    "BOS direction must be BULLISH or BEARISH."
                )

            if (
                not isinstance(event.break_index, int)
                or isinstance(event.break_index, bool)
                or not 0 <= event.break_index < len(candle_times)
            ):
                raise ValueError(
                    "BOS break_index is outside the M5 candle range."
                )

            event_time = M5ConfirmationDetector._to_utc_datetime(
                event.break_time,
                "BOS break_time",
            )

            expected_time = candle_times[event.break_index]

            if event_time != expected_time:
                raise ValueError(
                    "BOS break_time must match the opening time of its indexed M5 candle."
                )

    @staticmethod
    def _rejected(
        setup_id: str,
        direction: str,
        reason: str,
        qualification_time: Optional[datetime] = None,
    ) -> M5Confirmation:
        return M5Confirmation(
            setup_id=setup_id,
            direction=direction,
            state="REJECTED",
            qualification_time=qualification_time,
            confirmation_time=None,
            confirmation_break_time=None,
            confirmation_break_index=None,
            confirmation_event=None,
            candles_elapsed=0,
            reason=reason,
        )

    def _evaluate_setup(
        self,
        setup,
        candle_times: list[datetime],
        bos_events,
    ) -> M5Confirmation:
        direction = str(setup.direction)

        if direction not in {"BULLISH", "BEARISH"}:
            raise ValueError(
                f"Unsupported M15 setup direction: {direction}"
            )

        state = str(setup.state)

        if state != "QUALIFIED":
            setup_id = self._setup_identity(setup, None)

            return self._rejected(
                setup_id=setup_id,
                direction=direction,
                reason=(
                    f"M15 setup state is {state}; "
                    "only QUALIFIED setups can receive M5 confirmation."
                ),
            )

        if setup.qualification_time is None:
            raise ValueError(
                "A QUALIFIED M15 setup must have qualification_time."
            )

        qualification_time = self._to_utc_datetime(
            setup.qualification_time,
            "M15 qualification_time",
        )

        setup_id = self._setup_identity(
            setup,
            qualification_time,
        )

        eligible_indices = [
            index
            for index, candle_time in enumerate(candle_times)
            if candle_time >= qualification_time
        ]

        window_indices = eligible_indices[
            :self.confirmation_window_candles
        ]

        matching_events = [
            event
            for event in bos_events
            if (
                event.direction == direction
                and event.break_index in window_indices
            )
        ]

        matching_events.sort(key=lambda event: event.break_index)

        if matching_events:
            event = matching_events[0]

            return M5Confirmation(
                setup_id=setup_id,
                direction=direction,
                state="CONFIRMED",
                qualification_time=qualification_time,
                confirmation_time=(
                    candle_times[event.break_index]
                    + self.TIMEFRAME_DURATION
                ),
                confirmation_break_time=self._to_utc_datetime(
                    event.break_time,
                    "BOS break_time",
                ),
                confirmation_break_index=event.break_index,
                confirmation_event=event,
                candles_elapsed=event.break_index - window_indices[0] + 1,
                reason=(
                    f"{direction} M5 BOS confirmed the qualified "
                    "M15 setup within the confirmation window."
                ),
            )

        candles_elapsed = len(window_indices)

        if candles_elapsed >= self.confirmation_window_candles:
            return M5Confirmation(
                setup_id=setup_id,
                direction=direction,
                state="EXPIRED",
                qualification_time=qualification_time,
                confirmation_time=None,
                confirmation_break_time=None,
                confirmation_break_index=None,
                confirmation_event=None,
                candles_elapsed=candles_elapsed,
                reason=(
                    "No matching M5 BOS occurred within "
                    f"{self.confirmation_window_candles} eligible candles."
                ),
            )

        return M5Confirmation(
            setup_id=setup_id,
            direction=direction,
            state="WAITING",
            qualification_time=qualification_time,
            confirmation_time=None,
            confirmation_break_time=None,
            confirmation_break_index=None,
            confirmation_event=None,
            candles_elapsed=candles_elapsed,
            reason=(
                "Waiting for a matching M5 BOS. "
                f"{candles_elapsed}/{self.confirmation_window_candles} "
                "eligible candles have been evaluated."
            ),
        )

    def detect(
        self,
        m15_setups: M15SetupResult,
        m5_candles: pd.DataFrame,
        m5_market_structure: MarketStructureResult,
    ) -> M5ConfirmationResult:
        self._validate_inputs(
            m15_setups,
            m5_candles,
            m5_market_structure,
        )

        candle_times = [
            self._to_utc_datetime(
                value,
                f"M5 candle time at index {index}",
            )
            for index, value in enumerate(m5_candles["time"])
        ]

        bos_events = tuple(m5_market_structure.bos_events)

        self._validate_bos_events(
            bos_events,
            candle_times,
        )

        confirmations = tuple(
            self._evaluate_setup(
                setup,
                candle_times,
                bos_events,
            )
            for setup in m15_setups.setups
        )

        return M5ConfirmationResult(
            confirmations=confirmations,
        )

