
import math
from collections.abc import Iterable
from numbers import Integral, Real
from datetime import datetime

from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.break_models import StructureBreak
from research.swing_detection.registry_models import ProtectedLevel


class ProtectedLevelRegistry:
    """
    Maintains unique protected structural levels and their lifecycle.

    Unique identity:
        trend
        swing type
        swing index
        confirmation index

    Repeated BOS events are attached to the existing level.
    The first activation index is preserved.

    A level can only be retired by a valid opposite-direction CHoCH
    that references the exact protected swing and price.

    Retired levels remain in the historical registry.
    """

    def __init__(self):
        self._levels = {}

    @staticmethod
    def _identity(protected_swing: ProtectedSwing) -> tuple:
        swing = protected_swing.swing.swing

        return (
            protected_swing.trend,
            swing.swing_type,
            swing.swing_index,
            swing.confirmation_index,
        )

    @property
    def levels(self) -> tuple:
        return tuple(self._levels.values())

    def get_active_levels(self, as_of_index=None) -> list:
        """
        Return active levels.

        If as_of_index is supplied, only return levels whose
        activation index has been reached.
        """

        if as_of_index is not None:
            if (
                not isinstance(as_of_index, Integral)
                or isinstance(as_of_index, bool)
                or as_of_index < 0
            ):
                raise ValueError(
                    "as_of_index must be a non-negative integer."
                )

        return [
            level
            for level in self._levels.values()
            if (
                level.active
                and (
                    as_of_index is None
                    or level.activation_index <= as_of_index
                )
            )
        ]

    def get_retired_levels(self) -> list:
        return [
            level
            for level in self._levels.values()
            if not level.active
        ]

    def register(
        self,
        protected_swings: Iterable[ProtectedSwing],
    ) -> list:
        """
        Register protected swings and attach repeated BOS events.

        Returns the registry records created or updated.
        """

        registered = []

        for protected in protected_swings:

            if not isinstance(protected, ProtectedSwing):
                raise TypeError(
                    "Every item must be a ProtectedSwing."
                )

            event = protected.established_by

            if not isinstance(event, StructureBreak):
                raise TypeError(
                    "Protected swing must reference a StructureBreak."
                )

            if event.event_type != "BOS":
                raise ValueError(
                    "Protected levels must be established by BOS events."
                )

            if protected.trend not in ("BULLISH", "BEARISH"):
                raise ValueError(
                    "Protected swing has an invalid trend."
                )

            if event.direction != protected.trend:
                raise ValueError(
                    "Protected swing trend must match BOS direction."
                )

            if (
                not isinstance(protected.activation_index, Integral)
                or isinstance(protected.activation_index, bool)
                or protected.activation_index < 0
            ):
                raise ValueError(
                    "Activation index must be a non-negative integer."
                )

            if (
                not isinstance(event.break_index, Integral)
                or isinstance(event.break_index, bool)
                or event.break_index < 0
            ):
                raise ValueError(
                    "BOS break index must be a non-negative integer."
                )

            if protected.activation_index != event.break_index + 1:
                raise ValueError(
                    "Activation index must be one candle after the BOS."
                )

            identity = self._identity(protected)

            existing = self._levels.get(identity)

            if existing is None:

                level = ProtectedLevel(
                    protected_swing=protected,
                    bos_events=(event,),
                )

                self._levels[identity] = level
                registered.append(level)
                continue

            if existing.trend != protected.trend:
                raise ValueError(
                    "A structural swing cannot change protected trend."
                )

            if not existing.active:
                raise ValueError(
                    "A retired protected level cannot be reactivated."
                )

            if event not in existing.bos_events:

                updated = ProtectedLevel(
                    protected_swing=existing.protected_swing,
                    bos_events=existing.bos_events + (event,),
                    active=True,
                    retired_by=None,
                )

                self._levels[identity] = updated
                registered.append(updated)

            else:
                registered.append(existing)

        return registered

    def retire(
        self,
        protected_level: ProtectedLevel,
        choch_event: StructureBreak,
    ) -> ProtectedLevel:
        """
        Retire an active protected level using a valid opposite CHoCH.

        The CHoCH must reference the exact protected swing and price
        represented by the registered level.
        """

        if not isinstance(protected_level, ProtectedLevel):
            raise TypeError(
                "protected_level must be a ProtectedLevel."
            )

        if not isinstance(choch_event, StructureBreak):
            raise TypeError(
                "choch_event must be a StructureBreak."
            )

        if choch_event.event_type != "CHoCH":
            raise ValueError(
                "Only CHoCH events can retire protected levels."
            )

        if choch_event.direction not in ("BULLISH", "BEARISH"):
            raise ValueError("Invalid CHoCH direction.")

        if (
            not isinstance(choch_event.break_index, Integral)
            or isinstance(choch_event.break_index, bool)
            or choch_event.break_index < 0
        ):
            raise ValueError(
                "CHoCH break index must be a non-negative integer."
            )

        if (
            not isinstance(choch_event.break_time, datetime)
            or choch_event.break_time.tzinfo is None
            or choch_event.break_time.utcoffset() is None
        ):
            raise ValueError(
                "CHoCH break time must be timezone-aware."
            )

        if (
            not isinstance(choch_event.level_price, Real)
            or isinstance(choch_event.level_price, bool)
            or not math.isfinite(choch_event.level_price)
            or choch_event.level_price <= 0
        ):
            raise ValueError(
                "CHoCH level price must be finite and positive."
            )

        if (
            not isinstance(choch_event.close_price, Real)
            or isinstance(choch_event.close_price, bool)
            or not math.isfinite(choch_event.close_price)
            or choch_event.close_price <= 0
        ):
            raise ValueError(
                "CHoCH close price must be finite and positive."
            )

        expected_direction = (
            "BEARISH"
            if protected_level.trend == "BULLISH"
            else "BULLISH"
        )

        if protected_level.trend not in ("BULLISH", "BEARISH"):
            raise ValueError(
                "Protected level has an invalid trend."
            )

        if choch_event.direction != expected_direction:
            raise ValueError(
                "CHoCH direction must oppose the protected trend."
            )

        identity = self._identity(
            protected_level.protected_swing
        )

        existing = self._levels.get(identity)

        if existing is None:
            raise ValueError(
                "Protected level is not registered."
            )

        if not existing.active:
            raise ValueError(
                "Protected level has already been retired."
            )

        if choch_event.prior_trend != existing.trend:
            raise ValueError(
                "CHoCH prior trend must match the protected trend."
            )

        if choch_event.level_swing != existing.swing:
            raise ValueError(
                "CHoCH must reference the exact protected swing."
            )

        if not math.isclose(
            float(choch_event.level_price),
            float(existing.price),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "CHoCH level price must match the protected level price."
            )

        if existing.trend == "BULLISH":
            if choch_event.close_price >= existing.price:
                raise ValueError(
                    "Bearish CHoCH close must be below the protected level."
                )
        else:
            if choch_event.close_price <= existing.price:
                raise ValueError(
                    "Bullish CHoCH close must be above the protected level."
                )

        if choch_event.break_index < existing.activation_index:
            raise ValueError(
                "CHoCH cannot retire a level before it becomes active."
            )

        retired = ProtectedLevel(
            protected_swing=existing.protected_swing,
            bos_events=existing.bos_events,
            active=False,
            retired_by=choch_event,
        )

        self._levels[identity] = retired

        return retired
