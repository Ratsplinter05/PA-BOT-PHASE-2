from collections.abc import Iterable

from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.break_models import StructureBreak
from research.swing_detection.protected_models import ProtectedSwing


class ProtectedSwingTracker:
    """
    Identifies protected swings established by continuation BOS events.

    Bullish BOS:
        Protects the most recent confirmed Higher Low (HL)
        preceding the BOS candle.

    Bearish BOS:
        Protects the most recent confirmed Lower High (LH)
        preceding the BOS candle.

    A protected level becomes active only after the BOS candle
    has completed. Therefore activation_index is break_index + 1.
    """

    def track(
        self,
        swings: Iterable[ClassifiedSwing],
        bos_events: Iterable[StructureBreak],
    ) -> list[ProtectedSwing]:

        swing_list = list(swings)
        event_list = list(bos_events)

        for swing in swing_list:
            if not isinstance(swing, ClassifiedSwing):
                raise TypeError(
                    "Every swing must be a ClassifiedSwing."
                )

        for event in event_list:
            if not isinstance(event, StructureBreak):
                raise TypeError(
                    "Every BOS event must be a StructureBreak."
                )

            if event.event_type != "BOS":
                raise ValueError(
                    "ProtectedSwingTracker accepts BOS events only."
                )

            if event.direction not in ("BULLISH", "BEARISH"):
                raise ValueError("Invalid BOS direction.")

            if event.prior_trend != event.direction:
                raise ValueError(
                    "BOS direction must match its prior trend."
                )

            if (
                not isinstance(event.break_index, int)
                or isinstance(event.break_index, bool)
                or event.break_index < 0
            ):
                raise ValueError(
                    "BOS break_index must be a non-negative integer."
                )

        ordered_events = sorted(
            event_list,
            key=lambda event: (
                event.break_index,
                event.break_time,
                event.direction,
            ),
        )

        protected_history = []

        for event in ordered_events:

            if event.direction == "BULLISH":
                required_type = "LOW"
                required_classification = "HL"
                protected_trend = "BULLISH"

            else:
                required_type = "HIGH"
                required_classification = "LH"
                protected_trend = "BEARISH"

            eligible_swings = [
                swing
                for swing in swing_list
                if (
                    swing.swing.swing_type == required_type
                    and swing.classification == required_classification
                    and swing.swing.confirmation_index < event.break_index
                )
            ]

            if not eligible_swings:
                raise ValueError(
                    f"No eligible {required_classification} swing "
                    f"exists before {event.direction} BOS at index "
                    f"{event.break_index}."
                )

            latest_swing = max(
                eligible_swings,
                key=lambda swing: (
                    swing.swing.confirmation_index,
                    swing.swing.swing_index,
                ),
            )

            protected_history.append(
                ProtectedSwing(
                    trend=protected_trend,
                    swing=latest_swing,
                    established_by=event,
                    activation_index=event.break_index + 1,
                )
            )

        return protected_history
