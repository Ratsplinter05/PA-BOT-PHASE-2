import math
from numbers import Real
from typing import Iterable

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing


class StructureClassifier:
    """
    Classifies confirmed swing points into market-structure labels.

    HIGH comparisons:
        Higher price = HH
        Lower price = LH
        Equal within tolerance = EQH

    LOW comparisons:
        Higher price = HL
        Lower price = LL
        Equal within tolerance = EQL

    Each swing is compared only with the previous confirmed swing
    of the same type.

    Swings are processed in confirmation-time order.
    """

    def __init__(self, price_tolerance: float = 0.0):
        if (
            isinstance(price_tolerance, bool)
            or not isinstance(price_tolerance, Real)
        ):
            raise TypeError("price_tolerance must be numeric.")

        if not math.isfinite(float(price_tolerance)):
            raise ValueError("price_tolerance must be finite.")

        if price_tolerance < 0:
            raise ValueError("price_tolerance cannot be negative.")

        self.price_tolerance = float(price_tolerance)

    def classify(
        self,
        swings: Iterable[SwingPoint],
    ) -> list[ClassifiedSwing]:
        """
        Classify confirmed swings chronologically.

        The original SwingPoint objects are preserved.
        """

        swings = list(swings)

        for swing in swings:
            if not isinstance(swing, SwingPoint):
                raise TypeError(
                    "All items must be SwingPoint instances."
                )

            if swing.swing_type not in ("HIGH", "LOW"):
                raise ValueError("Invalid swing type.")

            if (
                not math.isfinite(float(swing.price))
                or swing.price <= 0
            ):
                raise ValueError("Swing prices must be finite and positive.")

            if swing.confirmation_time < swing.swing_time:
                raise ValueError(
                    "Confirmation time cannot precede swing time."
                )

        # Process swings by the time they became knowable.
        ordered_swings = sorted(
            swings,
            key=lambda swing: (
                swing.confirmation_time,
                swing.confirmation_index,
                swing.swing_index,
                swing.swing_type,
            ),
        )

        previous_swings = {
            "HIGH": None,
            "LOW": None,
        }

        classified = []

        for swing in ordered_swings:

            previous = previous_swings[swing.swing_type]

            if previous is None:
                classification = None
                previous_price = None
                previous_swing_time = None

            else:
                difference = swing.price - previous.price

                if abs(difference) <= self.price_tolerance:
                    classification = (
                        "EQH"
                        if swing.swing_type == "HIGH"
                        else "EQL"
                    )

                elif swing.swing_type == "HIGH":
                    classification = (
                        "HH" if difference > 0 else "LH"
                    )

                else:
                    classification = (
                        "HL" if difference > 0 else "LL"
                    )

                previous_price = previous.price
                previous_swing_time = previous.swing_time

            classified.append(
                ClassifiedSwing(
                    swing=swing,
                    classification=classification,
                    previous_price=previous_price,
                    previous_swing_time=previous_swing_time,
                )
            )

            # Every confirmed swing becomes the next comparison reference,
            # including equal highs and equal lows.
            previous_swings[swing.swing_type] = swing

        return classified
