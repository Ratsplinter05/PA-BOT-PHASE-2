from dataclasses import dataclass
from typing import Literal, Optional

from research.swing_detection.swing_models import SwingPoint


StructureLabel = Literal[
    "HH",
    "HL",
    "LH",
    "LL",
    "EQH",
    "EQL",
]


@dataclass(frozen=True)
class ClassifiedSwing:
    """
    A confirmed swing with its market-structure classification.

    swing:
        Original confirmed swing point.

    classification:
        HH, HL, LH, LL, EQH, EQL, or None for the first
        swing of its respective type.

    previous_price:
        Price of the previous confirmed swing of the same type.

    previous_swing_time:
        Pivot time of the previous confirmed swing of the same type.
    """

    swing: SwingPoint
    classification: Optional[StructureLabel]
    previous_price: Optional[float]
    previous_swing_time: Optional[object]
