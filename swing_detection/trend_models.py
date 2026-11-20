from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from research.swing_detection.structure_models import ClassifiedSwing

TrendState = Literal[
    "BULLISH",
    "BEARISH",
    "MIXED",
    "UNCONFIRMED",
]


@dataclass(frozen=True)
class TrendSnapshot:
    as_of_time: datetime
    trend: TrendState

    latest_high: Optional[ClassifiedSwing]
    latest_low: Optional[ClassifiedSwing]

    reason: str
