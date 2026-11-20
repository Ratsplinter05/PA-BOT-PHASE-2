from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from research.swing_detection.trend_models import TrendSnapshot


AlignmentState = Literal[
    "ALIGNED_BULLISH",
    "ALIGNED_BEARISH",
    "CONFLICT",
    "H4_MIXED",
    "H4_UNCONFIRMED",
    "H1_MIXED",
    "H1_UNCONFIRMED",
    "H4_MISSING",
    "H1_MISSING",
]


@dataclass(frozen=True)
class AlignmentResult:
    assessment_time: datetime

    alignment_state: AlignmentState
    directional_permission: bool
    permitted_direction: Optional[Literal["BULLISH", "BEARISH"]]

    h4_snapshot: Optional[TrendSnapshot]
    h1_snapshot: Optional[TrendSnapshot]

    h4_known_at_time: Optional[datetime]
    h1_known_at_time: Optional[datetime]

    reason: str
