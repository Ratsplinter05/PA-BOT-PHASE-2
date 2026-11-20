from dataclasses import dataclass
from datetime import datetime
from typing import Literal


SwingType = Literal["HIGH", "LOW"]


@dataclass(frozen=True)
class SwingPoint:
    """
    Represents a confirmed market swing.

    swing_time:
        Time of the candle where the swing actually formed.

    confirmation_time:
        Time of the candle that confirmed the swing.
        This is later than swing_time when right_bars > 0.
    """

    swing_type: SwingType
    price: float

    swing_time: datetime
    swing_index: int

    confirmation_time: datetime
    confirmation_index: int

    left_bars: int
    right_bars: int
