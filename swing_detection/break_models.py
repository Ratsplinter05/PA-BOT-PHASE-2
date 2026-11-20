from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.trend_models import TrendState


BreakDirection = Literal["BULLISH", "BEARISH"]
BreakEventType = Literal["BOS", "CHoCH"]


@dataclass(frozen=True)
class StructureBreak:
    """
    Represents a confirmed break of a market-structure level.

    event_type:
        BOS or CHoCH.

    direction:
        Direction of the structural break.

    break_time:
        Time of the candle that confirmed the break.

    break_index:
        Positional index of the break candle.

    level_price:
        Price of the structural level that was broken.

    level_swing:
        Original classified swing associated with the level.

    close_price:
        Closing price of the break candle.

    prior_trend:
        Trend state established before the break candle.
    """

    event_type: BreakEventType
    direction: BreakDirection

    break_time: datetime
    break_index: int

    level_price: float
    level_swing: ClassifiedSwing

    close_price: float
    prior_trend: TrendState
