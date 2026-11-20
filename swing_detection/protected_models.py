from dataclasses import dataclass
from typing import Literal

from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.break_models import StructureBreak


ProtectedTrend = Literal["BULLISH", "BEARISH"]


@dataclass(frozen=True)
class ProtectedSwing:
    """
    Represents a protected structural swing established by a BOS.

    trend:
        The trend direction supported by this protected swing.

    swing:
        The classified HL or LH that becomes protected.

    established_by:
        The BOS event that established the protected level.

    activation_index:
        The first candle index on which this level is available
        for downstream analysis. This is the candle after the
        BOS candle has closed.
    """

    trend: ProtectedTrend
    swing: ClassifiedSwing
    established_by: StructureBreak
    activation_index: int

    @property
    def price(self) -> float:
        return self.swing.swing.price
