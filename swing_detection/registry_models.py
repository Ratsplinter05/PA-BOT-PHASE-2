from dataclasses import dataclass
from typing import Optional, Tuple

from research.swing_detection.protected_models import ProtectedSwing
from research.swing_detection.break_models import StructureBreak


@dataclass(frozen=True)
class ProtectedLevel:
    """
    Registry record for one unique protected structural swing.

    Repeated BOS events are attached to the same record.
    The original activation index is never reset.
    """

    protected_swing: ProtectedSwing
    bos_events: Tuple[StructureBreak, ...]
    active: bool = True
    retired_by: Optional[StructureBreak] = None

    @property
    def trend(self) -> str:
        return self.protected_swing.trend

    @property
    def swing(self):
        return self.protected_swing.swing

    @property
    def price(self) -> float:
        return self.protected_swing.price

    @property
    def activation_index(self) -> int:
        return self.protected_swing.activation_index

    @property
    def established_by(self) -> StructureBreak:
        return self.protected_swing.established_by
