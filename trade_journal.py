"""
Trade journal.

Appends one row per event to a CSV file so every confirmed setup
and risk decision is permanently recorded.

Events logged:
  SETUP_CONFIRMED   - M5 confirmation received
  RISK_APPROVED     - risk manager approved the trade
  RISK_BLOCKED      - risk manager blocked the trade
  TRADE_OPENED      - order placed (Phase 3)
  TRADE_CLOSED      - position closed (Phase 3)

This module does not place, modify, or manage trading orders.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.logger import get_logger

log = get_logger("TRADE_JOURNAL")

DEFAULT_JOURNAL_PATH = Path("logs/trade_journal.csv")

FIELDNAMES = [
    "event_time",
    "event_type",
    "instrument",
    "direction",
    "setup_id",
    "entry_price",
    "stop_loss",
    "take_profit",
    "sl_distance",
    "lot_size",
    "risk_amount",
    "rr_ratio",
    "account_balance",
    "daily_loss_pct",
    "open_trade_count",
    "reason",
]


class TradeJournal:
    """
    Appends structured event rows to a CSV journal file.

    The file is created with a header row on first use.
    Subsequent writes append without rewriting the header.

    Thread-safety: not guaranteed. The scanner runs on a single
    thread so no locking is needed in Phase 2.
    """

    def __init__(self, journal_path: Path = DEFAULT_JOURNAL_PATH) -> None:
        if not isinstance(journal_path, Path):
            journal_path = Path(journal_path)
        self.journal_path = journal_path
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.journal_path.exists():
            with self.journal_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            log.info("Trade journal created | Path=%s", self.journal_path)

    def _write_row(self, row: dict) -> None:
        with self.journal_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)

    @staticmethod
    def _fmt(value, digits: int = 5) -> str:
        if value is None:
            return ""
        try:
            v = float(value)
            if not math.isfinite(v):
                return ""
            return f"{v:.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_setup_confirmed(
        self,
        instrument: str,
        direction: str,
        setup_id: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        sl_distance: float,
        lot_size: float,
        risk_amount: float,
        rr_ratio: float,
        account_balance: float,
        daily_loss_pct: float = 0.0,
        open_trade_count: int = 0,
    ) -> None:
        row = {
            "event_time": self._now(),
            "event_type": "SETUP_CONFIRMED",
            "instrument": instrument,
            "direction": direction,
            "setup_id": setup_id,
            "entry_price": self._fmt(entry_price),
            "stop_loss": self._fmt(stop_loss),
            "take_profit": self._fmt(take_profit),
            "sl_distance": self._fmt(sl_distance),
            "lot_size": self._fmt(lot_size, 2),
            "risk_amount": self._fmt(risk_amount, 2),
            "rr_ratio": self._fmt(rr_ratio, 1),
            "account_balance": self._fmt(account_balance, 2),
            "daily_loss_pct": self._fmt(daily_loss_pct, 2),
            "open_trade_count": open_trade_count,
            "reason": "Setup confirmed by M5 BOS.",
        }
        self._write_row(row)
        log.info(
            "Journal | SETUP_CONFIRMED | %s | %s | SetupID=%s",
            instrument,
            direction,
            setup_id,
        )

    def log_risk_approved(
        self,
        instrument: str,
        direction: str,
        setup_id: str,
        adjusted_lot_size: float,
        account_balance: float,
        daily_loss_pct: float = 0.0,
        open_trade_count: int = 0,
    ) -> None:
        row = {
            "event_time": self._now(),
            "event_type": "RISK_APPROVED",
            "instrument": instrument,
            "direction": direction,
            "setup_id": setup_id,
            "entry_price": "",
            "stop_loss": "",
            "take_profit": "",
            "sl_distance": "",
            "lot_size": self._fmt(adjusted_lot_size, 2),
            "risk_amount": "",
            "rr_ratio": "",
            "account_balance": self._fmt(account_balance, 2),
            "daily_loss_pct": self._fmt(daily_loss_pct, 2),
            "open_trade_count": open_trade_count,
            "reason": "Risk manager approved trade.",
        }
        self._write_row(row)
        log.info(
            "Journal | RISK_APPROVED | %s | SetupID=%s | Lots=%s",
            instrument,
            setup_id,
            self._fmt(adjusted_lot_size, 2),
        )

    def log_risk_blocked(
        self,
        instrument: str,
        direction: str,
        setup_id: str,
        reason: str,
        account_balance: float,
        daily_loss_pct: float = 0.0,
        open_trade_count: int = 0,
    ) -> None:
        row = {
            "event_time": self._now(),
            "event_type": "RISK_BLOCKED",
            "instrument": instrument,
            "direction": direction,
            "setup_id": setup_id,
            "entry_price": "",
            "stop_loss": "",
            "take_profit": "",
            "sl_distance": "",
            "lot_size": "",
            "risk_amount": "",
            "rr_ratio": "",
            "account_balance": self._fmt(account_balance, 2),
            "daily_loss_pct": self._fmt(daily_loss_pct, 2),
            "open_trade_count": open_trade_count,
            "reason": reason,
        }
        self._write_row(row)
        log.warning(
            "Journal | RISK_BLOCKED | %s | SetupID=%s | Reason=%s",
            instrument,
            setup_id,
            reason,
        )

    def log_trade_opened(
        self,
        instrument: str,
        direction: str,
        setup_id: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,
        account_balance: float,
    ) -> None:
        row = {
            "event_time": self._now(),
            "event_type": "TRADE_OPENED",
            "instrument": instrument,
            "direction": direction,
            "setup_id": setup_id,
            "entry_price": self._fmt(entry_price),
            "stop_loss": self._fmt(stop_loss),
            "take_profit": self._fmt(take_profit),
            "sl_distance": "",
            "lot_size": self._fmt(lot_size, 2),
            "risk_amount": "",
            "rr_ratio": "",
            "account_balance": self._fmt(account_balance, 2),
            "daily_loss_pct": "",
            "open_trade_count": "",
            "reason": "Order placed.",
        }
        self._write_row(row)
        log.info(
            "Journal | TRADE_OPENED | %s | %s | Entry=%s | SetupID=%s",
            instrument,
            direction,
            self._fmt(entry_price),
            setup_id,
        )

    def log_trade_closed(
        self,
        instrument: str,
        direction: str,
        setup_id: str,
        entry_price: float,
        close_price: float,
        lot_size: float,
        pnl: float,
        account_balance: float,
        reason: str = "Position closed.",
    ) -> None:
        row = {
            "event_time": self._now(),
            "event_type": "TRADE_CLOSED",
            "instrument": instrument,
            "direction": direction,
            "setup_id": setup_id,
            "entry_price": self._fmt(entry_price),
            "stop_loss": self._fmt(close_price),
            "take_profit": "",
            "sl_distance": "",
            "lot_size": self._fmt(lot_size, 2),
            "risk_amount": self._fmt(pnl, 2),
            "rr_ratio": "",
            "account_balance": self._fmt(account_balance, 2),
            "daily_loss_pct": "",
            "open_trade_count": "",
            "reason": reason,
        }
        self._write_row(row)
        log.info(
            "Journal | TRADE_CLOSED | %s | PnL=%s | SetupID=%s",
            instrument,
            self._fmt(pnl, 2),
            setup_id,
        )
