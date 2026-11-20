from pathlib import Path

journal_code = '''"""
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
'''

tests_code = '''"""Tests for TradeJournal."""

import csv
from pathlib import Path
import pytest

from src.trade_journal import TradeJournal, FIELDNAMES


@pytest.fixture
def journal(tmp_path):
    return TradeJournal(journal_path=tmp_path / "test_journal.csv")


def _read_rows(journal):
    with journal.journal_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestTradeJournalInit:

    def test_file_created_with_header(self, tmp_path):
        j = TradeJournal(journal_path=tmp_path / "j.csv")
        assert j.journal_path.exists()
        with j.journal_path.open() as f:
            header = f.readline().strip().split(",")
        assert header == FIELDNAMES

    def test_existing_file_not_overwritten(self, tmp_path):
        p = tmp_path / "j.csv"
        j1 = TradeJournal(journal_path=p)
        j1.log_risk_blocked("EURUSD", "BULLISH", "s1", "test", 10000.0)
        j2 = TradeJournal(journal_path=p)
        rows = _read_rows(j2)
        assert len(rows) == 1

    def test_parent_directory_created(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "j.csv"
        j = TradeJournal(journal_path=p)
        assert j.journal_path.exists()

    def test_string_path_accepted(self, tmp_path):
        p = str(tmp_path / "j.csv")
        j = TradeJournal(journal_path=p)
        assert j.journal_path.exists()


class TestLogSetupConfirmed:

    def test_row_written(self, journal):
        journal.log_setup_confirmed(
            instrument="EURUSD",
            direction="BULLISH",
            setup_id="s1",
            entry_price=1.1050,
            stop_loss=1.0980,
            take_profit=1.1190,
            sl_distance=0.0070,
            lot_size=1.43,
            risk_amount=100.0,
            rr_ratio=2.0,
            account_balance=10000.0,
            daily_loss_pct=0.0,
            open_trade_count=0,
        )
        rows = _read_rows(journal)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "SETUP_CONFIRMED"
        assert rows[0]["instrument"] == "EURUSD"
        assert rows[0]["direction"] == "BULLISH"
        assert rows[0]["setup_id"] == "s1"


class TestLogRiskApproved:

    def test_row_written(self, journal):
        journal.log_risk_approved(
            instrument="GBPUSD",
            direction="BEARISH",
            setup_id="s2",
            adjusted_lot_size=0.50,
            account_balance=10000.0,
        )
        rows = _read_rows(journal)
        assert rows[0]["event_type"] == "RISK_APPROVED"
        assert rows[0]["lot_size"] == "0.50"


class TestLogRiskBlocked:

    def test_row_written_with_reason(self, journal):
        journal.log_risk_blocked(
            instrument="XAUUSD",
            direction="BULLISH",
            setup_id="s3",
            reason="Daily loss limit reached.",
            account_balance=10000.0,
            daily_loss_pct=4.0,
        )
        rows = _read_rows(journal)
        assert rows[0]["event_type"] == "RISK_BLOCKED"
        assert "Daily loss" in rows[0]["reason"]


class TestLogTradeOpenedClosed:

    def test_opened_then_closed(self, journal):
        journal.log_trade_opened(
            instrument="BTCUSD",
            direction="BULLISH",
            setup_id="s4",
            entry_price=90000.0,
            stop_loss=89000.0,
            take_profit=92000.0,
            lot_size=0.01,
            account_balance=10000.0,
        )
        journal.log_trade_closed(
            instrument="BTCUSD",
            direction="BULLISH",
            setup_id="s4",
            entry_price=90000.0,
            close_price=92000.0,
            lot_size=0.01,
            pnl=200.0,
            account_balance=10200.0,
        )
        rows = _read_rows(journal)
        assert len(rows) == 2
        assert rows[0]["event_type"] == "TRADE_OPENED"
        assert rows[1]["event_type"] == "TRADE_CLOSED"

    def test_pnl_stored_in_risk_amount_column(self, journal):
        journal.log_trade_closed(
            instrument="EURUSD",
            direction="BEARISH",
            setup_id="s5",
            entry_price=1.1000,
            close_price=1.0860,
            lot_size=0.10,
            pnl=-100.0,
            account_balance=9900.0,
        )
        rows = _read_rows(journal)
        assert rows[0]["risk_amount"] == "-100.00"


class TestMultipleRows:

    def test_multiple_events_appended_in_order(self, journal):
        journal.log_risk_blocked("EURUSD", "BULLISH", "s1", "r1", 10000.0)
        journal.log_risk_blocked("GBPUSD", "BEARISH", "s2", "r2", 10000.0)
        journal.log_risk_approved("XAUUSD", "BULLISH", "s3", 0.10, 10000.0)
        rows = _read_rows(journal)
        assert len(rows) == 3
        assert rows[0]["setup_id"] == "s1"
        assert rows[1]["setup_id"] == "s2"
        assert rows[2]["event_type"] == "RISK_APPROVED"
'''

Path("src/trade_journal.py").write_text(journal_code, encoding="utf-8")
print("Written src/trade_journal.py", Path("src/trade_journal.py").stat().st_size, "bytes")

Path("tests/test_trade_journal.py").write_text(tests_code, encoding="utf-8")
print("Written tests/test_trade_journal.py", Path("tests/test_trade_journal.py").stat().st_size, "bytes")
