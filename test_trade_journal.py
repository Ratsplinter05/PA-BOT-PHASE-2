"""Tests for TradeJournal."""

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
