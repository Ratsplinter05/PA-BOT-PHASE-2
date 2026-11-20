"""Tests for RiskManager."""

from datetime import date, datetime, timezone
from unittest.mock import patch
import pytest

from src.risk_manager import RiskManager, DailyState


def _make_manager(**kwargs):
    defaults = dict(
        max_risk_percent=2.0,
        max_daily_loss_percent=4.0,
        max_concurrent_trades=1,
        max_trades_per_symbol_per_day=1,
    )
    defaults.update(kwargs)
    return RiskManager(**defaults)


def _check(manager, symbol="EURUSD", balance=10000.0,
           lots=0.10, sl=0.0050, pip_value=10.0):
    return manager.check(symbol, balance, lots, sl, pip_value)


class TestRiskManagerInit:

    def test_default_construction(self):
        rm = RiskManager()
        assert rm.max_risk_percent == 2.0
        assert rm.max_daily_loss_percent == 4.0
        assert rm.max_concurrent_trades == 1
        assert rm.max_trades_per_symbol_per_day == 1

    def test_invalid_risk_percent_type(self):
        with pytest.raises(TypeError):
            RiskManager(max_risk_percent="2")

    def test_invalid_risk_percent_zero(self):
        with pytest.raises(ValueError):
            RiskManager(max_risk_percent=0.0)

    def test_invalid_concurrent_trades_type(self):
        with pytest.raises(TypeError):
            RiskManager(max_concurrent_trades=1.5)

    def test_invalid_concurrent_trades_zero(self):
        with pytest.raises(ValueError):
            RiskManager(max_concurrent_trades=0)


class TestRiskCheckApproval:

    def test_valid_trade_approved(self):
        rm = _make_manager()
        result = _check(rm)
        assert result.approved
        assert result.adjusted_lot_size is not None

    def test_lot_size_capped_to_risk_limit(self):
        rm = _make_manager(max_risk_percent=2.0)
        # max_lot = 10000 * 2% / (0.0050 * 10) = 200 / 0.05 = 4000.00 lots
        # proposed 0.10 is well within limit so no cap needed
        result = _check(rm, lots=0.10)
        assert result.approved
        assert result.adjusted_lot_size == pytest.approx(0.10)

    def test_oversized_lot_capped(self):
        rm = _make_manager(max_risk_percent=1.0)
        # max_lot = 10000 * 1% / (0.0050 * 10) = 100 / 0.05 = 2000.00
        # proposed 9999 exceeds max → capped
        result = _check(rm, lots=9999.0, sl=0.0050, pip_value=10.0)
        assert result.approved
        assert result.adjusted_lot_size <= 2000.0

    def test_symbol_uppercased_internally(self):
        rm = _make_manager()
        result = rm.check("eurusd", 10000.0, 0.10, 0.0050, 10.0)
        assert result.approved


class TestDailyLossLimit:

    def test_trade_blocked_when_daily_loss_reached(self):
        rm = _make_manager(max_daily_loss_percent=4.0)
        _check(rm)  # initialise daily state
        # Simulate 4% loss on 10000 = 400
        rm._daily_state.realised_loss = 400.0
        result = _check(rm)
        assert not result.approved
        assert "Daily loss limit" in result.reason

    def test_trade_allowed_just_below_limit(self):
        rm = _make_manager(max_daily_loss_percent=4.0)
        _check(rm)
        rm._daily_state.realised_loss = 399.0
        result = _check(rm)
        assert result.approved


class TestConcurrentTradeLimit:

    def test_blocked_when_max_concurrent_reached(self):
        rm = _make_manager(max_concurrent_trades=1)
        _check(rm)
        rm.record_trade_opened("EURUSD", 10000.0)
        result = _check(rm, symbol="GBPUSD")
        assert not result.approved
        assert "Concurrent" in result.reason

    def test_allowed_when_under_limit(self):
        rm = _make_manager(max_concurrent_trades=2)
        _check(rm)
        rm.record_trade_opened("EURUSD", 10000.0)
        result = _check(rm, symbol="GBPUSD")
        assert result.approved


class TestSymbolDailyLimit:

    def test_blocked_when_symbol_limit_reached(self):
        rm = _make_manager(
            max_concurrent_trades=2,
            max_trades_per_symbol_per_day=1,
        )
        _check(rm)
        rm.record_trade_opened("EURUSD", 10000.0)
        result = _check(rm, symbol="EURUSD")
        assert not result.approved
        assert "Symbol trade limit" in result.reason

    def test_different_symbol_still_allowed(self):
        rm = _make_manager(
            max_concurrent_trades=2,
            max_trades_per_symbol_per_day=1,
        )
        _check(rm)
        rm.record_trade_opened("EURUSD", 10000.0)
        result = _check(rm, symbol="GBPUSD")
        assert result.approved


class TestRecordTradeClosed:

    def test_open_count_decrements(self):
        rm = _make_manager()
        _check(rm)
        rm.record_trade_opened("EURUSD", 10000.0)
        assert rm.daily_state.open_trade_count == 1
        rm.record_trade_closed("EURUSD", 10000.0, pnl=-50.0)
        assert rm.daily_state.open_trade_count == 0

    def test_loss_accumulated(self):
        rm = _make_manager()
        _check(rm)
        rm.record_trade_opened("EURUSD", 10000.0)
        rm.record_trade_closed("EURUSD", 10000.0, pnl=-200.0)
        assert rm.daily_state.realised_loss == pytest.approx(200.0)

    def test_winning_trade_does_not_increase_loss(self):
        rm = _make_manager()
        _check(rm)
        rm.record_trade_opened("EURUSD", 10000.0)
        rm.record_trade_closed("EURUSD", 10000.0, pnl=400.0)
        assert rm.daily_state.realised_loss == pytest.approx(0.0)

    def test_open_count_does_not_go_below_zero(self):
        rm = _make_manager()
        _check(rm)
        rm.record_trade_closed("EURUSD", 10000.0, pnl=0.0)
        assert rm.daily_state.open_trade_count == 0


class TestInputValidation:

    def test_empty_symbol_rejected(self):
        rm = _make_manager()
        result = rm.check("  ", 10000.0, 0.10, 0.0050, 10.0)
        assert not result.approved

    def test_zero_balance_rejected(self):
        rm = _make_manager()
        result = rm.check("EURUSD", 0.0, 0.10, 0.0050, 10.0)
        assert not result.approved

    def test_zero_sl_distance_rejected(self):
        rm = _make_manager()
        result = rm.check("EURUSD", 10000.0, 0.10, 0.0, 10.0)
        assert not result.approved