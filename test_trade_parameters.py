"""Tests for TradeParameterCalculator."""

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.trade_parameters import TradeParameterCalculator, TradeParameters


def _make_swing(price: float, swing_type: str = "LOW"):
    swing = MagicMock()
    swing.price = price
    swing.swing_type = swing_type
    return swing


def _make_level_swing(swing):
    level_swing = MagicMock()
    level_swing.swing = swing
    return level_swing


def _make_confirmation_event(close_price: float, swing_price: float, swing_type: str = "LOW"):
    event = MagicMock()
    event.close_price = close_price
    event.level_swing = _make_level_swing(_make_swing(swing_price, swing_type))
    return event


def _make_setup(
    direction: str = "BULLISH",
    state: str = "QUALIFIED",
    atr_value: float = 0.0010,
):
    setup = MagicMock()
    setup.direction = direction
    setup.state = state
    setup.atr_value = atr_value
    setup.broken_level = 1.1000
    setup.qualification_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return setup


def _make_confirmation(
    direction: str = "BULLISH",
    state: str = "CONFIRMED",
    close_price: float = 1.1050,
    swing_price: float = 1.0990,
    swing_type: str = "LOW",
    setup_id: str = "test-setup-001",
):
    conf = MagicMock()
    conf.direction = direction
    conf.state = state
    conf.setup_id = setup_id
    conf.confirmation_time = datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)
    conf.confirmation_event = _make_confirmation_event(
        close_price, swing_price, swing_type
    )
    return conf


class TestTradeParameterCalculatorInit:

    def test_default_construction(self):
        calc = TradeParameterCalculator()
        assert calc.rr_ratio == 2.0
        assert calc.risk_percent == 1.0

    def test_custom_construction(self):
        calc = TradeParameterCalculator(rr_ratio=3.0, risk_percent=0.5)
        assert calc.rr_ratio == 3.0
        assert calc.risk_percent == 0.5

    def test_invalid_rr_ratio_type(self):
        with pytest.raises(TypeError):
            TradeParameterCalculator(rr_ratio="2")

    def test_invalid_rr_ratio_zero(self):
        with pytest.raises(ValueError):
            TradeParameterCalculator(rr_ratio=0.0)

    def test_invalid_risk_percent_negative(self):
        with pytest.raises(ValueError):
            TradeParameterCalculator(risk_percent=-1.0)


class TestTradeParameterCalculatorBullish:

    def setup_method(self):
        self.calc = TradeParameterCalculator(rr_ratio=2.0, risk_percent=1.0)
        self.setup = _make_setup("BULLISH", "QUALIFIED", atr_value=0.0010)
        self.conf = _make_confirmation(
            direction="BULLISH",
            close_price=1.1050,
            swing_price=1.0990,
            swing_type="LOW",
        )

    def test_valid_bullish_trade(self):
        result = self.calc.calculate(
            self.setup, self.conf, "EURUSD", 10000.0, 10.0
        )
        assert result.valid
        assert result.parameters is not None
        p = result.parameters
        assert p.direction == "BULLISH"
        assert p.entry_price == pytest.approx(1.1050)
        # SL = swing_price - atr_buffer = 1.0990 - 0.0010 = 1.0980
        assert p.stop_loss == pytest.approx(1.0980)
        # TP = entry + sl_distance * 2 = 1.1050 + 0.0070 * 2 = 1.1190
        assert p.take_profit == pytest.approx(1.1190)
        assert p.stop_loss < p.entry_price < p.take_profit

    def test_lot_size_calculation(self):
        result = self.calc.calculate(
            self.setup, self.conf, "EURUSD", 10000.0, 10.0
        )
        p = result.parameters
        # risk_amount = 10000 * 1% = 100
        # sl_distance = 1.1050 - 1.0980 = 0.0070
        # lot_size = 100 / (0.0070 * 10) = 100 / 0.07 = 1428.57 -> 1428.57
        assert p.risk_amount == pytest.approx(100.0)
        assert p.lot_size >= 0.01

    def test_symbol_uppercased(self):
        result = self.calc.calculate(
            self.setup, self.conf, "eurusd", 10000.0, 10.0
        )
        assert result.parameters.symbol == "EURUSD"


class TestTradeParameterCalculatorBearish:

    def setup_method(self):
        self.calc = TradeParameterCalculator(rr_ratio=2.0, risk_percent=1.0)
        self.setup = _make_setup("BEARISH", "QUALIFIED", atr_value=0.0010)
        self.conf = _make_confirmation(
            direction="BEARISH",
            close_price=1.0950,
            swing_price=1.1010,
            swing_type="HIGH",
        )

    def test_valid_bearish_trade(self):
        result = self.calc.calculate(
            self.setup, self.conf, "EURUSD", 10000.0, 10.0
        )
        assert result.valid
        p = result.parameters
        assert p.direction == "BEARISH"
        assert p.entry_price == pytest.approx(1.0950)
        # SL = swing_price + atr_buffer = 1.1010 + 0.0010 = 1.1020
        assert p.stop_loss == pytest.approx(1.1020)
        assert p.stop_loss > p.entry_price
        assert p.take_profit < p.entry_price


class TestTradeParameterCalculatorRejections:

    def setup_method(self):
        self.calc = TradeParameterCalculator()

    def test_unconfirmed_confirmation_rejected(self):
        setup = _make_setup()
        conf = _make_confirmation(state="WAITING")
        result = self.calc.calculate(setup, conf, "EURUSD", 10000.0, 10.0)
        assert not result.valid
        assert "CONFIRMED" in result.rejection_reason

    def test_unqualified_setup_rejected(self):
        setup = _make_setup(state="EXPIRED")
        conf = _make_confirmation()
        result = self.calc.calculate(setup, conf, "EURUSD", 10000.0, 10.0)
        assert not result.valid
        assert "QUALIFIED" in result.rejection_reason

    def test_direction_mismatch_rejected(self):
        setup = _make_setup(direction="BULLISH")
        conf = _make_confirmation(direction="BEARISH")
        result = self.calc.calculate(setup, conf, "EURUSD", 10000.0, 10.0)
        assert not result.valid
        assert "direction" in result.rejection_reason.lower()

    def test_missing_atr_rejected(self):
        setup = _make_setup(atr_value=None)
        conf = _make_confirmation()
        result = self.calc.calculate(setup, conf, "EURUSD", 10000.0, 10.0)
        assert not result.valid
        assert "ATR" in result.rejection_reason

    def test_zero_account_balance_rejected(self):
        setup = _make_setup()
        conf = _make_confirmation()
        result = self.calc.calculate(setup, conf, "EURUSD", 0.0, 10.0)
        assert not result.valid

    def test_empty_symbol_rejected(self):
        setup = _make_setup()
        conf = _make_confirmation()
        result = self.calc.calculate(setup, conf, "  ", 10000.0, 10.0)
        assert not result.valid

    def test_sl_on_wrong_side_rejected(self):
        # Bullish trade where swing price is above entry
        setup = _make_setup("BULLISH", atr_value=0.0001)
        conf = _make_confirmation(
            direction="BULLISH",
            close_price=1.1000,
            swing_price=1.1100,  # swing LOW above entry - nonsense
        )
        result = self.calc.calculate(setup, conf, "EURUSD", 10000.0, 10.0)
        assert not result.valid