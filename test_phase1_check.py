from types import SimpleNamespace

import pandas as pd

from src import phase1_check


def make_candles(n, close=1.1):
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": close, "high": close + 0.001,
        "low": close - 0.001, "close": close,
        "is_forming_candle": False,
    })


class FakeMarket:
    def __init__(self, symbols, candles_by_tf):
        self.symbols = symbols
        self.candles_by_tf = candles_by_tf

    def discover_symbols(self):
        return self.symbols

    def resolve_symbol(self, requested, available):
        return requested if requested in available else None

    def enable_symbol(self, symbol):
        return SimpleNamespace(digits=5, point=0.00001, trade_mode=4,
                               volume_min=0.01, volume_step=0.01)

    def get_tick(self, symbol):
        return {"bid": 1.1, "ask": 1.1001}

    def get_closed_candles(self, symbol, timeframe, count=300):
        return self.candles_by_tf[timeframe]


class FakeValidator:
    def validate(self, candles, timeframe=None):
        return {"status": "PASS", "valid": True,
                "errors": [], "warnings": []}


class FakeConnection:
    disconnected = False

    def connect(self):
        terminal = SimpleNamespace(name="MT5")
        account = SimpleNamespace(company="Deriv", server="Deriv-Demo",
                                  login=1, currency="USD",
                                  balance=1000, equity=1000)
        return terminal, account

    def disconnect(self):
        FakeConnection.disconnected = True


SETTINGS = {
    "market": {"instruments": ["EURUSD", "XAUUSD"],
               "timeframes": {"M15": "M15"},
               "candles_per_timeframe": 10},
    "trading": {"execution_enabled": False},
}


def run_main(monkeypatch, market):
    FakeConnection.disconnected = False
    monkeypatch.setattr(phase1_check, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(phase1_check, "MarketData", lambda: market)
    monkeypatch.setattr(phase1_check, "CandleValidator", FakeValidator)
    monkeypatch.setattr(phase1_check, "MT5Connection", FakeConnection)
    return phase1_check.main()


def test_main_returns_zero_when_all_checks_pass(monkeypatch):
    market = FakeMarket(["EURUSD.0", "XAUUSD.0"], {"M15": make_candles(10)})
    assert run_main(monkeypatch, market) == 0
    assert FakeConnection.disconnected


def test_main_returns_one_when_symbol_missing(monkeypatch):
    market = FakeMarket(["EURUSD.0"], {"M15": make_candles(10)})
    assert run_main(monkeypatch, market) == 1
    assert FakeConnection.disconnected


def test_short_history_is_a_warning_not_a_pass():
    rows = phase1_check.check_instrument(
        FakeMarket([], {"M15": make_candles(5)}), FakeValidator(),
        "EURUSD", "EURUSD.0", ["M15"], 10)
    assert rows[0]["status"] == "WARNING"


def test_empty_candles_fail():
    rows = phase1_check.check_instrument(
        FakeMarket([], {"M15": pd.DataFrame()}), FakeValidator(),
        "EURUSD", "EURUSD.0", ["M15"], 10)
    assert rows[0]["status"] == "FAIL"


def test_resolve_broker_symbol_prefers_suffixed_name():
    market = FakeMarket(["EURUSD", "EURUSD.0"], {})
    assert phase1_check.resolve_broker_symbol(
        market, "EURUSD", market.symbols) == "EURUSD.0"
