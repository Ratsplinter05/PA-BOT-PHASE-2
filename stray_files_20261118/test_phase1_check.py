from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from src import phase1_check


def make_candles(n, close=1.1, last_open=None):
    """n M15 candles; the last one opened just before the latest quarter hour."""
    if last_open is None:
        now = pd.Timestamp.now(tz="UTC")
        last_open = now.floor("15min") - pd.Timedelta(minutes=15)
    times = pd.date_range(end=last_open, periods=n, freq="15min", tz="UTC")
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


# ---- freshness checks -------------------------------------------------

UTC = timezone.utc


def test_weekend_minutes_counts_only_closed_hours():
    fri = datetime(2026, 11, 6, 21, 0, tzinfo=UTC)   # Friday 21:00
    mon = datetime(2026, 11, 9, 0, 30, tzinfo=UTC)   # Monday 00:30
    # closure is Fri 22:00 -> Sun 22:00 = 48 hours
    assert phase1_check.weekend_minutes(fri, mon) == 48 * 60


def test_stale_candles_get_a_warning_after_a_week():
    old = pd.Timestamp("2026-11-06 20:00", tz="UTC")
    now = datetime(2026, 11, 16, 10, 30, tzinfo=UTC)
    rows = phase1_check.check_instrument(
        FakeMarket([], {"M15": make_candles(10, last_open=old)}),
        FakeValidator(), "EURUSD", "EURUSD.0", ["M15"], 10, now=now)
    assert rows[0]["status"] == "WARNING"
    assert "behind" in rows[0]["details"]


def test_fresh_candles_pass():
    now = datetime(2026, 11, 16, 10, 40, tzinfo=UTC)
    fresh = pd.Timestamp("2026-11-16 10:15", tz="UTC")
    rows = phase1_check.check_instrument(
        FakeMarket([], {"M15": make_candles(10, last_open=fresh)}),
        FakeValidator(), "EURUSD", "EURUSD.0", ["M15"], 10, now=now)
    assert rows[0]["status"] == "PASS"


def test_weekend_closure_is_not_counted_as_stale():
    # Last Friday candle closed 21:00; checked on Saturday midday (market closed).
    friday = pd.Timestamp("2026-11-06 20:45", tz="UTC")
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    rows = phase1_check.check_instrument(
        FakeMarket([], {"M15": make_candles(10, last_open=friday)}),
        FakeValidator(), "EURUSD", "EURUSD.0", ["M15"], 10, now=now)
    assert rows[0]["status"] == "PASS"
