"""Tests for LiveScanner (no MT5 connection required)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.live_scanner import LiveScanner, ScanResult


def _make_candle_df(rows=20):
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open":  [1.1000] * rows,
        "high":  [1.1050] * rows,
        "low":   [1.0950] * rows,
        "close": [1.1020] * rows,
        "tick_volume": [100] * rows,
        "spread": [10] * rows,
        "real_volume": [0] * rows,
        "is_forming_candle": [False] * rows,
    })


def _make_scanner(interval=15):
    with patch("src.live_scanner.MT5Connection"), \
         patch("src.live_scanner.MarketData"), \
         patch("src.live_scanner.StrategyOrchestrator"), \
         patch("src.live_scanner.TradeParameterCalculator"):
        scanner = LiveScanner(scan_interval_seconds=interval)
    return scanner


class TestLiveScannerInit:

    def test_default_construction(self):
        scanner = _make_scanner()
        assert scanner.scan_interval_seconds == 15
        assert scanner.candles_per_timeframe == 300
        assert isinstance(scanner._seen_setup_ids, set)

    def test_seen_setup_ids_starts_empty(self):
        scanner = _make_scanner()
        assert len(scanner._seen_setup_ids) == 0


class TestFetchCandles:

    def test_returns_dict_when_all_timeframes_succeed(self):
        scanner = _make_scanner()
        df = _make_candle_df()
        scanner.market_data.get_closed_candles.return_value = df
        result = scanner._fetch_candles("EURUSD.0")
        assert result is not None
        assert set(result.keys()) == {"H4", "H1", "M15", "M5"}

    def test_returns_none_when_any_timeframe_empty(self):
        scanner = _make_scanner()
        empty = pd.DataFrame()
        scanner.market_data.get_closed_candles.return_value = empty
        result = scanner._fetch_candles("EURUSD.0")
        assert result is None


class TestResolveSymbol:

    def test_resolves_known_instrument(self):
        scanner = _make_scanner()
        scanner.market_data.discover_symbols.return_value = ["EURUSD.0", "GBPUSD.0"]
        scanner.market_data.resolve_symbol.side_effect = (
            lambda candidate, available: candidate if candidate in available else None
        )
        result = scanner._resolve_broker_symbol("EURUSD")
        assert result == "EURUSD.0"

    def test_returns_none_for_unknown_instrument(self):
        scanner = _make_scanner()
        scanner.market_data.discover_symbols.return_value = ["EURUSD.0"]
        scanner.market_data.resolve_symbol.return_value = None
        result = scanner._resolve_broker_symbol("UNKNOWN")
        assert result is None


class TestProcessConfirmations:

    def _make_confirmation(self, state="CONFIRMED", setup_id="s1", direction="BULLISH"):
        c = MagicMock()
        c.state = state
        c.setup_id = setup_id
        c.direction = direction
        c.qualification_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        c.confirmation_time = datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)
        event = MagicMock()
        event.close_price = 1.1050
        level_swing = MagicMock()
        level_swing.swing = MagicMock()
        level_swing.swing.price = 1.0990
        level_swing.swing.swing_type = "LOW"
        event.level_swing = level_swing
        c.confirmation_event = event
        return c

    def _make_m15_setup(self, direction="BULLISH"):
        s = MagicMock()
        s.state = "QUALIFIED"
        s.direction = direction
        s.break_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        s.atr_value = 0.0010
        s.broken_level = 1.1000
        s.qualification_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        return s

    def _make_pipeline_result(self, confirmations, setups):
        result = MagicMock()
        result.m5_confirmations.confirmations = confirmations
        result.m15_setups.setups = setups
        return result

    def test_new_confirmed_setup_is_counted(self):
        scanner = _make_scanner()
        conf = self._make_confirmation()
        setup = self._make_m15_setup()
        result = self._make_pipeline_result([conf], [setup])

        param_result = MagicMock()
        param_result.valid = True
        param_result.parameters = MagicMock()
        param_result.parameters.direction = "BULLISH"
        param_result.parameters.entry_price = 1.1050
        param_result.parameters.stop_loss = 1.0980
        param_result.parameters.take_profit = 1.1190
        param_result.parameters.lot_size = 1.43
        param_result.parameters.risk_amount = 100.0
        param_result.parameters.rr_ratio = 2.0
        param_result.parameters.setup_id = "s1"
        scanner.calculator.calculate.return_value = param_result

        count = scanner._process_confirmations(
            "EURUSD", "EURUSD.0", result, 10000.0, 10.0
        )
        assert count == 1
        assert "s1" in scanner._seen_setup_ids

    def test_already_seen_setup_not_counted_again(self):
        scanner = _make_scanner()
        scanner._seen_setup_ids.add("s1")
        conf = self._make_confirmation(setup_id="s1")
        setup = self._make_m15_setup()
        result = self._make_pipeline_result([conf], [setup])
        count = scanner._process_confirmations(
            "EURUSD", "EURUSD.0", result, 10000.0, 10.0
        )
        assert count == 0

    def test_non_confirmed_state_skipped(self):
        scanner = _make_scanner()
        conf = self._make_confirmation(state="WAITING")
        result = self._make_pipeline_result([conf], [])
        count = scanner._process_confirmations(
            "EURUSD", "EURUSD.0", result, 10000.0, 10.0
        )
        assert count == 0


class TestScanInstrument:

    def test_skipped_when_candles_empty(self):
        scanner = _make_scanner()
        scanner.market_data.get_closed_candles.return_value = pd.DataFrame()
        result = scanner._scan_instrument("EURUSD", "EURUSD.0", 10000.0)
        assert isinstance(result, ScanResult)
        assert result.skipped_reason is not None
        assert result.confirmed_count == 0

    def test_skipped_when_orchestrator_raises(self):
        scanner = _make_scanner()
        df = _make_candle_df()
        scanner.market_data.get_closed_candles.return_value = df
        scanner.orchestrator.analyze.side_effect = RuntimeError("boom")
        result = scanner._scan_instrument("EURUSD", "EURUSD.0", 10000.0)
        assert result.skipped_reason is not None
        assert "Orchestrator error" in result.skipped_reason