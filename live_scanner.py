"""
Live scanner.

Runs a continuous scan loop that feeds fresh MT5 candle data
through the strategy pipeline and logs any confirmed setups
with their computed trade parameters.

This module is analysis-only in Phase 2. It does not place,
modify, or manage trading orders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5

from src.logger import get_logger
from src.market_data import MarketData
from src.mt5_connection import MT5Connection
from src.risk_manager import RiskManager
from src.trade_parameters import TradeParameterCalculator
from strategy.strategy_orchestrator import StrategyOrchestrator

log = get_logger("LIVE_SCANNER")


SYMBOL_CANDIDATES: dict[str, list[str]] = {
    "XAUUSD": ["XAUUSD.0", "XAUUSD"],
    "GBPUSD": ["GBPUSD.0", "GBPUSD"],
    "BTCUSD": ["BTCUSD.0", "BTCUSD"],
    "EURUSD": ["EURUSD.0", "EURUSD"],
    "US30":   ["Wall Street 30.0", "US30"],
    "NAS100": ["US Tech 100.0", "NAS100"],
}

TIMEFRAMES = ["H4", "H1", "M15", "M5"]
CANDLES_PER_TIMEFRAME = 300


@dataclass(frozen=True)
class ScanResult:
    """Record of one instrument scan."""

    instrument: str
    broker_symbol: str
    scan_time: datetime
    confirmed_count: int
    skipped_reason: Optional[str]


class LiveScanner:
    """
    Coordinates the live scanning loop.

    On each tick the scanner:
      1. Fetches closed candles for every instrument and timeframe.
      2. Runs the strategy orchestrator.
      3. For every new CONFIRMED setup, computes trade parameters.
      4. Logs the result.

    No orders are placed.
    """

    def __init__(
        self,
        scan_interval_seconds: int = 15,
        rr_ratio: float = 2.0,
        risk_percent: float = 1.0,
        candles_per_timeframe: int = CANDLES_PER_TIMEFRAME,
    ) -> None:
        self.scan_interval_seconds = scan_interval_seconds
        self.candles_per_timeframe = candles_per_timeframe

        self.connection = MT5Connection()
        self.market_data = MarketData()
        self.orchestrator = StrategyOrchestrator()
        self.calculator = TradeParameterCalculator(
            rr_ratio=rr_ratio,
            risk_percent=risk_percent,
        )
        self.risk_manager = RiskManager(
            max_risk_percent=risk_percent,
        )

        # Track setup IDs already processed so we do not log the
        # same confirmed setup on every scan tick.
        self._seen_setup_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def _resolve_broker_symbol(self, instrument: str) -> Optional[str]:
        """Return the broker symbol name for a logical instrument."""
        available = self.market_data.discover_symbols()
        for candidate in SYMBOL_CANDIDATES.get(instrument, [instrument]):
            resolved = self.market_data.resolve_symbol(candidate, available)
            if resolved:
                return resolved
        return None

    # ------------------------------------------------------------------
    # Candle fetching
    # ------------------------------------------------------------------

    def _fetch_candles(
        self,
        broker_symbol: str,
    ) -> Optional[dict[str, object]]:
        """
        Fetch closed candles for all timeframes for one symbol.

        Returns a dict keyed by timeframe, or None if any fetch fails.
        """
        candles: dict[str, object] = {}
        for tf in TIMEFRAMES:
            df = self.market_data.get_closed_candles(
                broker_symbol, tf, self.candles_per_timeframe
            )
            if df.empty:
                log.warning(
                    "Empty candle data | Symbol=%s | Timeframe=%s",
                    broker_symbol,
                    tf,
                )
                return None
            candles[tf] = df
        return candles

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    @staticmethod
    def _get_account_balance() -> Optional[float]:
        info = mt5.account_info()
        if info is None:
            return None
        return float(info.balance)

    @staticmethod
    def _get_pip_value(broker_symbol: str) -> Optional[float]:
        """
        Return the monetary value of one price unit per standard lot.

        MT5 trade_tick_value gives the value of one tick (trade_tick_size)
        per lot. We normalise to one price unit (point) so the calculator
        receives a consistent unit regardless of instrument.
        """
        info = mt5.symbol_info(broker_symbol)
        if info is None:
            return None
        tick_value = float(info.trade_tick_value)
        tick_size = float(info.trade_tick_size)
        if tick_size <= 0:
            return None
        point = float(info.point)
        if point <= 0:
            return None
        # Value per one price-unit per lot
        return tick_value * (point / tick_size)

    # ------------------------------------------------------------------
    # Setup processing
    # ------------------------------------------------------------------

    def _process_confirmations(
        self,
        instrument: str,
        broker_symbol: str,
        result,
        account_balance: float,
        pip_value: float,
    ) -> int:
        """Log and compute parameters for new confirmed setups."""
        new_count = 0

        for confirmation in result.m5_confirmations.confirmations:
            if confirmation.state != "CONFIRMED":
                continue
            if confirmation.setup_id in self._seen_setup_ids:
                continue

            self._seen_setup_ids.add(confirmation.setup_id)
            new_count += 1

            # Find the matching M15 setup
            matching_setup = None
            for setup in result.m15_setups.setups:
                if (
                    setup.state == "QUALIFIED"
                    and setup.direction == confirmation.direction
                    and setup.break_time == confirmation.qualification_time
                ):
                    matching_setup = setup
                    break

            if matching_setup is None:
                log.warning(
                    "CONFIRMED setup %s has no matching QUALIFIED M15 setup "
                    "| Instrument=%s | Direction=%s",
                    confirmation.setup_id,
                    instrument,
                    confirmation.direction,
                )
                continue

            param_result = self.calculator.calculate(
                setup=matching_setup,
                confirmation=confirmation,
                symbol=instrument,
                account_balance=account_balance,
                pip_value_per_lot=pip_value,
            )

            if param_result.valid:
                p = param_result.parameters
                risk_decision = self.risk_manager.check(
                    symbol=instrument,
                    account_balance=account_balance,
                    proposed_lot_size=p.lot_size,
                    sl_distance=p.sl_distance,
                    pip_value_per_lot=pip_value,
                )
                if not risk_decision.approved:
                    log.warning(
                        "Trade blocked by risk manager | "
                        "Instrument=%s | Reason=%s",
                        instrument,
                        risk_decision.reason,
                    )
                    continue
                p = param_result.parameters
                log.info(
                    "NEW CONFIRMED SETUP | Instrument=%s | Direction=%s | "
                    "Entry=%.5f | SL=%.5f | TP=%.5f | Lots=%.2f | "
                    "Risk=%.2f | RR=1:%.1f | SetupID=%s",
                    instrument,
                    p.direction,
                    p.entry_price,
                    p.stop_loss,
                    p.take_profit,
                    p.lot_size,
                    p.risk_amount,
                    p.rr_ratio,
                    p.setup_id,
                )
            else:
                log.warning(
                    "CONFIRMED setup %s | Parameters rejected | "
                    "Instrument=%s | Reason=%s",
                    confirmation.setup_id,
                    instrument,
                    param_result.rejection_reason,
                )

        return new_count

    # ------------------------------------------------------------------
    # Single instrument scan
    # ------------------------------------------------------------------

    def _scan_instrument(
        self,
        instrument: str,
        broker_symbol: str,
        account_balance: float,
    ) -> ScanResult:
        scan_time = datetime.now(timezone.utc)

        candles = self._fetch_candles(broker_symbol)
        if candles is None:
            return ScanResult(
                instrument=instrument,
                broker_symbol=broker_symbol,
                scan_time=scan_time,
                confirmed_count=0,
                skipped_reason="Failed to fetch candles.",
            )

        try:
            result = self.orchestrator.analyze(
                h4_candles=candles["H4"],
                h1_candles=candles["H1"],
                m15_candles=candles["M15"],
                m5_candles=candles["M5"],
                assessment_time=scan_time,
            )
        except Exception as exc:
            log.error(
                "Orchestrator error | Instrument=%s | %s",
                instrument,
                exc,
            )
            return ScanResult(
                instrument=instrument,
                broker_symbol=broker_symbol,
                scan_time=scan_time,
                confirmed_count=0,
                skipped_reason=f"Orchestrator error: {exc}",
            )

        pip_value = self._get_pip_value(broker_symbol)
        if pip_value is None or pip_value <= 0:
            log.warning(
                "Could not retrieve pip value | Instrument=%s",
                instrument,
            )
            pip_value = 1.0  # fallback: logs will show lot size as approximate

        confirmed = self._process_confirmations(
            instrument=instrument,
            broker_symbol=broker_symbol,
            result=result,
            account_balance=account_balance,
            pip_value=pip_value,
        )

        return ScanResult(
            instrument=instrument,
            broker_symbol=broker_symbol,
            scan_time=scan_time,
            confirmed_count=confirmed,
            skipped_reason=None,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the continuous scan loop. Blocks until interrupted."""
        log.info(
            "Live scanner starting | Instruments=%d | Interval=%ds",
            len(SYMBOL_CANDIDATES),
            self.scan_interval_seconds,
        )

        self.connection.connect()

        try:
            while True:
                tick_start = time.monotonic()
                log.info("Scan tick | %s", datetime.now(timezone.utc).isoformat())

                account_balance = self._get_account_balance()
                if account_balance is None or account_balance <= 0:
                    log.error("Could not retrieve account balance. Skipping tick.")
                    time.sleep(self.scan_interval_seconds)
                    continue

                for instrument in SYMBOL_CANDIDATES:
                    broker_symbol = self._resolve_broker_symbol(instrument)
                    if broker_symbol is None:
                        log.warning(
                            "Could not resolve broker symbol | Instrument=%s",
                            instrument,
                        )
                        continue

                    scan_result = self._scan_instrument(
                        instrument=instrument,
                        broker_symbol=broker_symbol,
                        account_balance=account_balance,
                    )

                    if scan_result.skipped_reason:
                        log.warning(
                            "Scan skipped | Instrument=%s | Reason=%s",
                            instrument,
                            scan_result.skipped_reason,
                        )

                elapsed = time.monotonic() - tick_start
                sleep_time = max(
                    0.0,
                    self.scan_interval_seconds - elapsed,
                )
                log.info(
                    "Tick complete | Elapsed=%.2fs | Sleeping=%.2fs",
                    elapsed,
                    sleep_time,
                )
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Live scanner stopped by user.")
        finally:
            self.connection.disconnect()
            log.info("MT5 connection closed.")