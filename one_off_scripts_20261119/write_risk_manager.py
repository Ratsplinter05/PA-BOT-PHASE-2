from pathlib import Path

code = '''"""
Risk manager.

Gates every confirmed setup against account-level and daily
risk limits before trade parameters are accepted.

This module is analysis-only in Phase 2. It does not place,
modify, or manage trading orders.

Rules enforced:
  1. Risk per trade does not exceed max_risk_percent of balance.
  2. Daily loss does not exceed max_daily_loss_percent of the
     opening balance for that trading day (UTC).
  3. No more than max_concurrent_trades open at the same time.
  4. No more than max_trades_per_symbol_per_day per symbol per day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from src.logger import get_logger

log = get_logger("RISK_MANAGER")

MAX_RISK_PERCENT = 2.0
MAX_DAILY_LOSS_PERCENT = 4.0
MAX_CONCURRENT_TRADES = 1
MAX_TRADES_PER_SYMBOL_PER_DAY = 1


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of one risk check."""

    approved: bool
    reason: str
    adjusted_lot_size: Optional[float] = None


@dataclass
class DailyState:
    """Tracks risk state for one UTC trading day."""

    trading_date: date
    opening_balance: float
    realised_loss: float = 0.0
    open_trade_count: int = 0
    trades_per_symbol: dict = field(default_factory=dict)

    def loss_percent(self) -> float:
        if self.opening_balance <= 0:
            return 0.0
        return (self.realised_loss / self.opening_balance) * 100.0

    def trades_today_for(self, symbol: str) -> int:
        return self.trades_per_symbol.get(symbol.upper(), 0)


class RiskManager:
    """
    Enforces account-level and daily risk limits.

    Call check() before accepting trade parameters.
    Call record_trade_opened() after a trade is placed.
    Call record_trade_closed() after a trade closes.

    No orders are placed or managed by this class.
    """

    def __init__(
        self,
        max_risk_percent: float = MAX_RISK_PERCENT,
        max_daily_loss_percent: float = MAX_DAILY_LOSS_PERCENT,
        max_concurrent_trades: int = MAX_CONCURRENT_TRADES,
        max_trades_per_symbol_per_day: int = MAX_TRADES_PER_SYMBOL_PER_DAY,
    ) -> None:

        for name, value in [
            ("max_risk_percent", max_risk_percent),
            ("max_daily_loss_percent", max_daily_loss_percent),
        ]:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite.")

        for name, value in [
            ("max_concurrent_trades", max_concurrent_trades),
            ("max_trades_per_symbol_per_day", max_trades_per_symbol_per_day),
        ]:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")

        self.max_risk_percent = float(max_risk_percent)
        self.max_daily_loss_percent = float(max_daily_loss_percent)
        self.max_concurrent_trades = max_concurrent_trades
        self.max_trades_per_symbol_per_day = max_trades_per_symbol_per_day
        self._daily_state: Optional[DailyState] = None

    def _today(self) -> date:
        return datetime.now(timezone.utc).date()

    def _ensure_daily_state(self, account_balance: float) -> DailyState:
        today = self._today()
        if (
            self._daily_state is None
            or self._daily_state.trading_date != today
        ):
            carry_open = (
                self._daily_state.open_trade_count
                if self._daily_state is not None
                else 0
            )
            self._daily_state = DailyState(
                trading_date=today,
                opening_balance=account_balance,
                open_trade_count=carry_open,
            )
            log.info(
                "Daily state initialised | Date=%s | OpeningBalance=%.2f",
                today,
                account_balance,
            )
        return self._daily_state

    def _max_lot_size(
        self,
        account_balance: float,
        sl_distance: float,
        pip_value_per_lot: float,
        min_lot_size: float = 0.01,
    ) -> float:
        risk_amount = account_balance * self.max_risk_percent / 100.0
        if sl_distance <= 0 or pip_value_per_lot <= 0:
            return min_lot_size
        raw = risk_amount / (sl_distance * pip_value_per_lot)
        return max(round(raw, 2), min_lot_size)

    def check(
        self,
        symbol: str,
        account_balance: float,
        proposed_lot_size: float,
        sl_distance: float,
        pip_value_per_lot: float,
    ) -> RiskDecision:
        """Decide whether this trade is permitted under current risk limits."""

        if not isinstance(symbol, str) or not symbol.strip():
            return RiskDecision(
                approved=False,
                reason="Symbol must be a non-empty string.",
            )

        symbol = symbol.strip().upper()

        for name, value in [
            ("account_balance", account_balance),
            ("proposed_lot_size", proposed_lot_size),
            ("sl_distance", sl_distance),
            ("pip_value_per_lot", pip_value_per_lot),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                return RiskDecision(
                    approved=False,
                    reason=f"{name} must be a positive finite number.",
                )

        state = self._ensure_daily_state(account_balance)

        loss_pct = state.loss_percent()
        if loss_pct >= self.max_daily_loss_percent:
            reason = (
                f"Daily loss limit reached: {loss_pct:.2f}% of opening "
                f"balance ({self.max_daily_loss_percent:.1f}% max). "
                "No new trades until tomorrow."
            )
            log.warning(reason)
            return RiskDecision(approved=False, reason=reason)

        if state.open_trade_count >= self.max_concurrent_trades:
            reason = (
                f"Concurrent trade limit reached: "
                f"{state.open_trade_count}/{self.max_concurrent_trades} "
                "open trades."
            )
            log.warning(reason)
            return RiskDecision(approved=False, reason=reason)

        trades_today = state.trades_today_for(symbol)
        if trades_today >= self.max_trades_per_symbol_per_day:
            reason = (
                f"Symbol trade limit reached: {trades_today} trade(s) "
                f"already taken on {symbol} today "
                f"({self.max_trades_per_symbol_per_day} max)."
            )
            log.warning(reason)
            return RiskDecision(approved=False, reason=reason)

        max_lots = self._max_lot_size(
            account_balance, sl_distance, pip_value_per_lot
        )
        adjusted = min(proposed_lot_size, max_lots)

        if adjusted != proposed_lot_size:
            log.info(
                "Lot size capped | Symbol=%s | Proposed=%.2f | "
                "Adjusted=%.2f | MaxRisk=%.1f%%",
                symbol,
                proposed_lot_size,
                adjusted,
                self.max_risk_percent,
            )

        log.info(
            "Trade approved | Symbol=%s | Lots=%.2f | "
            "OpenTrades=%d | TradesToday=%d | DailyLoss=%.2f%%",
            symbol,
            adjusted,
            state.open_trade_count,
            trades_today,
            loss_pct,
        )

        return RiskDecision(
            approved=True,
            reason="All risk checks passed.",
            adjusted_lot_size=adjusted,
        )

    def record_trade_opened(
        self, symbol: str, account_balance: float
    ) -> None:
        """Call this after a trade is successfully placed."""
        state = self._ensure_daily_state(account_balance)
        state.open_trade_count += 1
        sym = symbol.strip().upper()
        state.trades_per_symbol[sym] = (
            state.trades_per_symbol.get(sym, 0) + 1
        )
        log.info(
            "Trade opened | Symbol=%s | OpenTrades=%d | TradesToday=%d",
            sym,
            state.open_trade_count,
            state.trades_per_symbol[sym],
        )

    def record_trade_closed(
        self,
        symbol: str,
        account_balance: float,
        pnl: float,
    ) -> None:
        """Call this after a trade closes. pnl is negative for a loss."""
        state = self._ensure_daily_state(account_balance)
        state.open_trade_count = max(0, state.open_trade_count - 1)
        if pnl < 0:
            state.realised_loss += abs(pnl)
        log.info(
            "Trade closed | Symbol=%s | PnL=%.2f | "
            "DailyLoss=%.2f%% | OpenTrades=%d",
            symbol.strip().upper(),
            pnl,
            state.loss_percent(),
            state.open_trade_count,
        )

    @property
    def daily_state(self) -> Optional[DailyState]:
        return self._daily_state
'''

Path("src/risk_manager.py").write_text(code, encoding="utf-8")
print("Written", Path("src/risk_manager.py").stat().st_size, "bytes")
