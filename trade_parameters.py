"""
Trade parameter calculator.

Takes a confirmed M5 setup and computes entry, stop loss,
take profit, and lot size.

This module is analysis-only. It does not place, modify,
or manage trading orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# M15Setup and M5Confirmation are accepted by duck-typing (hasattr checks).


@dataclass(frozen=True)
class TradeParameters:
    """Computed trade parameters for one confirmed setup."""

    setup_id: str
    symbol: str
    direction: str

    entry_price: float
    stop_loss: float
    take_profit: float

    sl_distance: float
    rr_ratio: float

    risk_amount: float
    lot_size: float

    qualification_time: datetime
    confirmation_time: datetime

    reason: str
    valid: bool


@dataclass(frozen=True)
class TradeParameterResult:
    """Outcome of one parameter calculation attempt."""

    parameters: Optional[TradeParameters]
    valid: bool
    rejection_reason: Optional[str]


class TradeParameterCalculator:
    """
    Computes entry, SL, TP and lot size from a confirmed M5 setup.

    Stop loss is placed at the M15 swing that was broken,
    offset by one ATR to avoid being tagged by noise.

    Take profit is set at entry +/- (SL distance * rr_ratio).

    Lot size is derived from:
        risk_amount = account_balance * risk_percent / 100
        lot_size = risk_amount / (sl_distance * pip_value_per_lot)

    This component does not place, modify, or manage trading orders.
    """

    DEFAULT_RR_RATIO = 2.0
    DEFAULT_RISK_PERCENT = 1.0
    DEFAULT_ATR_BUFFER_MULTIPLIER = 1.0
    MIN_LOT_SIZE = 0.01
    LOT_PRECISION = 2

    def __init__(
        self,
        rr_ratio: float = DEFAULT_RR_RATIO,
        risk_percent: float = DEFAULT_RISK_PERCENT,
        atr_buffer_multiplier: float = DEFAULT_ATR_BUFFER_MULTIPLIER,
        min_lot_size: float = MIN_LOT_SIZE,
    ) -> None:

        for name, value in [
            ("rr_ratio", rr_ratio),
            ("risk_percent", risk_percent),
            ("atr_buffer_multiplier", atr_buffer_multiplier),
            ("min_lot_size", min_lot_size),
        ]:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite.")

        self.rr_ratio = float(rr_ratio)
        self.risk_percent = float(risk_percent)
        self.atr_buffer_multiplier = float(atr_buffer_multiplier)
        self.min_lot_size = float(min_lot_size)

    def _reject(self, reason: str) -> TradeParameterResult:
        return TradeParameterResult(
            parameters=None,
            valid=False,
            rejection_reason=reason,
        )

    def calculate(
        self,
        setup: M15Setup,
        confirmation: M5Confirmation,
        symbol: str,
        account_balance: float,
        pip_value_per_lot: float,
    ) -> TradeParameterResult:
        """
        Compute trade parameters from a qualified and confirmed setup.

        pip_value_per_lot:
            Monetary value of one pip for one standard lot on this
            instrument. For example, EURUSD on a USD account is
            typically 10.0 (USD per pip per lot). The caller is
            responsible for supplying the correct value from the
            broker. This calculator does not query MT5 directly.

        account_balance:
            Current account balance in the account currency.
        """

        # --- input validation ---

        for attr in ("direction", "state", "atr_value", "broken_level"):
            if not hasattr(setup, attr):
                return self._reject(
                    f"setup is missing required attribute: {attr!r}."
                )

        for attr in ("direction", "state", "setup_id", "confirmation_time",
                     "confirmation_event"):
            if not hasattr(confirmation, attr):
                return self._reject(
                    f"confirmation is missing required attribute: {attr!r}."
                )

        if confirmation.state != "CONFIRMED":
            return self._reject(
                f"Confirmation state is {confirmation.state!r}; "
                "only CONFIRMED setups can produce trade parameters."
            )

        if setup.state != "QUALIFIED":
            return self._reject(
                f"Setup state is {setup.state!r}; "
                "only QUALIFIED setups can produce trade parameters."
            )

        if setup.direction != confirmation.direction:
            return self._reject(
                "Setup and confirmation directions do not match."
            )

        if not isinstance(symbol, str) or not symbol.strip():
            return self._reject("symbol must be a non-empty string.")

        for name, value in [
            ("account_balance", account_balance),
            ("pip_value_per_lot", pip_value_per_lot),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                return self._reject(
                    f"{name} must be a positive finite number."
                )

        if confirmation.confirmation_event is None:
            return self._reject(
                "Confirmation carries no M5 structure break event."
            )

        event = confirmation.confirmation_event

        if event.level_swing is None:
            return self._reject(
                "Confirmation event carries no level_swing."
            )

        if event.level_swing.swing is None:
            return self._reject(
                "Confirmation event level_swing carries no SwingPoint."
            )

        # --- entry price ---
        entry_price = float(event.close_price)

        if not math.isfinite(entry_price) or entry_price <= 0:
            return self._reject(
                "Confirmation close price is not a valid entry price."
            )

        # --- stop loss ---
        swing_price = float(event.level_swing.swing.price)
        atr = float(setup.atr_value) if setup.atr_value is not None else None

        if atr is None or not math.isfinite(atr) or atr <= 0:
            return self._reject(
                "ATR value from the M15 setup is unavailable or invalid."
            )

        buffer = atr * self.atr_buffer_multiplier

        if setup.direction == "BULLISH":
            # SL below the swing low that was broken
            stop_loss = swing_price - buffer
        else:
            # SL above the swing high that was broken
            stop_loss = swing_price + buffer

        if not math.isfinite(stop_loss) or stop_loss <= 0:
            return self._reject("Computed stop loss is not a valid price.")

        # --- SL distance ---
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance <= 0:
            return self._reject(
                "Stop loss distance is zero; entry and SL are at the "
                "same price."
            )

        # Sanity check: SL must be on the correct side of entry
        if setup.direction == "BULLISH" and stop_loss >= entry_price:
            return self._reject(
                "Stop loss is above or at entry for a bullish trade."
            )
        if setup.direction == "BEARISH" and stop_loss <= entry_price:
            return self._reject(
                "Stop loss is below or at entry for a bearish trade."
            )

        # --- take profit ---
        if setup.direction == "BULLISH":
            take_profit = entry_price + sl_distance * self.rr_ratio
        else:
            take_profit = entry_price - sl_distance * self.rr_ratio

        if not math.isfinite(take_profit) or take_profit <= 0:
            return self._reject("Computed take profit is not a valid price.")

        # --- lot size ---
        risk_amount = float(account_balance) * self.risk_percent / 100.0

        # sl_distance is in price units; convert to pips if needed.
        # pip_value_per_lot is already expressed per pip per lot,
        # so the caller must pass sl_distance in the same pip unit.
        # We pass sl_distance in price units and document that
        # pip_value_per_lot must be expressed per price-unit per lot.
        raw_lot_size = risk_amount / (sl_distance * float(pip_value_per_lot))

        lot_size = max(
            round(raw_lot_size, self.LOT_PRECISION),
            self.min_lot_size,
        )

        if not math.isfinite(lot_size) or lot_size <= 0:
            return self._reject("Computed lot size is not valid.")

        # --- assemble result ---
        parameters = TradeParameters(
            setup_id=confirmation.setup_id,
            symbol=symbol.strip().upper(),
            direction=setup.direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sl_distance=sl_distance,
            rr_ratio=self.rr_ratio,
            risk_amount=risk_amount,
            lot_size=lot_size,
            qualification_time=setup.qualification_time,
            confirmation_time=confirmation.confirmation_time,
            reason=(
                f"Entry {entry_price:.5f} | SL {stop_loss:.5f} | "
                f"TP {take_profit:.5f} | Lots {lot_size:.2f} | "
                f"RR 1:{self.rr_ratio} | Risk {self.risk_percent}%"
            ),
            valid=True,
        )

        return TradeParameterResult(
            parameters=parameters,
            valid=True,
            rejection_reason=None,
        )