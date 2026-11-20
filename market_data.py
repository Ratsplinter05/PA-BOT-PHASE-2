from datetime import datetime, timedelta, timezone
import math
import pandas as pd
import MetaTrader5 as mt5

from src.logger import get_logger

log = get_logger("MARKET_DATA")


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MarketData:

    # --------------------------------------------------
    # SYMBOL DISCOVERY
    # --------------------------------------------------

    def discover_symbols(self):
        symbols = mt5.symbols_get()

        if symbols is None:
            raise RuntimeError(
                f"Could not retrieve broker symbols: {mt5.last_error()}"
            )

        return [s.name for s in symbols]

    def resolve_symbol(self, requested, available):
        if requested in available:
            return requested

        lookup = {name.upper(): name for name in available}

        return lookup.get(requested.upper())

    def enable_symbol(self, symbol):
        info = mt5.symbol_info(symbol)

        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")

        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(
                    f"Could not enable {symbol}: {mt5.last_error()}"
                )

        info = mt5.symbol_info(symbol)

        if info is None:
            raise RuntimeError(
                f"Could not retrieve updated information for {symbol}"
            )

        return info

    # --------------------------------------------------
    # RECENT RAW BID / ASK QUOTES
    # --------------------------------------------------

    def get_recent_quote(
        self,
        symbol,
        lookback_seconds=30,
        max_age_seconds=10
    ):
        """
        Retrieve and validate the latest historical Bid/Ask tick.

        Uses COPY_TICKS_INFO to retrieve quote-related updates.
        """

        # --------------------------------------------------
        # INPUT VALIDATION
        # --------------------------------------------------

        try:
            lookback_seconds = float(lookback_seconds)
            max_age_seconds = float(max_age_seconds)
        except (TypeError, ValueError):
            raise ValueError("Quote timing parameters must be numeric.")

        if not math.isfinite(lookback_seconds):
            raise ValueError("lookback_seconds must be finite.")

        if not math.isfinite(max_age_seconds):
            raise ValueError("max_age_seconds must be finite.")

        if lookback_seconds <= 0:
            raise ValueError("lookback_seconds must be positive.")

        if max_age_seconds < 0:
            raise ValueError("max_age_seconds cannot be negative.")

        info = self.enable_symbol(symbol)

        point = float(info.point)
        tick_size = float(info.trade_tick_size)
        digits = int(info.digits)

        if not math.isfinite(point) or point <= 0:
            raise ValueError(f"Invalid point size for {symbol}.")

        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError(f"Invalid tick size for {symbol}.")

        # --------------------------------------------------
        # RETRIEVE QUOTE TICKS
        # --------------------------------------------------

        request_end = datetime.now(timezone.utc)

        request_start = request_end - timedelta(
            seconds=lookback_seconds
        )

        ticks = mt5.copy_ticks_range(
            symbol,
            request_start,
            request_end,
            mt5.COPY_TICKS_INFO
        )

        if ticks is None or len(ticks) == 0:
            raise RuntimeError(
                f"No recent quote ticks for {symbol}: "
                f"{mt5.last_error()}"
            )

        ticks = sorted(
            ticks,
            key=lambda tick: int(tick["time_msc"])
        )

        latest = ticks[-1]

        # --------------------------------------------------
        # TIMESTAMP VALIDATION
        # --------------------------------------------------

        timestamp_ms = int(latest["time_msc"])

        if timestamp_ms <= 0:
            raise ValueError(
                f"Invalid tick timestamp for {symbol}."
            )

        timestamp = datetime.fromtimestamp(
            timestamp_ms / 1000,
            tz=timezone.utc
        )

        now = datetime.now(timezone.utc)

        age_seconds = (
            now - timestamp
        ).total_seconds()

        if age_seconds < 0:
            raise ValueError(
                f"Future-dated tick detected for {symbol}: "
                f"{age_seconds:.3f} seconds."
            )

        if timestamp < request_start or timestamp > request_end:
            raise ValueError(
                f"Tick timestamp outside requested window for {symbol}."
            )

        # --------------------------------------------------
        # BID / ASK VALIDATION
        # --------------------------------------------------

        bid = float(latest["bid"])
        ask = float(latest["ask"])

        if not math.isfinite(bid) or not math.isfinite(ask):
            raise ValueError(
                f"Non-finite Bid or Ask for {symbol}."
            )

        if bid <= 0 or ask <= 0:
            raise ValueError(
                f"Non-positive Bid or Ask for {symbol}."
            )

        if ask < bid:
            raise ValueError(
                f"Crossed market detected for {symbol}."
            )

        # --------------------------------------------------
        # PRICE PRECISION AND TICK ALIGNMENT
        # --------------------------------------------------

        tolerance = tick_size * 1e-6

        for price_name, price in [
            ("Bid", bid),
            ("Ask", ask)
        ]:

            rounded_price = round(price, digits)

            if not math.isclose(
                price,
                rounded_price,
                rel_tol=0.0,
                abs_tol=tolerance
            ):
                raise ValueError(
                    f"{price_name} exceeds permitted precision "
                    f"for {symbol}."
                )

            nearest_tick = (
                round(price / tick_size) * tick_size
            )

            if not math.isclose(
                price,
                nearest_tick,
                rel_tol=0.0,
                abs_tol=tolerance
            ):
                raise ValueError(
                    f"{price_name} is not aligned with broker "
                    f"tick size for {symbol}."
                )

        # --------------------------------------------------
        # SPREAD VALIDATION
        # --------------------------------------------------

        spread = ask - bid

        if not math.isfinite(spread) or spread < 0:
            raise ValueError(
                f"Invalid spread detected for {symbol}."
            )

        spread_points_raw = spread / point
        spread_points = round(spread_points_raw)

        if not math.isclose(
            spread_points_raw,
            spread_points,
            rel_tol=0.0,
            abs_tol=1e-6
        ):
            raise ValueError(
                f"Spread is not aligned with point size for {symbol}."
            )

        # --------------------------------------------------
        # BROKER FIXED-SPREAD VALIDATION
        # --------------------------------------------------

        if not info.spread_float and info.spread > 0:

            expected_spread = float(info.spread) * point

            if not math.isclose(
                spread,
                expected_spread,
                rel_tol=0.0,
                abs_tol=tolerance
            ):
                raise ValueError(
                    f"Spread specification mismatch for {symbol}: "
                    f"observed={spread}, "
                    f"expected={expected_spread}."
                )

        # --------------------------------------------------
        # QUOTE FRESHNESS
        # --------------------------------------------------

        if age_seconds > max_age_seconds:
            raise RuntimeError(
                f"Stale quote for {symbol}: "
                f"{age_seconds:.2f} seconds old."
            )

        # --------------------------------------------------
        # RETURN VALIDATED QUOTE
        # --------------------------------------------------

        quote = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "spread_points": spread_points,
            "point": point,
            "time": timestamp.isoformat(),
            "age_seconds": round(age_seconds, 3),
            "valid": True,
            "message": "Quote validated successfully.",
            "source": "copy_ticks_range"
        }

        log.info(
            "Valid quote for %s | Bid: %s | Ask: %s | "
            "Spread: %s points | Age: %.3f seconds",
            symbol,
            bid,
            ask,
            spread_points,
            age_seconds
        )

        return quote

    # --------------------------------------------------
    # COMPATIBILITY METHOD
    # --------------------------------------------------

    def get_tick(self, symbol):
        """
        Retains the original get_tick() interface.

        The underlying quote now comes from raw tick data,
        not symbol_info_tick().
        """

        try:
            return self.get_recent_quote(symbol)

        except (RuntimeError, ValueError) as error:
            log.warning(
                "Could not retrieve quote for %s: %s",
                symbol,
                error
            )

            return None

    # --------------------------------------------------
    # CANDLE DATA
    # --------------------------------------------------

    def get_candles(self, symbol, timeframe, count=300):
        """
        Retrieve candles including the currently forming candle.

        The latest candle is identified using
        is_forming_candle=True.
        """

        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        if count <= 0:
            raise ValueError("Candle count must be positive")

        self.enable_symbol(symbol)

        rates = mt5.copy_rates_from_pos(
            symbol,
            TIMEFRAME_MAP[timeframe],
            0,
            int(count)
        )

        if rates is None or len(rates) == 0:
            log.warning(
                "No rates for %s %s: %s",
                symbol,
                timeframe,
                mt5.last_error()
            )

            return pd.DataFrame()

        df = pd.DataFrame(rates)

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

        df["is_forming_candle"] = False

        # The last returned candle is the current candle.
        df.loc[df.index[-1], "is_forming_candle"] = True

        return df

    # --------------------------------------------------
    # COMPLETED CANDLES ONLY
    # --------------------------------------------------

    def get_closed_candles(self, symbol, timeframe, count=300):
        """
        Retrieve completed candles only.

        MT5 position 0 is the current forming candle.
        Starting at position 1 excludes it.
        """

        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        if count <= 0:
            raise ValueError("Candle count must be positive")

        self.enable_symbol(symbol)

        rates = mt5.copy_rates_from_pos(
            symbol,
            TIMEFRAME_MAP[timeframe],
            1,
            int(count)
        )

        if rates is None or len(rates) == 0:
            log.warning(
                "No completed candles for %s %s: %s",
                symbol,
                timeframe,
                mt5.last_error()
            )

            return pd.DataFrame()

        df = pd.DataFrame(rates)

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

        df = df.sort_values(
            "time"
        ).reset_index(drop=True)

        df["is_forming_candle"] = False

        # Basic OHLC integrity checks.
        invalid_rows = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        if invalid_rows.any():
            raise ValueError(
                f"Invalid OHLC data detected for {symbol} {timeframe}"
            )

        return df
