from datetime import datetime, timezone
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
    def discover_symbols(self):
        symbols = mt5.symbols_get()
        if symbols is None:
            raise RuntimeError(f"Could not retrieve broker symbols: {mt5.last_error()}")
        return [s.name for s in symbols]

    def resolve_symbol(self, requested, available):
        # Exact match first, then case-insensitive exact match.
        if requested in available:
            return requested
        lookup = {name.upper(): name for name in available}
        if requested.upper() in lookup:
            return lookup[requested.upper()]
        return None

    def enable_symbol(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not enable {symbol}: {mt5.last_error()}")
        return mt5.symbol_info(symbol)

    def get_tick(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log.warning("No tick returned for %s: %s", symbol, mt5.last_error())
            return None
        return {
            "symbol": symbol,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
            "bid": tick.bid,
            "ask": tick.ask,
            "spread_points": (
                (tick.ask - tick.bid) / mt5.symbol_info(symbol).point
                if mt5.symbol_info(symbol) and mt5.symbol_info(symbol).point else None
            ),
        }

    def get_candles(self, symbol, timeframe, count=300):
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(
            symbol, TIMEFRAME_MAP[timeframe], 0, int(count)
        )
        if rates is None or len(rates) == 0:
            log.warning("No rates for %s %s: %s", symbol, timeframe, mt5.last_error())
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        # MT5 position 0 is the currently forming candle. Keep it identifiable;
        # strategy code in later phases must use closed candles only.
        df["is_forming_candle"] = False
        df.loc[df.index[-1], "is_forming_candle"] = True
        return df
