from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

from src.config import load_settings
from src.logger import get_logger
from src.market_data import MarketData, TIMEFRAME_MAP


log = get_logger("HISTORICAL_DATA")


TIMEFRAME_SECONDS = {
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
}


class HistoricalDataCollector:
    """
    Retrieves historical OHLC candle data from MT5.

    Features:
    - Uses timezone-aware UTC date ranges.
    - Retrieves historical candles in manageable batches.
    - Supports configurable incomplete-candle exclusion.
    - Identifies incomplete candles when exclusion is disabled.
    - Preserves duplicate timestamps for later validation.
    - Provides collection metadata for research reporting.
    - Does not place or modify trades.

    MT5 must already be connected before collection.
    """

    def __init__(self):
        self.settings = load_settings()
        self.config = self.settings["research"]["historical_data"]

        self.market_data = MarketData()

        self.batch_size = int(
            self.config.get("batch_size", 5000)
        )

        if self.batch_size < 2:
            raise ValueError(
                "Historical batch_size must be at least 2."
            )

        if not self.config.get("enabled", False):
            raise RuntimeError(
                "Historical data collection is disabled."
            )

        self.exclude_forming_candles = self.config.get(
            "exclude_forming_candles",
            True
        )

        if not isinstance(self.exclude_forming_candles, bool):
            raise ValueError(
                "exclude_forming_candles must be True or False."
            )

    @staticmethod
    def _validate_datetime(value, name):
        """Validate and convert a datetime to UTC."""

        if not isinstance(value, datetime):
            raise TypeError(
                f"{name} must be a datetime object."
            )

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{name} must be timezone-aware."
            )

        return value.astimezone(timezone.utc)

    def collect(self, symbol, timeframe, start, end):
        """
        Collect historical candles for one symbol and timeframe.

        Returns:
            pandas.DataFrame containing retrieved candles.

        This interface is retained for backward compatibility.
        """

        result, _ = self._collect_internal(
            symbol,
            timeframe,
            start,
            end,
        )

        return result

    def collect_with_metadata(self, symbol, timeframe, start, end):
        """
        Collect historical candles and return collection metadata.

        Returns:
            tuple:
                - pandas.DataFrame containing retained candles.
                - dict containing collection metadata.
        """

        return self._collect_internal(
            symbol,
            timeframe,
            start,
            end,
        )

    def _collect_internal(self, symbol, timeframe, start, end):
        """Shared collection implementation for both public methods."""

        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}"
            )

        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(
                f"Historical collection is not configured "
                f"for timeframe: {timeframe}"
            )

        configured_timeframes = self.settings["market"]["timeframes"]

        if timeframe not in configured_timeframes:
            raise ValueError(
                f"Timeframe {timeframe} is not configured "
                f"in settings.yaml."
            )

        start = self._validate_datetime(start, "start")
        end = self._validate_datetime(end, "end")

        if start >= end:
            raise ValueError(
                "Start time must be earlier than end time."
            )

        collection_time = datetime.now(timezone.utc)

        if end > collection_time:
            raise ValueError(
                "End time cannot be in the future."
            )

        timeframe_seconds = TIMEFRAME_SECONDS[timeframe]

        log.info(
            "Starting historical collection | "
            "Symbol=%s | Timeframe=%s | Start=%s | End=%s",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat()
        )

        # Verify that the broker symbol can be enabled.
        symbol_enabled = self.market_data.enable_symbol(symbol)

        if not symbol_enabled:
            raise RuntimeError(
                f"Unable to enable MT5 broker symbol: {symbol}"
            )

        batch_window = timedelta(
            seconds=timeframe_seconds * (self.batch_size - 1)
        )

        cursor = start
        batches = []
        batch_number = 0
        raw_retrieved_candles = 0

        while cursor <= end:

            batch_number += 1

            batch_end = min(
                cursor + batch_window,
                end
            )

            rates = mt5.copy_rates_range(
                symbol,
                TIMEFRAME_MAP[timeframe],
                cursor,
                batch_end
            )

            if rates is None:
                raise RuntimeError(
                    f"MT5 historical retrieval failed for "
                    f"{symbol} {timeframe}. "
                    f"Batch {batch_number}. "
                    f"Error: {mt5.last_error()}"
                )

            if len(rates) > 0:

                df = pd.DataFrame(rates)

                raw_retrieved_candles += len(df)

                df["time"] = pd.to_datetime(
                    df["time"],
                    unit="s",
                    utc=True
                )

                batches.append(df)

                log.info(
                    "Retrieved batch %s | %s candles",
                    batch_number,
                    len(df)
                )

            else:

                log.warning(
                    "Empty historical batch | "
                    "Symbol=%s | Timeframe=%s | Batch=%s",
                    symbol,
                    timeframe,
                    batch_number
                )

            # MT5 date-range retrieval uses inclusive boundaries.
            cursor = batch_end + timedelta(seconds=1)

        if batches:
            result = pd.concat(
                batches,
                ignore_index=True
            )
        else:
            result = pd.DataFrame()

        if not result.empty:

            # Restrict candle opening times to the requested range.
            in_range_mask = (
                (result["time"] >= start)
                & (result["time"] <= end)
            )

            in_range_candles = int(in_range_mask.sum())
            out_of_range_candles = int(
                len(result) - in_range_candles
            )

            result = result.loc[in_range_mask].copy()

            candle_close_time = (
                result["time"]
                + pd.to_timedelta(
                    timeframe_seconds,
                    unit="s"
                )
            )

            forming_mask = candle_close_time > collection_time
            within_range_mask = candle_close_time <= end

            if self.exclude_forming_candles:

                completed_mask = (
                    within_range_mask
                    & (~forming_mask)
                )

                result = result.loc[completed_mask].copy()

                result["is_forming_candle"] = False

            else:

                result["is_forming_candle"] = (
                    forming_mask.loc[result.index].values
                )

            result = result.sort_values(
                "time"
            ).reset_index(drop=True)

        else:
            in_range_candles = 0
            out_of_range_candles = 0
            result = pd.DataFrame()

        retained_candles = len(result)

        excluded_candles = (
            in_range_candles - retained_candles
        )

        first_candle = (
            result["time"].iloc[0].isoformat()
            if not result.empty
            else None
        )

        last_candle = (
            result["time"].iloc[-1].isoformat()
            if not result.empty
            else None
        )

        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "collection_time": collection_time.isoformat(),
            "batch_count": batch_number,
            "raw_retrieved_candles": raw_retrieved_candles,
            "in_range_candles": in_range_candles,
            "out_of_range_candles": out_of_range_candles,
            "excluded_candles": excluded_candles,
            "retained_candles": retained_candles,
            "first_candle": first_candle,
            "last_candle": last_candle,
            "exclude_forming_candles": self.exclude_forming_candles,
        }

        log.info(
            "Historical collection completed | "
            "Symbol=%s | Timeframe=%s | "
            "Batches=%s | Retrieved=%s | "
            "InRange=%s | Retained=%s | Excluded=%s",
            symbol,
            timeframe,
            batch_number,
            raw_retrieved_candles,
            in_range_candles,
            retained_candles,
            excluded_candles
        )

        if not result.empty:

            log.info(
                "Historical coverage | First=%s | Last=%s",
                first_candle,
                last_candle
            )

        else:

            log.warning(
                "No retained historical candles for %s %s.",
                symbol,
                timeframe
            )

        return result, metadata
