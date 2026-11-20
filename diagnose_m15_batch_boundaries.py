
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

from src.config import load_settings
from src.mt5_connection import MT5Connection
from src.market_data import MarketData, TIMEFRAME_MAP


SYMBOL = "EURUSD.0"
TIMEFRAME = "M15"
EXPECTED_SERVER = "Deriv-Demo"


def print_range_result(label, start, end):
    rates = mt5.copy_rates_range(
        SYMBOL,
        TIMEFRAME_MAP[TIMEFRAME],
        start,
        end,
    )

    if rates is None:
        print(f"{label}: FAILED | {mt5.last_error()}")
        return None

    if len(rates) == 0:
        print(f"{label}: EMPTY")
        return 0

    first = datetime.fromtimestamp(
        int(rates[0]["time"]),
        tz=timezone.utc,
    )

    last = datetime.fromtimestamp(
        int(rates[-1]["time"]),
        tz=timezone.utc,
    )

    print(
        f"{label}: {len(rates)} candles | "
        f"First={first.isoformat()} | "
        f"Last={last.isoformat()}"
    )

    return len(rates)


def main():
    settings = load_settings()

    if settings.get("trading", {}).get("execution_enabled") is not False:
        raise RuntimeError(
            "Diagnostic blocked: execution must remain disabled."
        )

    history = settings["research"]["historical_data"]

    history_days = int(history.get("history_days", 180))
    batch_size = int(history.get("batch_size", 5000))

    connection = MT5Connection()
    market_data = MarketData()
    connected = False

    try:
        terminal, account = connection.connect()
        connected = True

        if EXPECTED_SERVER not in str(account.server):
            raise RuntimeError(
                f"Unexpected server: {account.server}"
            )

        market_data.enable_symbol(SYMBOL)

        # Reconstruct the original collection request.
        end = datetime(2026, 11, 5, 7, 52, 3, 857260,
                       tzinfo=timezone.utc)

        start = end - timedelta(days=history_days)

        batch_window = timedelta(
            seconds=900 * (batch_size - 1)
        )

        print("\n========== ORIGINAL COLLECTION WINDOW ==========")
        print(f"Start: {start.isoformat()}")
        print(f"End: {end.isoformat()}")
        print(f"Batch size: {batch_size}")
        print(f"Batch window: {batch_window}")

        cursor = start
        batch_number = 0
        last_batch_start = None

        while cursor <= end:
            batch_number += 1

            batch_end = min(
                cursor + batch_window,
                end,
            )

            print_range_result(
                f"Original batch {batch_number}",
                cursor,
                batch_end,
            )

            last_batch_start = cursor

            cursor = batch_end + timedelta(seconds=1)

        print("\n========== FINAL BATCH SPLIT TEST ==========")

        split_start = last_batch_start
        split_end = end

        print(f"Final batch start: {split_start.isoformat()}")
        print(f"Final batch end:   {split_end.isoformat()}")

        # First reproduce the complete final batch.
        print_range_result(
            "Complete final batch",
            split_start,
            split_end,
        )

        # Now divide that same range into seven-day windows.
        chunk_start = split_start
        chunk_number = 0
        total_split_candles = 0

        while chunk_start <= split_end:
            chunk_number += 1

            chunk_end = min(
                chunk_start + timedelta(days=7),
                split_end,
            )

            count = print_range_result(
                f"Seven-day chunk {chunk_number}",
                chunk_start,
                chunk_end,
            )

            if count is not None:
                total_split_candles += count

            chunk_start = chunk_end + timedelta(seconds=1)

        print("\n========== SUMMARY ==========")
        print(f"Seven-day total: {total_split_candles}")
        print("No datasets were modified.")
        print("No trades were placed or modified.")

    finally:
        if connected:
            connection.disconnect()

        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()
