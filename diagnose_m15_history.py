
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

from src.config import load_settings
from src.logger import get_logger
from src.mt5_connection import MT5Connection
from src.market_data import MarketData, TIMEFRAME_MAP


log = get_logger("M15_HISTORY_DIAGNOSTIC")

SYMBOL = "EURUSD.0"
TIMEFRAMES = ("M15", "M5")
EXPECTED_DEMO_SERVER = "Deriv-Demo"


def main():
    settings = load_settings()

    if settings.get("trading", {}).get("execution_enabled") is not False:
        raise RuntimeError(
            "Diagnostic blocked: trade execution must be disabled."
        )

    connection = MT5Connection()
    market_data = MarketData()
    connected = False

    try:
        terminal, account = connection.connect()
        connected = True

        server = str(account.server)

        print("\n========== CONNECTION ==========")
        print(f"Terminal: {terminal.name}")
        print(f"Server: {server}")

        if EXPECTED_DEMO_SERVER not in server:
            raise RuntimeError(
                f"Expected Deriv-Demo, received {server}"
            )

        market_data.enable_symbol(SYMBOL)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=3)

        print("\n========== REQUEST WINDOW ==========")
        print(f"Start: {start.isoformat()}")
        print(f"End:   {end.isoformat()}")

        for timeframe in TIMEFRAMES:
            print(
                f"\n========== {timeframe} DIAGNOSTIC =========="
            )

            timeframe_constant = TIMEFRAME_MAP[timeframe]

            print("\n--- Method 1: copy_rates_range ---")

            range_rates = mt5.copy_rates_range(
                SYMBOL,
                timeframe_constant,
                start,
                end,
            )

            if range_rates is None:
                print(f"Result: None")
                print(f"MT5 error: {mt5.last_error()}")
            else:
                print(f"Returned candles: {len(range_rates)}")

                if len(range_rates) > 0:
                    first = datetime.fromtimestamp(
                        int(range_rates[0]["time"]),
                        tz=timezone.utc,
                    )

                    last = datetime.fromtimestamp(
                        int(range_rates[-1]["time"]),
                        tz=timezone.utc,
                    )

                    print(f"First candle: {first.isoformat()}")
                    print(f"Last candle:  {last.isoformat()}")

                    print("Last 5 candles:")

                    for rate in range_rates[-5:]:
                        timestamp = datetime.fromtimestamp(
                            int(rate["time"]),
                            tz=timezone.utc,
                        )

                        print(
                            f"{timestamp.isoformat()} | "
                            f"Open={rate['open']} | "
                            f"High={rate['high']} | "
                            f"Low={rate['low']} | "
                            f"Close={rate['close']}"
                        )

                print(f"MT5 status: {mt5.last_error()}")

            print("\n--- Method 2: copy_rates_from_pos ---")

            position_rates = mt5.copy_rates_from_pos(
                SYMBOL,
                timeframe_constant,
                0,
                500,
            )

            if position_rates is None:
                print("Result: None")
                print(f"MT5 error: {mt5.last_error()}")

            else:
                print(
                    f"Returned candles: {len(position_rates)}"
                )

                if len(position_rates) > 0:
                    latest = position_rates[-1]

                    latest_time = datetime.fromtimestamp(
                        int(latest["time"]),
                        tz=timezone.utc,
                    )

                    print(
                        "Latest returned candle: "
                        f"{latest_time.isoformat()}"
                    )

                    print("Last 5 candles:")

                    for rate in position_rates[-5:]:
                        timestamp = datetime.fromtimestamp(
                            int(rate["time"]),
                            tz=timezone.utc,
                        )

                        print(
                            f"{timestamp.isoformat()} | "
                            f"Open={rate['open']} | "
                            f"Close={rate['close']}"
                        )

                print(f"MT5 status: {mt5.last_error()}")

        print("\n========== DIAGNOSTIC COMPLETED ==========")
        print("No historical datasets were modified.")
        print("No trades were placed or modified.")

    finally:
        if connected:
            connection.disconnect()

        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()
