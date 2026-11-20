
from datetime import datetime, timezone

import pandas as pd
import MetaTrader5 as mt5

from src.config import load_settings
from src.market_data import MarketData
from src.mt5_connection import MT5Connection


# Broker symbol names previously identified.
# The strategy labels remain unchanged in settings.yaml.
SYMBOL_CANDIDATES = {
    "XAUUSD": ["XAUUSD.0", "XAUUSD"],
    "GBPUSD": ["GBPUSD.0", "GBPUSD"],
    "BTCUSD": ["BTCUSD.0", "BTCUSD"],
    "EURUSD": ["EURUSD.0", "EURUSD"],
    "US30": ["Wall Street 30.0", "US30"],
    "NAS100": ["US Tech 100.0", "NAS100"],
}

OHLC_COLUMNS = ["open", "high", "low", "close"]


def validate_candles(df, requested_count):
    """Independently validate completed candle data."""

    problems = []

    if df.empty:
        return "FAIL", ["No completed candles returned."]

    required_columns = ["time", *OHLC_COLUMNS, "is_forming_candle"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return "FAIL", [
            f"Missing columns: {missing_columns}"
        ]

    if len(df) < requested_count:
        problems.append(
            f"Only {len(df)} of {requested_count} requested candles returned."
        )

    if df["time"].isna().any():
        problems.append("Missing timestamps detected.")

    if not df["time"].is_monotonic_increasing:
        problems.append("Timestamps are not chronological.")

    if df["time"].duplicated().any():
        problems.append("Duplicate timestamps detected.")

    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        problems.append("Timestamp column is not datetime.")

    if df[OHLC_COLUMNS].isna().any().any():
        problems.append("Missing OHLC values detected.")

    if not df["is_forming_candle"].eq(False).all():
        problems.append("A forming candle was included.")

    if not df.empty and not df[OHLC_COLUMNS].empty:
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        if invalid_ohlc.any():
            problems.append("Invalid OHLC relationships detected.")

    if problems:
        return "WARNING", problems

    return "PASS", []


def main():
    settings = load_settings()

    instruments = settings["market"]["instruments"]
    timeframes = settings["market"]["timeframes"]
    candle_count = int(
        settings["market"]["candles_per_timeframe"]
    )

    market_data = MarketData()
    results = []

    print("=" * 75)
    print("PA-MS TRADING BOT | MARKET DATA DIAGNOSTIC")
    print("=" * 75)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Instruments: {len(instruments)}")
    print(f"Timeframes: {len(timeframes)}")
    print(f"Requested candles per test: {candle_count}")
    print("Trading execution: DISABLED")
    print("Order placement: NOT USED")
    print("=" * 75)

    try:
        with MT5Connection():
            print("\nMT5 CONNECTION: SUCCESS")
            print("Discovering broker symbols...\n")

            available_symbols = market_data.discover_symbols()

            print(
                f"Broker symbols discovered: {len(available_symbols)}"
            )

            for instrument in instruments:

                candidates = SYMBOL_CANDIDATES.get(
                    instrument,
                    [instrument]
                )

                broker_symbol = None

                for candidate in candidates:
                    broker_symbol = market_data.resolve_symbol(
                        candidate,
                        available_symbols
                    )

                    if broker_symbol:
                        break

                if broker_symbol is None:
                    for timeframe in timeframes:
                        results.append({
                            "instrument": instrument,
                            "symbol": "NOT FOUND",
                            "timeframe": timeframe,
                            "status": "FAIL",
                            "candles": 0,
                            "details": "Broker symbol could not be resolved."
                        })

                    continue

                print(
                    f"\n{instrument} -> {broker_symbol}"
                )

                try:
                    market_data.enable_symbol(broker_symbol)

                except Exception as error:
                    for timeframe in timeframes:
                        results.append({
                            "instrument": instrument,
                            "symbol": broker_symbol,
                            "timeframe": timeframe,
                            "status": "FAIL",
                            "candles": 0,
                            "details": str(error)
                        })

                    continue

                for timeframe in timeframes:

                    try:
                        df = market_data.get_closed_candles(
                            broker_symbol,
                            timeframe,
                            count=candle_count
                        )

                        status, problems = validate_candles(
                            df,
                            candle_count
                        )

                        details = "; ".join(problems) if problems else "All checks passed."

                        first_time = (
                            df["time"].iloc[0].isoformat()
                            if not df.empty else "N/A"
                        )

                        last_time = (
                            df["time"].iloc[-1].isoformat()
                            if not df.empty else "N/A"
                        )

                        results.append({
                            "instrument": instrument,
                            "symbol": broker_symbol,
                            "timeframe": timeframe,
                            "status": status,
                            "candles": len(df),
                            "details": details,
                            "first_candle": first_time,
                            "last_candle": last_time
                        })

                    except Exception as error:
                        results.append({
                            "instrument": instrument,
                            "symbol": broker_symbol,
                            "timeframe": timeframe,
                            "status": "FAIL",
                            "candles": 0,
                            "details": str(error)
                        })

    except Exception as error:
        print("\nMT5 CONNECTION OR DIAGNOSTIC ERROR")
        print(str(error))

    print("\n")
    print("=" * 100)
    print("MARKET DATA TEST RESULTS")
    print("=" * 100)

    if results:
        report = pd.DataFrame(results)

        display_columns = [
            "instrument",
            "symbol",
            "timeframe",
            "status",
            "candles",
            "details"
        ]

        print(
            report[display_columns].to_string(index=False)
        )

        passed = sum(
            result["status"] == "PASS"
            for result in results
        )

        warnings = sum(
            result["status"] == "WARNING"
            for result in results
        )

        failed = sum(
            result["status"] == "FAIL"
            for result in results
        )

        print("\n")
        print("=" * 75)
        print("FINAL SUMMARY")
        print("=" * 75)
        print(f"Total checks: {len(results)}")
        print(f"PASS: {passed}")
        print(f"WARNING: {warnings}")
        print(f"FAIL: {failed}")
        print("=" * 75)

        report.to_csv(
            "market_data_test_results.csv",
            index=False
        )

        print(
            "\nDetailed report saved to:"
            "\nmarket_data_test_results.csv"
        )

    else:
        print("No instrument/timeframe tests were completed.")

    print("\nDiagnostic finished.")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
