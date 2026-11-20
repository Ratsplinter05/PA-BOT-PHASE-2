
import math
from datetime import datetime, timezone

import pandas as pd

from src.config import load_settings
from src.market_data import MarketData
from src.mt5_connection import MT5Connection


SYMBOL_CANDIDATES = {
    "XAUUSD": ["XAUUSD.0", "XAUUSD"],
    "GBPUSD": ["GBPUSD.0", "GBPUSD"],
    "BTCUSD": ["BTCUSD.0", "BTCUSD"],
    "EURUSD": ["EURUSD.0", "EURUSD"],
    "US30": ["Wall Street 30.0", "US30"],
    "NAS100": ["US Tech 100.0", "NAS100"],
}

LOOKBACK_SECONDS = 30
MAX_AGE_SECONDS = 10


def main():

    settings = load_settings()
    instruments = settings["market"]["instruments"]

    market_data = MarketData()
    results = []

    print("=" * 90)
    print("PA-MS TRADING BOT | LIVE MARKET QUOTE DIAGNOSTIC")
    print("=" * 90)

    print(
        f"Started: {datetime.now(timezone.utc).isoformat()}"
    )

    print(f"Instruments: {len(instruments)}")
    print(f"Tick lookback: {LOOKBACK_SECONDS} seconds")
    print(f"Maximum quote age: {MAX_AGE_SECONDS} seconds")
    print("Trading execution: DISABLED")
    print("Order placement: NOT USED")

    print("=" * 90)

    try:

        with MT5Connection():

            print("\nMT5 CONNECTION: SUCCESS")

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

                    results.append({
                        "instrument": instrument,
                        "symbol": "NOT FOUND",
                        "status": "FAIL",
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "spread_points": None,
                        "age_seconds": None,
                        "details": "Broker symbol could not be resolved."
                    })

                    continue

                print(f"\nChecking {instrument} -> {broker_symbol}")

                try:

                    quote = market_data.get_recent_quote(
                        broker_symbol,
                        lookback_seconds=LOOKBACK_SECONDS,
                        max_age_seconds=MAX_AGE_SECONDS
                    )

                    bid = quote["bid"]
                    ask = quote["ask"]
                    spread = quote["spread"]
                    spread_points = quote["spread_points"]
                    point = quote["point"]
                    age = quote["age_seconds"]

                    problems = []

                    if not all(
                        math.isfinite(value)
                        for value in [bid, ask, spread, point]
                    ):
                        problems.append("Non-finite price or spread value.")

                    if bid <= 0 or ask <= 0:
                        problems.append("Invalid Bid or Ask.")

                    if ask < bid:
                        problems.append("Ask is below Bid.")

                    if point <= 0:
                        problems.append("Invalid instrument point size.")

                    if not math.isclose(
                        spread,
                        ask - bid,
                        rel_tol=1e-9,
                        abs_tol=1e-12
                    ):
                        problems.append("Spread calculation mismatch.")

                    if age > MAX_AGE_SECONDS:
                        problems.append("Quote is stale.")

                    if not quote["valid"]:
                        problems.append(quote["message"])
                    if any(
                        "stale" in problem.lower()
                        for problem in problems
                    ) and not any(
                        problem != "Quote is stale."
                        for problem in problems
                    ):
                        status = "WARNING"

                    elif problems:
                        status = "FAIL"

                    else:
                        status = "PASS"

                    details = (
                        "; ".join(problems)
                        if problems
                        else "Live quote validated successfully."
                    )

                    results.append({
                        "instrument": instrument,
                        "symbol": broker_symbol,
                        "status": status,
                        "bid": bid,
                        "ask": ask,
                        "spread": spread,
                        "spread_points": spread_points,
                        "age_seconds": age,
                        "timestamp_utc": quote["time"],
                        "details": details
                    })

                except RuntimeError as error:

                    message = str(error)
                    message_lower = message.lower()

                    if (
                        "no recent quote ticks" in message_lower
                        or "stale quote" in message_lower
                    ):
                        status = "WARNING"
                    else:
                        status = "FAIL"

                    results.append({
                        "instrument": instrument,
                        "symbol": broker_symbol,
                        "status": status,
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "spread_points": None,
                        "age_seconds": None,
                        "details": message
                    })

                    results.append({
                        "instrument": instrument,
                        "symbol": broker_symbol,
                        "status": status,
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "spread_points": None,
                        "age_seconds": None,
                        "details": message
                    })

                except Exception as error:

                    results.append({
                        "instrument": instrument,
                        "symbol": broker_symbol,
                        "status": "FAIL",
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "spread_points": None,
                        "age_seconds": None,
                        "details": str(error)
                    })

    except Exception as error:

        print("\nMT5 CONNECTION OR DIAGNOSTIC ERROR")
        print(str(error))

        completed_instruments = {
            item["instrument"] for item in results
        }

        for instrument in instruments:

            if instrument not in completed_instruments:

                results.append({
                    "instrument": instrument,
                    "symbol": "NOT CHECKED",
                    "status": "FAIL",
                    "bid": None,
                    "ask": None,
                    "spread": None,
                    "spread_points": None,
                    "age_seconds": None,
                    "details": (
                        f"Diagnostic could not complete: {error}"
                    )
                })

    print("\n")
    print("=" * 110)
    print("LIVE QUOTE TEST RESULTS")
    print("=" * 110)

    if results:

        report = pd.DataFrame(results)

        display_columns = [
            "instrument",
            "symbol",
            "status",
            "bid",
            "ask",
            "spread",
            "spread_points",
            "age_seconds",
            "details"
        ]

        print(
            report[display_columns].to_string(index=False)
        )

        passed = sum(
            item["status"] == "PASS"
            for item in results
        )

        warnings = sum(
            item["status"] == "WARNING"
            for item in results
        )

        failed = sum(
            item["status"] == "FAIL"
            for item in results
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
            "live_quote_test_results.csv",
            index=False
        )

        print(
            "\nDetailed report saved to:"
            "\nlive_quote_test_results.csv"
        )

    else:

        print("No quote tests were completed.")

    print("\nDiagnostic finished.")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
