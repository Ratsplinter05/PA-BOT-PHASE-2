
from datetime import datetime, timezone

import pandas as pd

from src.config import load_settings
from src.market_data import MarketData
from src.mt5_connection import MT5Connection
from src.data_validation import CandleValidator


SYMBOL_CANDIDATES = {
    "XAUUSD": ["XAUUSD.0", "XAUUSD"],
    "GBPUSD": ["GBPUSD.0", "GBPUSD"],
    "BTCUSD": ["BTCUSD.0", "BTCUSD"],
    "EURUSD": ["EURUSD.0", "EURUSD"],
    "US30": ["Wall Street 30.0", "US30"],
    "NAS100": ["US Tech 100.0", "NAS100"],
}


def main():

    settings = load_settings()

    instruments = settings["market"]["instruments"]
    timeframes = list(settings["market"]["timeframes"].keys())

    candle_count = int(
        settings["market"]["candles_per_timeframe"]
    )

    market_data = MarketData()
    validator = CandleValidator()

    results = []

    print("=" * 85)
    print("PA-MS TRADING BOT | DATA INTEGRITY DIAGNOSTIC")
    print("=" * 85)

    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Instruments: {len(instruments)}")
    print(f"Timeframes: {len(timeframes)}")
    print(f"Requested candles: {candle_count}")
    print("Trading execution: DISABLED")
    print("Order placement: NOT USED")

    print("=" * 85)

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
                    f"\nChecking {instrument} -> {broker_symbol}"
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

                    print(f"  Validating {timeframe}...")

                    try:

                        candles = market_data.get_closed_candles(
                            broker_symbol,
                            timeframe,
                            count=candle_count
                        )

                        validation = validator.validate(
                            candles,
                            timeframe=timeframe
                        )

                        status = validation["status"]

                        problems = list(validation["errors"])
                        warnings = list(validation["warnings"])

                        if (
                            validation["valid"]
                            and len(candles) < candle_count
                        ):

                            status = "WARNING"

                            warnings.append(
                                f"Only {len(candles)} of "
                                f"{candle_count} requested candles returned."
                            )

                        details = "; ".join(
                            problems + warnings
                        )

                        if not details:
                            details = "All integrity checks passed."

                        first_candle = (
                            candles["time"].iloc[0].isoformat()
                            if not candles.empty else "N/A"
                        )

                        last_candle = (
                            candles["time"].iloc[-1].isoformat()
                            if not candles.empty else "N/A"
                        )

                        results.append({

                            "instrument": instrument,
                            "symbol": broker_symbol,
                            "timeframe": timeframe,
                            "status": status,
                            "candles": len(candles),
                            "first_candle": first_candle,
                            "last_candle": last_candle,
                            "details": details

                        })

                        print(
                            f"    {status} | "
                            f"{len(candles)} candles | "
                            f"{details}"
                        )

                    except Exception as error:

                        results.append({

                            "instrument": instrument,
                            "symbol": broker_symbol,
                            "timeframe": timeframe,
                            "status": "FAIL",
                            "candles": 0,
                            "first_candle": "N/A",
                            "last_candle": "N/A",
                            "details": str(error)

                        })

                        print(f"    FAIL | {error}")

    except Exception as error:

        print("\nMT5 CONNECTION OR DIAGNOSTIC ERROR")
        print(str(error))

        completed = {
            (
                result["instrument"],
                result["timeframe"]
            )
            for result in results
        }

        for instrument in instruments:

            for timeframe in timeframes:

                key = (instrument, timeframe)

                if key not in completed:

                    results.append({

                        "instrument": instrument,
                        "symbol": "NOT CHECKED",
                        "timeframe": timeframe,
                        "status": "FAIL",
                        "candles": 0,
                        "first_candle": "N/A",
                        "last_candle": "N/A",
                        "details": (
                            f"Diagnostic could not complete: {error}"
                        )

                    })

    print("\n")
    print("=" * 105)
    print("DATA INTEGRITY RESULTS")
    print("=" * 105)

    report = pd.DataFrame(results)

    if not report.empty:

        print(
            report.to_string(index=False)
        )

        passed = (report["status"] == "PASS").sum()
        warnings = (report["status"] == "WARNING").sum()
        failed = (report["status"] == "FAIL").sum()

        print("\n")
        print("=" * 75)
        print("FINAL SUMMARY")
        print("=" * 75)

        print(f"Total checks: {len(report)}")
        print(f"PASS: {passed}")
        print(f"WARNING: {warnings}")
        print(f"FAIL: {failed}")

        print("=" * 75)

        report.to_csv(
            "data_integrity_results.csv",
            index=False
        )

        print(
            "\nDetailed report saved to:"
            "\ndata_integrity_results.csv"
        )

    else:

        print("No validation results were generated.")

    print("\nDiagnostic finished.")
    print("No trades were placed.")


if __name__ == "__main__":
    main()
