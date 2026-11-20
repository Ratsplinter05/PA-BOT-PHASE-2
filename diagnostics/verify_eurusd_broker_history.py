from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "historical" / "raw" / "EURUSD"

SYMBOL = "EURUSD.0"
TOLERANCE = 1e-8

TIMEFRAMES = {
    "H1": mt5.TIMEFRAME_H1,
    "M15": mt5.TIMEFRAME_M15,
}

OHLC_COLUMNS = ["open", "high", "low", "close"]


def load_stored_data(timeframe):
    path = DATA_DIR / f"EURUSD_{timeframe}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    required = ["time"] + OHLC_COLUMNS

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{timeframe}: Missing required columns: {missing}"
        )

    df = df[required].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    return df.sort_values("time").reset_index(drop=True)


def load_broker_data(timeframe, stored):
    start = stored["time"].min().to_pydatetime()
    end = stored["time"].max().to_pydatetime()

    rates = mt5.copy_rates_range(
        SYMBOL,
        TIMEFRAMES[timeframe],
        start,
        end,
    )

    if rates is None or len(rates) == 0:
        print(f"\n{timeframe}: MT5 returned no historical rates.")
        print(f"MT5 error: {mt5.last_error()}")
        return pd.DataFrame()

    broker = pd.DataFrame(rates)

    broker["time"] = pd.to_datetime(
        broker["time"],
        unit="s",
        utc=True,
    )

    broker = broker[["time"] + OHLC_COLUMNS]

    return broker.sort_values("time").reset_index(drop=True)


def audit_timeframe(timeframe):
    print("\n" + "=" * 85)
    print(f"EURUSD {timeframe} STORED HISTORY AUDIT")
    print("=" * 85)

    stored = load_stored_data(timeframe)
    broker = load_broker_data(timeframe, stored)

    print(f"Stored candles: {len(stored)}")
    print(f"Broker candles returned: {len(broker)}")

    if broker.empty:
        print("RESULT: INCOMPLETE - broker data unavailable.")
        return

    if stored["time"].duplicated().any():
        print("WARNING: Duplicate timestamps in stored data.")

    if broker["time"].duplicated().any():
        print("WARNING: Duplicate timestamps in broker data.")

    comparison = stored.merge(
        broker,
        on="time",
        how="outer",
        suffixes=("_stored", "_broker"),
        indicator=True,
    )

    stored_only = comparison[
        comparison["_merge"] == "left_only"
    ]

    broker_only = comparison[
        comparison["_merge"] == "right_only"
    ]

    matched = comparison[
        comparison["_merge"] == "both"
    ].copy()

    mismatches = []

    for column in OHLC_COLUMNS:
        difference = (
            matched[f"{column}_stored"]
            - matched[f"{column}_broker"]
        ).abs()

        matched[f"{column}_difference"] = difference

        if (difference > TOLERANCE).any():
            mismatches.append(column)

    mismatch_mask = pd.Series(False, index=matched.index)

    for column in OHLC_COLUMNS:
        mismatch_mask |= (
            matched[f"{column}_difference"] > TOLERANCE
        )

    changed = matched[mismatch_mask].copy()

    unchanged = matched[~mismatch_mask]

    print(f"Matching timestamps: {len(matched)}")
    print(f"Unchanged OHLC records: {len(unchanged)}")
    print(f"Changed OHLC records: {len(changed)}")
    print(f"Stored-only timestamps: {len(stored_only)}")
    print(f"Broker-only timestamps: {len(broker_only)}")

    if not changed.empty:
        print("\nFIRST 20 CHANGED CANDLES")
        print("-" * 85)

        display_columns = ["time"]

        for column in OHLC_COLUMNS:
            display_columns.extend([
                f"{column}_stored",
                f"{column}_broker",
                f"{column}_difference",
            ])

        print(
            changed[display_columns]
            .head(20)
            .to_string(index=False)
        )

    if not stored_only.empty:
        print("\nFIRST 10 STORED-ONLY TIMESTAMPS")
        print(stored_only["time"].head(10).to_string(index=False))

    if not broker_only.empty:
        print("\nFIRST 10 BROKER-ONLY TIMESTAMPS")
        print(broker_only["time"].head(10).to_string(index=False))

    print("\nTIMEFRAME RESULT")
    print("-" * 85)

    if (
        changed.empty
        and stored_only.empty
        and broker_only.empty
        and len(matched) == len(stored)
    ):
        print("PASS - Stored history agrees with broker history.")
    else:
        print("REVIEW REQUIRED - Historical differences detected.")


def main():
    print("\n========== EURUSD COMPLETE HISTORY AUDIT ==========")

    if not mt5.initialize(
        path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe",
        timeout=60000,
    ):
        print("MT5 INITIALIZATION FAILED")
        print(mt5.last_error())
        return

    try:
        account = mt5.account_info()

        print(f"Symbol: {SYMBOL}")
        print(
            f"Account server: "
            f"{account.server if account else 'Unknown'}"
        )

        if not mt5.symbol_select(SYMBOL, True):
            print("SYMBOL SELECTION FAILED")
            print(mt5.last_error())
            return

        for timeframe in TIMEFRAMES:
            audit_timeframe(timeframe)

    finally:
        mt5.shutdown()

    print("\n========== HISTORY AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()