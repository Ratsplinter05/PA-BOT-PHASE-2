from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "historical" / "raw" / "EURUSD"

TARGET = pd.Timestamp("2026-11-03 16:00:00", tz="UTC")

def load_timeframe(timeframe):
    path = DATA_DIR / f"EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(path)

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    return df

def print_records(title, df):
    print(f"\n{title}")
    print("=" * 75)

    if df.empty:
        print("No records found.")
        return

    columns = ["time", "open", "high", "low", "close"]
    print(df[columns].to_string(index=False))

def main():
    h1 = load_timeframe("H1")
    m15 = load_timeframe("M15")

    h1_parent = h1.loc[h1["time"] == TARGET]
    m15_children = m15.loc[
        (m15["time"] >= TARGET)
        & (m15["time"] < TARGET + pd.Timedelta(hours=1))
    ].copy()

    print("\n========== EURUSD H1/M15 DISCREPANCY INSPECTION ==========")
    print(f"Target H1 candle: {TARGET}")

    print_records("H1 PARENT CANDLE", h1_parent)
    print_records("FOUR M15 CHILD CANDLES", m15_children)

    if len(h1_parent) != 1:
        print(f"\nExpected exactly one H1 candle, found {len(h1_parent)}.")
        return

    if len(m15_children) != 4:
        print(f"\nExpected four M15 candles, found {len(m15_children)}.")
        return

    parent = h1_parent.iloc[0]

    reconstructed = {
        "open": m15_children.iloc[0]["open"],
        "high": m15_children["high"].max(),
        "low": m15_children["low"].min(),
        "close": m15_children.iloc[-1]["close"],
    }

    print("\nRECONSTRUCTED H1 FROM M15")
    print("=" * 75)

    for field, child_value in reconstructed.items():
        parent_value = float(parent[field])
        child_value = float(child_value)
        difference = parent_value - child_value

        print(
            f"{field.upper():<8} "
            f"H1={parent_value:.8f} | "
            f"M15={child_value:.8f} | "
            f"Difference={difference:.8f}"
        )

    print("\nM15 CANDLE TIMESTAMPS")
    print("=" * 75)

    for timestamp in m15_children["time"]:
        print(timestamp)

    print("\n========== INSPECTION COMPLETE ==========")

if __name__ == "__main__":
    main()
