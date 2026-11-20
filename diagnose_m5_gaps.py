
from pathlib import Path

import pandas as pd


INSTRUMENT = "EURUSD"
TIMEFRAME = "M5"
EXPECTED_SECONDS = 300
DISPLAY_CANDLES = 3

DATA_PATH = (
    Path("data")
    / "historical"
    / "raw"
    / INSTRUMENT
    / f"{INSTRUMENT}_{TIMEFRAME}.parquet"
)

M15_PATH = (
    Path("data")
    / "historical"
    / "raw"
    / INSTRUMENT
    / "EURUSD_M15.parquet"
)


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    data = pd.read_parquet(path)

    data["time"] = pd.to_datetime(
        data["time"],
        utc=True,
    )

    return data.sort_values(
        "time"
    ).reset_index(drop=True)


def describe_timestamp(timestamp):
    return (
        f"{timestamp.isoformat()} | "
        f"{timestamp.strftime('%A')} | "
        f"{timestamp.strftime('%H:%M UTC')}"
    )


def find_gaps(data):
    timestamps = (
        data["time"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    gaps = []

    for index in range(1, len(timestamps)):
        previous_time = timestamps.iloc[index - 1]
        next_time = timestamps.iloc[index]

        difference = (
            next_time - previous_time
        ).total_seconds()

        if difference <= EXPECTED_SECONDS:
            continue

        missing_intervals = int(
            difference // EXPECTED_SECONDS
        ) - 1

        gaps.append({
            "previous": previous_time,
            "next": next_time,
            "duration_seconds": difference,
            "duration_minutes": difference / 60,
            "missing_intervals": missing_intervals,
            "previous_index": index - 1,
            "next_index": index,
        })

    return timestamps, gaps


def print_candle_window(data, gap, label, index):
    print(f"\n{label}")

    start = max(
        0,
        index - DISPLAY_CANDLES + 1,
    )

    end = min(
        len(data),
        index + DISPLAY_CANDLES + 1,
    )

    window = data.iloc[start:end]

    for _, row in window.iterrows():
        marker = ""

        if row["time"] == gap["previous"]:
            marker = "  <-- LAST BEFORE GAP"

        elif row["time"] == gap["next"]:
            marker = "  <-- FIRST AFTER GAP"

        print(
            f"{row['time'].isoformat()} | "
            f"O={row['open']} | "
            f"H={row['high']} | "
            f"L={row['low']} | "
            f"C={row['close']}"
            f"{marker}"
        )


def inspect_m15_parent(gap, m15_timestamps):
    previous_parent = gap["previous"].floor("15min")
    next_parent = gap["next"].floor("15min")

    print("\nM15 PARENT INTERVAL CHECK")

    print(
        f"Previous M5 candle parent: "
        f"{previous_parent}"
    )

    print(
        f"Next M5 candle parent: "
        f"{next_parent}"
    )

    print(
        "Previous parent exists in M15: "
        f"{previous_parent in m15_timestamps}"
    )

    print(
        "Next parent exists in M15: "
        f"{next_parent in m15_timestamps}"
    )

    if previous_parent == next_parent:
        print(
            "Both M5 candles belong to the same M15 interval."
        )
    else:
        print(
            "The gap crosses M15 parent intervals."
        )


def main():
    print("\n========== M5 GAP DIAGNOSTIC ==========")
    print("Read-only investigation.")
    print("No datasets will be modified.")
    print("No MT5 connection or trading operations.")

    m5 = load_data(DATA_PATH)
    m15 = load_data(M15_PATH)

    m15_timestamps = set(m15["time"])

    timestamps, gaps = find_gaps(m5)

    print(f"\nM5 dataset: {DATA_PATH}")
    print(f"M5 candles: {len(m5)}")
    print(f"Unique timestamps: {len(timestamps)}")

    print(f"\nM15 dataset: {M15_PATH}")
    print(f"M15 candles: {len(m15)}")

    print(f"\nTotal M5 gaps detected: {len(gaps)}")

    review_gaps = [
        gap for gap in gaps
        if not (
            gap["previous"].weekday() == 4
            and gap["next"].weekday() in (6, 0)
            and gap["duration_seconds"] >= 24 * 3600
        )
    ]

    print(
        f"Gaps requiring review under current audit rules: "
        f"{len(review_gaps)}"
    )

    if not review_gaps:
        print("\nNo M5 gaps require review.")
        return

    for number, gap in enumerate(review_gaps, start=1):

        previous = gap["previous"]
        following = gap["next"]

        print(
            f"\n{'=' * 65}"
        )

        print(f"GAP {number} OF {len(review_gaps)}")

        print(
            f"Last candle before gap: "
            f"{describe_timestamp(previous)}"
        )

        print(
            f"First candle after gap: "
            f"{describe_timestamp(following)}"
        )

        print(
            f"Gap duration: "
            f"{gap['duration_minutes']:.2f} minutes"
        )

        print(
            f"Expected missing M5 intervals: "
            f"{gap['missing_intervals']}"
        )

        print(
            f"Previous weekday: {previous.strftime('%A')}"
        )

        print(
            f"Next weekday: {following.strftime('%A')}"
        )

        previous_index = gap["previous_index"]
        next_index = gap["next_index"]

        print_candle_window(
            m5,
            gap,
            "CANDLES BEFORE THE GAP",
            previous_index,
        )

        print_candle_window(
            m5,
            gap,
            "CANDLES AFTER THE GAP",
            next_index,
        )

        inspect_m15_parent(
            gap,
            m15_timestamps,
        )

    print(
        f"\n{'=' * 65}"
    )

    print("\n========== DIAGNOSTIC SUMMARY ==========")
    print(f"M5 candles inspected: {len(m5)}")
    print(f"Total gaps: {len(gaps)}")
    print(f"Gaps requiring review: {len(review_gaps)}")
    print("\nDiagnostic completed.")
    print("No source datasets were modified.")


if __name__ == "__main__":
    main()
