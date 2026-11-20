
from pathlib import Path

import pandas as pd

from research.historical_validation import HistoricalDataValidator


INSTRUMENT = "EURUSD"

DATA_DIRECTORY = (
    Path("data")
    / "historical"
    / "raw"
    / INSTRUMENT
)

TIMEFRAMES = {
    "M15": 900,
    "M5": 300,
}


def load_dataset(timeframe):
    path = (
        DATA_DIRECTORY
        / f"{INSTRUMENT}_{timeframe}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    data = pd.read_parquet(path)

    data["time"] = pd.to_datetime(
        data["time"],
        utc=True,
    )

    return data, path


def classify_gap(previous_time, next_time):
    duration = next_time - previous_time

    weekend_gap = (
        previous_time.weekday() == 4
        and next_time.weekday() in (6, 0)
        and duration.total_seconds() >= 24 * 3600
    )

    if weekend_gap:
        return "POSSIBLE_WEEKEND_CLOSURE"

    return "REQUIRES_REVIEW"


def audit_timeframe(timeframe, data):
    expected_seconds = TIMEFRAMES[timeframe]

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

        duration = next_time - previous_time

        if duration.total_seconds() <= expected_seconds:
            continue

        gaps.append({
            "previous": previous_time,
            "next": next_time,
            "duration_hours": (
                duration.total_seconds() / 3600
            ),
            "classification": classify_gap(
                previous_time,
                next_time,
            ),
        })

    return timestamps, gaps


def print_timeframe_report(
    timeframe,
    data,
    timestamps,
    gaps,
):
    print(f"\n========== {timeframe} DATASET AUDIT ==========")

    print(f"Candles: {len(data)}")
    print(f"Unique timestamps: {len(timestamps)}")
    print(f"First candle: {timestamps.iloc[0]}")
    print(f"Last candle: {timestamps.iloc[-1]}")

    print(
        "Duplicate timestamps: "
        f"{data['time'].duplicated().sum()}"
    )

    print(
        "Chronological: "
        f"{timestamps.is_monotonic_increasing}"
    )

    print(f"Potential gaps: {len(gaps)}")

    weekend_count = sum(
        gap["classification"] == "POSSIBLE_WEEKEND_CLOSURE"
        for gap in gaps
    )

    review_count = sum(
        gap["classification"] == "REQUIRES_REVIEW"
        for gap in gaps
    )

    print(f"Possible weekend gaps: {weekend_count}")
    print(f"Gaps requiring review: {review_count}")

    print("\nLargest gaps:")

    largest_gaps = sorted(
        gaps,
        key=lambda item: item["duration_hours"],
        reverse=True,
    )[:15]

    if not largest_gaps:
        print("No gaps detected.")

    for gap in largest_gaps:
        print(
            f"{gap['previous']} -> {gap['next']} | "
            f"{gap['duration_hours']:.2f} hours | "
            f"{gap['classification']}"
        )


def audit_cross_timeframe(m15, m5):
    print("\n========== M5 / M15 ALIGNMENT AUDIT ==========")

    m15_times = set(m15["time"])
    m5_times = set(m5["time"])

    if not m15_times or not m5_times:
        raise RuntimeError(
            "One or both datasets contain no timestamps."
        )

    m15_start = min(m15_times)
    m15_end = max(m15_times)

    m5_start = min(m5_times)
    m5_end = max(m5_times)

    print(f"M15 range: {m15_start} -> {m15_end}")
    print(f"M5 range:  {m5_start} -> {m5_end}")

    end_difference = m5_end - m15_end

    print(
        "Difference between latest candles: "
        f"{end_difference}"
    )

    m5_timestamp_series = pd.Series(
        sorted(m5_times)
    )

    m5_parent_times = set(
        m5_timestamp_series.dt.floor("15min")
    )

    matching_m15 = m5_parent_times.intersection(
        m15_times
    )

    print(f"M15 candles: {len(m15_times)}")
    print(f"M5 parent intervals: {len(m5_parent_times)}")
    print(
        "M5 parent intervals matching M15: "
        f"{len(matching_m15)}"
    )

    unmatched = m5_parent_times - m15_times

    print(
        "M5 parent intervals without M15 candle: "
        f"{len(unmatched)}"
    )

    latest_m5_parent = max(m5_parent_times)

    print(f"Latest M5 parent interval: {latest_m5_parent}")

    if latest_m5_parent not in m15_times:
        print(
            "WARNING: Latest M5 interval has no matching M15 candle."
        )

    if end_difference.total_seconds() > 3600:
        print(
            "WARNING: M15 is more than one hour behind M5."
        )

    print("\nLatest unmatched M5 parent intervals:")

    for timestamp in sorted(unmatched)[-10:]:
        print(timestamp)


def main():
    print("EURUSD M15/M5 HISTORICAL DATA AUDIT")
    print("Read-only audit. No dataset files will be changed.")

    validator = HistoricalDataValidator()

    datasets = {}
    audit_results = {}

    for timeframe in ("M15", "M5"):
        data, path = load_dataset(timeframe)

        datasets[timeframe] = data

        print(f"\nLoaded: {path}")

        validation = validator.validate(
            data,
            timeframe,
        )

        print(
            f"Validator status: {validation['status']}"
        )
        print(
            f"Validator valid: {validation['valid']}"
        )

        timestamps, gaps = audit_timeframe(
            timeframe,
            data,
        )

        audit_results[timeframe] = (
            timestamps,
            gaps,
        )

        print_timeframe_report(
            timeframe,
            data,
            timestamps,
            gaps,
        )

    audit_cross_timeframe(
        datasets["M15"],
        datasets["M5"],
    )

    print("\n========== AUDIT COMPLETED ==========")
    print("No source datasets were modified.")


if __name__ == "__main__":
    main()
