
from collections import Counter
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

from research.swing_detection.market_structure_engine import (
    MarketStructureEngine,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = (
    PROJECT_ROOT / "data" / "historical" / "raw" / "EURUSD"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "data" / "historical" / "reports"
)

H4_FILE = DATA_DIRECTORY / "EURUSD_H4.parquet"
H1_FILE = DATA_DIRECTORY / "EURUSD_H1.parquet"

OUTPUT_FILE = (
    OUTPUT_DIRECTORY / "EURUSD_historical_alignment_validation.csv"
)

H4_DURATION = timedelta(hours=4)
H1_DURATION = timedelta(hours=1)

ASSESSMENT_INTERVAL = "1h"

TIMEFRAME_CONFIG = {
    "H4": {
        "duration": H4_DURATION,
        "file": H4_FILE,
    },
    "H1": {
        "duration": H1_DURATION,
        "file": H1_FILE,
    },
}

REQUIRED_COLUMNS = {
    "time",
    "open",
    "high",
    "low",
    "close",
}


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_historical_data(path: Path, timeframe: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{timeframe} historical dataset not found: {path}"
        )

    candles = pd.read_parquet(path)

    if candles.empty:
        raise ValueError(
            f"{timeframe} historical dataset is empty."
        )

    missing_columns = REQUIRED_COLUMNS.difference(
        candles.columns
    )

    if missing_columns:
        raise ValueError(
            f"{timeframe} dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    candles = candles.copy()

    candles["time"] = pd.to_datetime(
        candles["time"],
        errors="coerce",
    )

    if candles["time"].isna().any():
        raise ValueError(
            f"{timeframe} contains invalid timestamps."
        )

    if not isinstance(
        candles["time"].dtype,
        pd.DatetimeTZDtype,
    ):
        raise ValueError(
            f"{timeframe} timestamps must be timezone-aware."
        )

    if str(candles["time"].dt.tz) != "UTC":
        raise ValueError(
            f"{timeframe} timestamps must use UTC."
        )

    if candles["time"].duplicated().any():
        raise ValueError(
            f"{timeframe} contains duplicate timestamps."
        )

    if not candles["time"].is_monotonic_increasing:
        raise ValueError(
            f"{timeframe} timestamps are not chronological."
        )

    if "is_forming_candle" in candles.columns:

        if candles["is_forming_candle"].isna().any():
            raise ValueError(
                f"{timeframe} forming-candle field contains missing values."
            )

        if candles["is_forming_candle"].astype(bool).any():
            raise ValueError(
                f"{timeframe} contains forming candles. "
                "Completed candles only are permitted."
            )

    print(
        f"{timeframe}: loaded {len(candles):,} completed candles."
    )

    print(
        f"  First candle: {candles['time'].iloc[0]}"
    )

    print(
        f"  Last candle:  {candles['time'].iloc[-1]}"
    )

    return candles


# ============================================================
# ASSESSMENT TIMELINE
# ============================================================

def build_assessment_timeline(
    h4_candles: pd.DataFrame,
    h1_candles: pd.DataFrame,
) -> list:

    h4_first_available = (
        h4_candles["time"].iloc[0] + H4_DURATION
    )

    h1_first_available = (
        h1_candles["time"].iloc[0] + H1_DURATION
    )

    h4_last_available = (
        h4_candles["time"].iloc[-1] + H4_DURATION
    )

    h1_last_available = (
        h1_candles["time"].iloc[-1] + H1_DURATION
    )

    # Only assess times covered by both datasets.
    start_time = max(
        h4_first_available,
        h1_first_available,
    )

    end_time = min(
        h4_last_available,
        h1_last_available,
    )

    if start_time > end_time:
        raise ValueError(
            "H4 and H1 datasets have no overlapping assessment period."
        )

    assessment_times = pd.date_range(
        start=start_time,
        end=end_time,
        freq=ASSESSMENT_INTERVAL,
    )

    if assessment_times.empty:
        raise ValueError(
            "No historical assessment timestamps could be generated."
        )

    return [
        timestamp.to_pydatetime()
        for timestamp in assessment_times
    ]


# ============================================================
# SNAPSHOT INTEGRITY VALIDATION
# ============================================================

def validate_snapshot_selection(
    selected_snapshot,
    snapshots,
    assessment_time,
    duration,
    timeframe_name,
) -> None:

    eligible_snapshots = [
        snapshot
        for snapshot in snapshots
        if (
            snapshot.as_of_time + duration
            <= assessment_time
        )
    ]

    if not eligible_snapshots:

        if selected_snapshot is not None:
            raise AssertionError(
                f"{timeframe_name} selected a snapshot "
                "when no eligible snapshot existed."
            )

        return

    expected_latest = max(
        eligible_snapshots,
        key=lambda snapshot: snapshot.as_of_time,
    )

    if selected_snapshot is None:
        raise AssertionError(
            f"{timeframe_name} failed to select an eligible snapshot."
        )

    if (
        selected_snapshot.as_of_time
        != expected_latest.as_of_time
    ):
        raise AssertionError(
            f"{timeframe_name} did not select the latest "
            "eligible snapshot."
        )

    known_at = (
        selected_snapshot.as_of_time + duration
    )

    if known_at > assessment_time:
        raise AssertionError(
            f"{timeframe_name} used a snapshot before "
            "its confirmation boundary."
        )


# ============================================================
# REPORT HELPERS
# ============================================================

def get_swing_details(snapshot, swing_type):

    if snapshot is None:
        return None, None

    swing = (
        snapshot.latest_high
        if swing_type == "HIGH"
        else snapshot.latest_low
    )

    if swing is None:
        return None, None

    return (
        swing.classification,
        swing.swing.price,
    )


def build_result_record(
    result,
    assessment_number,
):

    h4 = result.h4_snapshot
    h1 = result.h1_snapshot

    h4_high_class, h4_high_price = get_swing_details(
        h4,
        "HIGH",
    )

    h4_low_class, h4_low_price = get_swing_details(
        h4,
        "LOW",
    )

    h1_high_class, h1_high_price = get_swing_details(
        h1,
        "HIGH",
    )

    h1_low_class, h1_low_price = get_swing_details(
        h1,
        "LOW",
    )

    return {
        "assessment_number": assessment_number,
        "assessment_time": result.assessment_time.isoformat(),

        "alignment_state": result.alignment_state,
        "directional_permission": result.directional_permission,
        "permitted_direction": result.permitted_direction,
        "reason": result.reason,

        "h4_trend": h4.trend if h4 else None,
        "h4_snapshot_as_of": (
            h4.as_of_time.isoformat() if h4 else None
        ),
        "h4_known_at": (
            result.h4_known_at_time.isoformat()
            if result.h4_known_at_time else None
        ),
        "h4_high_classification": h4_high_class,
        "h4_high_price": h4_high_price,
        "h4_low_classification": h4_low_class,
        "h4_low_price": h4_low_price,

        "h1_trend": h1.trend if h1 else None,
        "h1_snapshot_as_of": (
            h1.as_of_time.isoformat() if h1 else None
        ),
        "h1_known_at": (
            result.h1_known_at_time.isoformat()
            if result.h1_known_at_time else None
        ),
        "h1_high_classification": h1_high_class,
        "h1_high_price": h1_high_price,
        "h1_low_classification": h1_low_class,
        "h1_low_price": h1_low_price,
    }


# ============================================================
# MAIN HISTORICAL VALIDATION
# ============================================================

def run_validation():

    print("\n" + "=" * 65)
    print("PA-MS HISTORICAL MULTI-TIMEFRAME ALIGNMENT VALIDATION")
    print("=" * 65)

    print("\n[1] Loading historical datasets")

    h4_candles = load_historical_data(
        H4_FILE,
        "H4",
    )

    h1_candles = load_historical_data(
        H1_FILE,
        "H1",
    )

    print("\n[2] Generating historical market structure")

    engine = MarketStructureEngine()

    h4_analysis = engine.analyze(
        candles=h4_candles,
        timeframe="H4",
    )

    print(
        f"H4 snapshots generated: "
        f"{len(h4_analysis.trend_snapshots):,}"
    )

    h1_analysis = engine.analyze(
        candles=h1_candles,
        timeframe="H1",
    )

    print(
        f"H1 snapshots generated: "
        f"{len(h1_analysis.trend_snapshots):,}"
    )

    print("\n[3] Building historical assessment timeline")

    assessment_times = build_assessment_timeline(
        h4_candles,
        h1_candles,
    )

    print(
        f"Assessment points: {len(assessment_times):,}"
    )

    print(
        f"First assessment: {assessment_times[0]}"
    )

    print(
        f"Last assessment:  {assessment_times[-1]}"
    )

    print("\n[4] Evaluating historical alignment")

    alignment_engine = engine.alignment_engine

    records = []

    integrity_checks = 0

    for assessment_number, assessment_time in enumerate(
        assessment_times,
        start=1,
    ):

        result = alignment_engine.evaluate(
            h4_snapshots=h4_analysis.trend_snapshots,
            h1_snapshots=h1_analysis.trend_snapshots,
            assessment_time=assessment_time,
        )

        # Independently verify the selected H4 snapshot.
        validate_snapshot_selection(
            selected_snapshot=result.h4_snapshot,
            snapshots=h4_analysis.trend_snapshots,
            assessment_time=assessment_time,
            duration=H4_DURATION,
            timeframe_name="H4",
        )

        # Independently verify the selected H1 snapshot.
        validate_snapshot_selection(
            selected_snapshot=result.h1_snapshot,
            snapshots=h1_analysis.trend_snapshots,
            assessment_time=assessment_time,
            duration=H1_DURATION,
            timeframe_name="H1",
        )

        # Verify directional permission.
        if result.directional_permission:

            if result.permitted_direction not in (
                "BULLISH",
                "BEARISH",
            ):
                raise AssertionError(
                    "Directional permission was granted "
                    "without a valid permitted direction."
                )

            if result.h4_snapshot is None or result.h1_snapshot is None:
                raise AssertionError(
                    "Directional permission was granted "
                    "without both timeframe snapshots."
                )

            if not (
                result.h4_snapshot.trend
                == result.h1_snapshot.trend
                == result.permitted_direction
            ):
                raise AssertionError(
                    "Directional permission does not match "
                    "both timeframe trends."
                )

        elif result.permitted_direction is not None:

            raise AssertionError(
                "A direction was assigned without "
                "directional permission."
            )

        records.append(
            build_result_record(
                result,
                assessment_number,
            )
        )

        integrity_checks += 1

        if assessment_number % 250 == 0:
            print(
                f"  Evaluated {assessment_number:,} "
                f"/ {len(assessment_times):,} points"
            )

    # ========================================================
    # SAVE CSV REPORT
    # ========================================================

    print("\n[5] Saving historical validation report")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = pd.DataFrame(records)

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("HISTORICAL ALIGNMENT VALIDATION SUMMARY")
    print("=" * 65)

    print(
        f"\nTotal assessment points: {len(records):,}"
    )

    print(
        f"Integrity checks passed: {integrity_checks:,}"
    )

    print("\nAlignment state distribution:")

    state_counts = Counter(
        record["alignment_state"]
        for record in records
    )

    for state, count in sorted(state_counts.items()):

        percentage = (
            count / len(records)
        ) * 100

        print(
            f"  {state:<22} "
            f"{count:>6,} "
            f"({percentage:>6.2f}%)"
        )

    permitted_count = sum(
        record["directional_permission"]
        for record in records
    )

    bullish_count = sum(
        record["permitted_direction"] == "BULLISH"
        for record in records
    )

    bearish_count = sum(
        record["permitted_direction"] == "BEARISH"
        for record in records
    )

    blocked_count = len(records) - permitted_count

    print("\nDirectional permission:")

    print(f"  Permission granted: {permitted_count:,}")
    print(f"  Bullish permission: {bullish_count:,}")
    print(f"  Bearish permission: {bearish_count:,}")
    print(f"  Permission blocked: {blocked_count:,}")

    print("\nScenario coverage:")

    required_scenarios = [
        "ALIGNED_BULLISH",
        "ALIGNED_BEARISH",
        "CONFLICT",
        "H4_MIXED",
        "H1_MIXED",
        "H4_UNCONFIRMED",
        "H1_UNCONFIRMED",
        "H4_MISSING",
        "H1_MISSING",
    ]

    for scenario in required_scenarios:

        count = state_counts.get(scenario, 0)

        status = (
            "OBSERVED"
            if count > 0
            else "NOT OBSERVED"
        )

        print(
            f"  {scenario:<22} "
            f"{count:>6,}  {status}"
        )

    print("\nConfirmation-time integrity:")

    print(
        "  All selected H4 snapshots were available "
        "at their assessment times."
    )

    print(
        "  All selected H1 snapshots were available "
        "at their assessment times."
    )

    print(
        "  Latest eligible snapshots were selected."
    )

    print(
        "  Directional permission matched both trends."
    )

    print("\nReport saved to:")

    print(OUTPUT_FILE)

    print("\n" + "=" * 65)
    print("HISTORICAL ALIGNMENT VALIDATION COMPLETED")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_validation()
