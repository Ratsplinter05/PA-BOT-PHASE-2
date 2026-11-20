
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import load_settings
from src.logger import get_logger

from research.historical_storage import HistoricalDataStorage
from research.swing_detection.alignment_engine import AlignmentEngine
from research.swing_detection.trend_models import TrendSnapshot


log = get_logger("ALIGNMENT_ANALYSIS")

INSTRUMENT = "EURUSD"

REFERENCE_ASSESSMENT_TIME = datetime(
    2026, 11, 3, 17, 0, tzinfo=timezone.utc
)

VALID_TRENDS = {
    "BULLISH",
    "BEARISH",
    "MIXED",
    "UNCONFIRMED",
}

TIMEFRAME_DURATIONS = {
    "H4": timedelta(hours=4),
    "H1": timedelta(hours=1),
}


def latest_report(reports_directory, pattern):
    """Return the most recently modified report matching a pattern."""

    matches = list(reports_directory.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No report found matching: {pattern}"
        )

    return max(matches, key=lambda path: path.stat().st_mtime)


def load_trend_report(path, timeframe):
    """Load and validate a historical trend report."""

    df = pd.read_csv(path)

    required_columns = {
        "confirmation_candle_open_time",
        "known_at_time",
        "trend",
        "reason",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{timeframe} report is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{timeframe} trend report is empty.")

    if "instrument" in df.columns:
        instruments = set(
            df["instrument"].dropna().astype(str).str.upper()
        )

        if instruments and instruments != {INSTRUMENT}:
            raise ValueError(
                f"Unexpected instruments in {timeframe} report: "
                f"{instruments}"
            )

    if "timeframe" in df.columns:
        timeframes = set(
            df["timeframe"].dropna().astype(str).str.upper()
        )

        if timeframes and timeframes != {timeframe}:
            raise ValueError(
                f"Unexpected timeframes in report: {timeframes}"
            )

    df["confirmation_candle_open_time"] = pd.to_datetime(
        df["confirmation_candle_open_time"],
        utc=True,
        errors="coerce",
    )

    df["known_at_time"] = pd.to_datetime(
        df["known_at_time"],
        utc=True,
        errors="coerce",
    )

    if df[
        ["confirmation_candle_open_time", "known_at_time"]
    ].isna().any().any():
        raise ValueError(
            f"{timeframe} report contains invalid timestamps."
        )

    df["trend"] = df["trend"].astype(str).str.upper()

    invalid_trends = set(df["trend"]) - VALID_TRENDS

    if invalid_trends:
        raise ValueError(
            f"Invalid {timeframe} trend states: {invalid_trends}"
        )

    if df["confirmation_candle_open_time"].duplicated().any():
        raise ValueError(
            f"{timeframe} report contains duplicate confirmation times."
        )

    if df["known_at_time"].duplicated().any():
        raise ValueError(
            f"{timeframe} report contains duplicate known-at times."
        )

    df = df.sort_values(
        "known_at_time"
    ).reset_index(drop=True)

    return df


def build_snapshots(df, timeframe):
    """
    Reconstruct TrendSnapshot objects.

    The engine calculates knowable time from:
        confirmation candle open + timeframe duration.
    """

    duration = TIMEFRAME_DURATIONS[timeframe]

    snapshots = []
    metadata = {}

    for _, row in df.iterrows():

        confirmation_time = (
            row["confirmation_candle_open_time"].to_pydatetime()
        )

        recorded_known_at = row["known_at_time"].to_pydatetime()

        calculated_known_at = confirmation_time + duration

        if calculated_known_at != recorded_known_at:
            raise ValueError(
                f"{timeframe} known-at mismatch. "
                f"Confirmation={confirmation_time}, "
                f"Recorded={recorded_known_at}, "
                f"Calculated={calculated_known_at}"
            )

        snapshot = TrendSnapshot(
            as_of_time=confirmation_time,
            trend=row["trend"],
            latest_high=None,
            latest_low=None,
            reason=str(row["reason"]),
        )

        snapshots.append(snapshot)

        metadata[confirmation_time] = row.to_dict()

    return snapshots, metadata


def get_dataset_end(storage, instrument, timeframe):
    """
    Validate a historical candle dataset and return its closing time.

    The final candle's opening time is not the dataset's actual
    assessment horizon. Its timeframe duration must be added.
    """

    candles = storage.load_dataset(instrument, timeframe)

    if candles.empty:
        raise RuntimeError(
            f"{timeframe} historical candle dataset is empty."
        )

    if "time" not in candles.columns:
        raise ValueError(
            f"{timeframe} candle dataset is missing the 'time' column."
        )

    candle_times = pd.to_datetime(
        candles["time"],
        utc=True,
        errors="coerce",
    )

    if candle_times.isna().any():
        raise ValueError(
            f"{timeframe} dataset contains invalid candle timestamps."
        )

    if candle_times.duplicated().any():
        raise ValueError(
            f"{timeframe} dataset contains duplicate candle timestamps."
        )

    if not candle_times.is_monotonic_increasing:
        raise ValueError(
            f"{timeframe} candle timestamps are not chronological."
        )

    if "is_forming_candle" in candles.columns:
        forming = candles["is_forming_candle"].fillna(False)

        if forming.any():
            raise RuntimeError(
                f"{timeframe} dataset contains forming candles."
            )

    last_candle_open = candle_times.iloc[-1]

    data_end = (
        last_candle_open
        + pd.Timedelta(TIMEFRAME_DURATIONS[timeframe])
    )

    return candles, data_end


def main():

    print("\n========== ALIGNMENT ANALYSIS ==========")

    settings = load_settings()

    trading = settings.get("trading", {})
    market = settings.get("market", {})
    research = settings.get("research", {})
    historical = research.get("historical_data", {})

    # SAFETY CHECKS

    if trading.get("execution_enabled") is not False:
        raise RuntimeError(
            "Alignment analysis blocked. "
            "Trade execution must remain disabled."
        )

    if historical.get("enabled") is not True:
        raise RuntimeError(
            "Historical research is disabled."
        )

    configured_timeframes = market.get("timeframes", {})

    for timeframe in ("H4", "H1"):
        if timeframe not in configured_timeframes:
            raise RuntimeError(
                f"{timeframe} is not configured in settings.yaml."
            )

    storage = HistoricalDataStorage()

    reports_directory = storage.reports_directory

    print(f"Instrument: {INSTRUMENT}")
    print(f"Reports directory: {reports_directory}")
    print("Execution enabled: False")

    # STEP 1: FIND SOURCE REPORTS

    h4_path = latest_report(
        reports_directory,
        f"{INSTRUMENT}_H4_trend_history_*.csv",
    )

    h1_path = latest_report(
        reports_directory,
        f"{INSTRUMENT}_H1_trend_history_*.csv",
    )

    print("\n========== SOURCE REPORTS ==========")
    print(f"H4: {h4_path.name}")
    print(f"H1: {h1_path.name}")

    # STEP 2: LOAD AND VALIDATE REPORTS

    h4_df = load_trend_report(h4_path, "H4")
    h1_df = load_trend_report(h1_path, "H1")

    print("\n========== SOURCE DATA ==========")
    print(f"H4 snapshots: {len(h4_df)}")
    print(f"H1 snapshots: {len(h1_df)}")

    # STEP 3: RECONSTRUCT SNAPSHOTS

    h4_snapshots, h4_metadata = build_snapshots(
        h4_df, "H4"
    )

    h1_snapshots, h1_metadata = build_snapshots(
        h1_df, "H1"
    )

    # STEP 4: IDENTIFY SHARED MARKET DATA COVERAGE

    h4_candles, h4_data_end = get_dataset_end(
        storage,
        INSTRUMENT,
        "H4",
    )

    h1_candles, h1_data_end = get_dataset_end(
        storage,
        INSTRUMENT,
        "H1",
    )

    earliest_common = max(
        h4_df["known_at_time"].min(),
        h1_df["known_at_time"].min(),
    )

    # The shared assessment horizon is determined by the
    # shorter candle dataset, not the last trend snapshot.

    latest_common = min(
        h4_data_end,
        h1_data_end,
    )

    if earliest_common > latest_common:
        raise RuntimeError(
            "H4 and H1 reports have no overlapping historical period."
        )

    print("\n========== SHARED HISTORY ==========")
    print(f"H4 candle count: {len(h4_candles)}")
    print(f"H1 candle count: {len(h1_candles)}")
    print(f"H4 market data ends: {h4_data_end}")
    print(f"H1 market data ends: {h1_data_end}")
    print(f"Shared assessment start: {earliest_common}")
    print(f"Shared assessment end: {latest_common}")

    # STEP 5: GENERATE ASSESSMENT TIMES

    assessment_times = set(
        h4_df.loc[
            h4_df["known_at_time"].between(
                earliest_common,
                latest_common,
            ),
            "known_at_time",
        ].tolist()
    )

    assessment_times.update(
        h1_df.loc[
            h1_df["known_at_time"].between(
                earliest_common,
                latest_common,
            ),
            "known_at_time",
        ].tolist()
    )

    # Explicitly include the previously audited reference point.

    reference_timestamp = pd.Timestamp(
        REFERENCE_ASSESSMENT_TIME
    )

    if earliest_common <= reference_timestamp <= latest_common:
        assessment_times.add(reference_timestamp)

    assessment_times = sorted(assessment_times)

    if not assessment_times:
        raise RuntimeError(
            "No valid assessment times were generated."
        )

    print("\n========== ASSESSMENT TIMES ==========")
    print(f"Assessment points generated: {len(assessment_times)}")
    print(f"Reference point included: {reference_timestamp in assessment_times}")

    # STEP 6: EVALUATE ALIGNMENT CHRONOLOGICALLY

    engine = AlignmentEngine()

    history_records = []

    for assessment_timestamp in assessment_times:

        assessment_time = assessment_timestamp.to_pydatetime()

        result = engine.evaluate(
            h4_snapshots=h4_snapshots,
            h1_snapshots=h1_snapshots,
            assessment_time=assessment_time,
        )

        h4_row = (
            h4_metadata.get(result.h4_snapshot.as_of_time)
            if result.h4_snapshot is not None
            else None
        )

        h1_row = (
            h1_metadata.get(result.h1_snapshot.as_of_time)
            if result.h1_snapshot is not None
            else None
        )

        # Confirm selected snapshots match their source reports.

        if result.h4_snapshot is not None:

            if h4_row is None:
                raise RuntimeError(
                    "Selected H4 snapshot has no source metadata."
                )

            if (
                result.h4_known_at_time
                != h4_row["known_at_time"].to_pydatetime()
            ):
                raise RuntimeError(
                    "H4 selected snapshot does not match "
                    "source known-at time."
                )

        if result.h1_snapshot is not None:

            if h1_row is None:
                raise RuntimeError(
                    "Selected H1 snapshot has no source metadata."
                )

            if (
                result.h1_known_at_time
                != h1_row["known_at_time"].to_pydatetime()
            ):
                raise RuntimeError(
                    "H1 selected snapshot does not match "
                    "source known-at time."
                )

        # Explicit no-lookahead checks.

        if (
            result.h4_known_at_time is not None
            and result.h4_known_at_time > result.assessment_time
        ):
            raise RuntimeError("H4 lookahead detected.")

        if (
            result.h1_known_at_time is not None
            and result.h1_known_at_time > result.assessment_time
        ):
            raise RuntimeError("H1 lookahead detected.")

        history_records.append({
            "instrument": INSTRUMENT,
            "assessment_time": result.assessment_time,

            "h4_trend": (
                result.h4_snapshot.trend
                if result.h4_snapshot else None
            ),
            "h4_confirmation_candle_open_time": (
                result.h4_snapshot.as_of_time
                if result.h4_snapshot else None
            ),
            "h4_known_at_time": result.h4_known_at_time,
            "h4_latest_high_classification": (
                h4_row.get("latest_high_classification")
                if h4_row else None
            ),
            "h4_latest_low_classification": (
                h4_row.get("latest_low_classification")
                if h4_row else None
            ),

            "h1_trend": (
                result.h1_snapshot.trend
                if result.h1_snapshot else None
            ),
            "h1_confirmation_candle_open_time": (
                result.h1_snapshot.as_of_time
                if result.h1_snapshot else None
            ),
            "h1_known_at_time": result.h1_known_at_time,
            "h1_latest_high_classification": (
                h1_row.get("latest_high_classification")
                if h1_row else None
            ),
            "h1_latest_low_classification": (
                h1_row.get("latest_low_classification")
                if h1_row else None
            ),

            "alignment_state": result.alignment_state,
            "directional_permission": result.directional_permission,
            "permitted_direction": result.permitted_direction,
            "reason": result.reason,
        })

    history_df = pd.DataFrame(history_records)

    # STEP 7: REFERENCE POINT VALIDATION

    reference_matches = history_df[
        history_df["assessment_time"]
        == reference_timestamp
    ]

    reference_passed = False
    reference_details = {}

    if len(reference_matches) == 1:

        reference = reference_matches.iloc[0]

        reference_details = {
            "h4_trend": reference["h4_trend"],
            "h1_trend": reference["h1_trend"],
            "alignment_state": reference["alignment_state"],
            "directional_permission": reference["directional_permission"],
        }

        reference_passed = (
            reference["h4_trend"] == "MIXED"
            and reference["h1_trend"] == "BEARISH"
            and reference["alignment_state"] == "H4_MIXED"
            and not bool(reference["directional_permission"])
        )

    # STEP 8: SUMMARY STATISTICS

    state_summary = (
        history_df.groupby("alignment_state")
        .size()
        .reset_index(name="assessment_count")
    )

    state_summary["percentage"] = (
        state_summary["assessment_count"]
        / len(history_df)
        * 100
    ).round(2)

    permission_count = int(
        history_df["directional_permission"].sum()
    )

    summary_records = [
        {
            "metric": "instrument",
            "value": INSTRUMENT,
        },
        {
            "metric": "assessment_count",
            "value": len(history_df),
        },
        {
            "metric": "h4_market_data_end",
            "value": h4_data_end,
        },
        {
            "metric": "h1_market_data_end",
            "value": h1_data_end,
        },
        {
            "metric": "shared_history_start",
            "value": earliest_common,
        },
        {
            "metric": "shared_history_end",
            "value": latest_common,
        },
        {
            "metric": "directional_permission_count",
            "value": permission_count,
        },
        {
            "metric": "directional_permission_percentage",
            "value": round(
                permission_count / len(history_df) * 100,
                2,
            ),
        },
        {
            "metric": "directional_permission_denied_count",
            "value": len(history_df) - permission_count,
        },
        {
            "metric": "h4_source_report",
            "value": h4_path.name,
        },
        {
            "metric": "h1_source_report",
            "value": h1_path.name,
        },
    ]

    for _, row in state_summary.iterrows():

        summary_records.append({
            "metric": f"alignment_state_{row['alignment_state']}",
            "value": (
                f"{row['assessment_count']} "
                f"({row['percentage']}%)"
            ),
        })

    summary_df = pd.DataFrame(summary_records)

    # STEP 9: AUDIT

    audit_records = [
        {
            "check": "execution_disabled",
            "passed": trading.get("execution_enabled") is False,
            "details": "Trade execution remains disabled.",
        },
        {
            "check": "h4_source_valid",
            "passed": not h4_df.empty and not h4_candles.empty,
            "details": (
                f"{len(h4_df)} H4 snapshots and "
                f"{len(h4_candles)} candles loaded."
            ),
        },
        {
            "check": "h1_source_valid",
            "passed": not h1_df.empty and not h1_candles.empty,
            "details": (
                f"{len(h1_df)} H1 snapshots and "
                f"{len(h1_candles)} candles loaded."
            ),
        },
        {
            "check": "h4_known_at_consistency",
            "passed": True,
            "details": (
                "All H4 recorded known-at times match calculated times."
            ),
        },
        {
            "check": "h1_known_at_consistency",
            "passed": True,
            "details": (
                "All H1 recorded known-at times match calculated times."
            ),
        },
        {
            "check": "shared_history_valid",
            "passed": earliest_common <= latest_common,
            "details": (
                f"{earliest_common} to {latest_common}; "
                f"H4 data ends {h4_data_end}; "
                f"H1 data ends {h1_data_end}."
            ),
        },
        {
            "check": "no_lookahead",
            "passed": True,
            "details": (
                "Every selected H4 and H1 snapshot was known "
                "at or before its assessment time."
            ),
        },
        {
            "check": "reference_point",
            "passed": reference_passed,
            "details": str(reference_details),
        },
        {
            "check": "assessment_history_nonempty",
            "passed": not history_df.empty,
            "details": (
                f"{len(history_df)} assessments generated."
            ),
        },
    ]

    audit_df = pd.DataFrame(audit_records)

    # STEP 10: SAVE REPORTS

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    history_path = storage.save_report(
        history_df,
        f"EURUSD_H4_H1_alignment_history_{timestamp}",
    )

    summary_path = storage.save_report(
        summary_df,
        f"EURUSD_H4_H1_alignment_summary_{timestamp}",
    )

    audit_path = storage.save_report(
        audit_df,
        f"EURUSD_H4_H1_alignment_audit_{timestamp}",
    )

    # FINAL RESULTS

    print("\n========== ALIGNMENT RESULTS ==========")

    print(f"Assessments: {len(history_df)}")
    print(f"Directional permissions granted: {permission_count}")
    print(
        "Directional permissions denied: "
        f"{len(history_df) - permission_count}"
    )

    print("\n========== ALIGNMENT STATE COUNTS ==========")
    print(state_summary.to_string(index=False))

    print("\n========== REFERENCE POINT ==========")
    print(f"Assessment: {REFERENCE_ASSESSMENT_TIME}")
    print(f"Results: {reference_details}")
    print(f"Reference passed: {reference_passed}")

    print("\n========== AUDIT RESULTS ==========")
    print(audit_df.to_string(index=False))

    if not audit_df["passed"].all():
        raise RuntimeError(
            "Alignment audit failed. Review the audit report."
        )

    print("\n========== SAVED REPORTS ==========")
    print(history_path)
    print(summary_path)
    print(audit_path)

    print("\nALIGNMENT ANALYSIS COMPLETED SUCCESSFULLY.")

    log.info(
        "Alignment analysis completed | "
        "Assessments=%s | Permissions=%s",
        len(history_df),
        permission_count,
    )


if __name__ == "__main__":
    main()
