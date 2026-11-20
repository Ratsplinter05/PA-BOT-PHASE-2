
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import load_settings
from src.logger import get_logger

from research.historical_storage import HistoricalDataStorage
from research.swing_detection.fractal_detector import FractalSwingDetector
from research.swing_detection.structure_classifier import StructureClassifier
from research.swing_detection.bos_detector import BOSDetector
from research.swing_detection.alignment_engine import AlignmentEngine
from research.swing_detection.trend_models import TrendSnapshot


log = get_logger("M15_BOS_ANALYSIS")

INSTRUMENT = "EURUSD"

TIMEFRAME_DURATIONS = {
    "M15": timedelta(minutes=15),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
}

VALID_TRENDS = {
    "BULLISH",
    "BEARISH",
    "MIXED",
    "UNCONFIRMED",
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
                f"Unexpected timeframes in {timeframe} report."
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

    duration = TIMEFRAME_DURATIONS[timeframe]

    expected_known_at = (
        df["confirmation_candle_open_time"] + duration
    )

    if not (
        expected_known_at == df["known_at_time"]
    ).all():
        raise ValueError(
            f"{timeframe} report contains inconsistent known-at times."
        )

    return df.sort_values(
        "known_at_time"
    ).reset_index(drop=True)


def build_snapshots(df):
    """Reconstruct TrendSnapshot objects from a trend report."""

    snapshots = []
    metadata = {}

    for _, row in df.iterrows():

        confirmation_time = (
            row["confirmation_candle_open_time"].to_pydatetime()
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


def load_and_validate_candles(storage, timeframe):

    candles = storage.load_dataset(
        INSTRUMENT,
        timeframe,
    )

    if candles.empty:
        raise RuntimeError(
            f"{timeframe} historical dataset is empty."
        )

    required_columns = {
        "time",
        "high",
        "low",
        "close",
    }

    missing = required_columns - set(candles.columns)

    if missing:
        raise ValueError(
            f"{timeframe} dataset is missing columns: {sorted(missing)}"
        )

    candle_times = pd.to_datetime(
        candles["time"],
        utc=True,
        errors="coerce",
    )

    if candle_times.isna().any():
        raise ValueError(
            f"{timeframe} dataset contains invalid timestamps."
        )

    if candle_times.duplicated().any():
        raise ValueError(
            f"{timeframe} dataset contains duplicate timestamps."
        )

    if not candle_times.is_monotonic_increasing:
        raise ValueError(
            f"{timeframe} dataset is not chronological."
        )

    if "is_forming_candle" in candles.columns:
        if candles["is_forming_candle"].fillna(False).any():
            raise RuntimeError(
                f"{timeframe} dataset contains forming candles."
            )

    data_end = (
        candle_times.iloc[-1]
        + pd.Timedelta(TIMEFRAME_DURATIONS[timeframe])
    )

    return candles, data_end


def main():

    print("\n========== M15 BOS ANALYSIS ==========")

    # --------------------------------------------------------
    # 1. SAFETY AND CONFIGURATION
    # --------------------------------------------------------

    settings = load_settings()

    trading = settings.get("trading", {})
    market = settings.get("market", {})
    research = settings.get("research", {})
    historical = research.get("historical_data", {})

    if trading.get("execution_enabled") is not False:
        raise RuntimeError(
            "M15 BOS analysis blocked. "
            "Trade execution must remain disabled."
        )

    if historical.get("enabled") is not True:
        raise RuntimeError(
            "Historical research is disabled."
        )

    configured_timeframes = market.get("timeframes", {})

    for timeframe in ("M15", "H1", "H4"):
        if timeframe not in configured_timeframes:
            raise RuntimeError(
                f"{timeframe} is not configured in settings.yaml."
            )

    storage = HistoricalDataStorage()
    reports_directory = storage.reports_directory

    # --------------------------------------------------------
    # 2. LOAD SOURCE REPORTS
    # --------------------------------------------------------

    print("\n========== SOURCE REPORTS ==========")

    m15_path = latest_report(
        reports_directory,
        f"{INSTRUMENT}_M15_trend_history_*.csv",
    )

    h1_path = latest_report(
        reports_directory,
        f"{INSTRUMENT}_H1_trend_history_*.csv",
    )

    h4_path = latest_report(
        reports_directory,
        f"{INSTRUMENT}_H4_trend_history_*.csv",
    )

    print(f"M15: {m15_path.name}")
    print(f"H1:  {h1_path.name}")
    print(f"H4:  {h4_path.name}")

    # --------------------------------------------------------
    # 3. LOAD AND VALIDATE TREND REPORTS
    # --------------------------------------------------------

    m15_df = load_trend_report(m15_path, "M15")
    h1_df = load_trend_report(h1_path, "H1")
    h4_df = load_trend_report(h4_path, "H4")

    m15_snapshots, m15_metadata = build_snapshots(m15_df)
    h1_snapshots, h1_metadata = build_snapshots(h1_df)
    h4_snapshots, h4_metadata = build_snapshots(h4_df)

    print("\n========== TREND SNAPSHOTS ==========")
    print(f"M15 snapshots: {len(m15_snapshots)}")
    print(f"H1 snapshots:  {len(h1_snapshots)}")
    print(f"H4 snapshots:  {len(h4_snapshots)}")

    # --------------------------------------------------------
    # 4. LOAD HISTORICAL CANDLE DATA
    # --------------------------------------------------------

    print("\n========== HISTORICAL CANDLE DATA ==========")

    m15_candles, m15_data_end = load_and_validate_candles(
        storage, "M15"
    )

    h1_candles, h1_data_end = load_and_validate_candles(
        storage, "H1"
    )

    h4_candles, h4_data_end = load_and_validate_candles(
        storage, "H4"
    )

    print(f"M15 candles: {len(m15_candles)}")
    print(f"H1 candles:  {len(h1_candles)}")
    print(f"H4 candles:  {len(h4_candles)}")

    # --------------------------------------------------------
    # 5. DETERMINE SHARED ALIGNMENT HISTORY
    # --------------------------------------------------------

    earliest_common = max(
        m15_df["known_at_time"].min(),
        h1_df["known_at_time"].min(),
        h4_df["known_at_time"].min(),
    )

    latest_common = min(
        m15_data_end,
        h1_data_end,
        h4_data_end,
    )

    if earliest_common > latest_common:
        raise RuntimeError(
            "M15, H1 and H4 have no shared historical assessment period."
        )

    print("\n========== SHARED HISTORY ==========")
    print(f"Shared start: {earliest_common}")
    print(f"Shared end:   {latest_common}")

    # --------------------------------------------------------
    # 6. DETECT M15 SWINGS AND MARKET STRUCTURE
    # --------------------------------------------------------

    print("\n========== M15 STRUCTURE ==========")

    swing_detector = FractalSwingDetector(
        left_bars=2,
        right_bars=2,
    )

    swings = swing_detector.detect(m15_candles)

    structure_classifier = StructureClassifier(
        price_tolerance=0.0,
    )

    classified_swings = structure_classifier.classify(
        swings
    )

    print(f"Confirmed swings: {len(swings)}")
    print(f"Classified swings: {len(classified_swings)}")

    # --------------------------------------------------------
    # 7. DETECT M15 BREAKS OF STRUCTURE
    # --------------------------------------------------------

    print("\n========== M15 BOS DETECTION ==========")

    bos_detector = BOSDetector()

    bos_events = bos_detector.detect(
        candles=m15_candles,
        swings=classified_swings,
        trend_snapshots=m15_snapshots,
        timeframe="M15",
    )

    print(f"Total M15 BOS events: {len(bos_events)}")

    # --------------------------------------------------------
    # 8. EVALUATE H4/H1 PERMISSION AT BOS CLOSE
    # --------------------------------------------------------

    print("\n========== H4/H1 BOS PERMISSION ==========")

    alignment_engine = AlignmentEngine()

    event_records = []

    permission_granted = 0
    permission_denied = 0
    outside_history = 0

    for event in bos_events:

        decision_time = (
            event.break_time + TIMEFRAME_DURATIONS["M15"]
        )

        # Do not assess alignment outside the shared
        # historical coverage of all three timeframes.

        if not (
            earliest_common
            <= decision_time
            <= latest_common
        ):

            alignment_state = "OUTSIDE_ALIGNMENT_HISTORY"
            directional_permission = False
            permitted_direction = None
            h4_trend = None
            h1_trend = None
            h4_known_at = None
            h1_known_at = None
            alignment_reason = (
                "BOS decision time is outside shared H4/H1/M15 coverage."
            )

            outside_history += 1

        else:

            result = alignment_engine.evaluate(
                h4_snapshots=h4_snapshots,
                h1_snapshots=h1_snapshots,
                assessment_time=decision_time,
            )

            if (
                result.h4_known_at_time is not None
                and result.h4_known_at_time > decision_time
            ):
                raise RuntimeError(
                    "H4 lookahead detected during BOS permission check."
                )

            if (
                result.h1_known_at_time is not None
                and result.h1_known_at_time > decision_time
            ):
                raise RuntimeError(
                    "H1 lookahead detected during BOS permission check."
                )

            alignment_state = result.alignment_state
            directional_permission = result.directional_permission
            permitted_direction = result.permitted_direction

            h4_trend = (
                result.h4_snapshot.trend
                if result.h4_snapshot else None
            )

            h1_trend = (
                result.h1_snapshot.trend
                if result.h1_snapshot else None
            )

            h4_known_at = result.h4_known_at_time
            h1_known_at = result.h1_known_at_time
            alignment_reason = result.reason

        direction_matches = (
            directional_permission
            and permitted_direction == event.direction
        )

        if direction_matches:
            qualification = "PERMITTED"
            permission_granted += 1

        else:
            qualification = "REJECTED"
            permission_denied += 1

        swing = event.level_swing.swing

        event_records.append({

            "instrument": INSTRUMENT,
            "timeframe": "M15",

            "event_type": event.event_type,
            "direction": event.direction,

            "break_time": event.break_time,
            "decision_time": decision_time,
            "break_index": event.break_index,

            "broken_level": event.level_price,
            "close_price": event.close_price,

            "broken_swing_type": swing.swing_type,
            "broken_swing_classification": (
                event.level_swing.classification
            ),
            "broken_swing_time": swing.swing_time,
            "broken_swing_index": swing.swing_index,
            "broken_swing_confirmation_index": (
                swing.confirmation_index
            ),

            "m15_prior_trend": event.prior_trend,

            "h4_trend": h4_trend,
            "h4_known_at_time": h4_known_at,

            "h1_trend": h1_trend,
            "h1_known_at_time": h1_known_at,

            "alignment_state": alignment_state,
            "directional_permission": directional_permission,
            "permitted_direction": permitted_direction,

            "direction_matches_permission": direction_matches,
            "qualification": qualification,

            "alignment_reason": alignment_reason,

        })

    event_df = pd.DataFrame(event_records)

    # --------------------------------------------------------
    # 9. BUILD PERMITTED BOS REPORT
    # --------------------------------------------------------

    if event_df.empty:

        permitted_df = pd.DataFrame(
            columns=event_df.columns
        )

    else:

        permitted_df = event_df[
            event_df["qualification"] == "PERMITTED"
        ].copy()

    # --------------------------------------------------------
    # 10. SUMMARY STATISTICS
    # --------------------------------------------------------

    if event_df.empty:

        direction_summary = pd.DataFrame(
            columns=["direction", "event_count"]
        )

        qualification_summary = pd.DataFrame(
            columns=["qualification", "event_count"]
        )

    else:

        direction_summary = (
            event_df.groupby("direction")
            .size()
            .reset_index(name="event_count")
        )

        qualification_summary = (
            event_df.groupby("qualification")
            .size()
            .reset_index(name="event_count")
        )

    summary_records = [
        {
            "metric": "instrument",
            "value": INSTRUMENT,
        },
        {
            "metric": "m15_candles",
            "value": len(m15_candles),
        },
        {
            "metric": "m15_confirmed_swings",
            "value": len(swings),
        },
        {
            "metric": "m15_trend_snapshots",
            "value": len(m15_snapshots),
        },
        {
            "metric": "total_bos_events",
            "value": len(bos_events),
        },
        {
            "metric": "permitted_bos_events",
            "value": permission_granted,
        },
        {
            "metric": "rejected_bos_events",
            "value": permission_denied,
        },
        {
            "metric": "outside_alignment_history",
            "value": outside_history,
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
            "metric": "execution_enabled",
            "value": trading.get("execution_enabled"),
        },
    ]

    for _, row in direction_summary.iterrows():
        summary_records.append({
            "metric": f"bos_direction_{row['direction']}",
            "value": row["event_count"],
        })

    for _, row in qualification_summary.iterrows():
        summary_records.append({
            "metric": f"bos_{row['qualification'].lower()}",
            "value": row["event_count"],
        })

    summary_df = pd.DataFrame(summary_records)

    # --------------------------------------------------------
    # 11. AUDIT
    # --------------------------------------------------------

    audit_records = [
        {
            "check": "execution_disabled",
            "passed": trading.get("execution_enabled") is False,
            "details": "Trade execution remains disabled.",
        },
        {
            "check": "historical_research_enabled",
            "passed": historical.get("enabled") is True,
            "details": "Historical research is enabled.",
        },
        {
            "check": "all_timeframes_configured",
            "passed": all(
                tf in configured_timeframes
                for tf in ("M15", "H1", "H4")
            ),
            "details": "M15, H1 and H4 are configured.",
        },
        {
            "check": "trend_reports_valid",
            "passed": (
                not m15_df.empty
                and not h1_df.empty
                and not h4_df.empty
            ),
            "details": "All three trend reports loaded and validated.",
        },
        {
            "check": "candle_datasets_valid",
            "passed": (
                not m15_candles.empty
                and not h1_candles.empty
                and not h4_candles.empty
            ),
            "details": "All three candle datasets loaded and validated.",
        },
        {
            "check": "shared_history_valid",
            "passed": earliest_common <= latest_common,
            "details": (
                f"{earliest_common} to {latest_common}."
            ),
        },
        {
            "check": "bos_decision_after_candle_close",
            "passed": all(
                record["decision_time"]
                == record["break_time"]
                + TIMEFRAME_DURATIONS["M15"]
                for record in event_records
            ),
            "details": (
                "Every BOS decision is assessed at the M15 "
                "break candle close."
            ),
        },
        {
            "check": "no_alignment_lookahead",
            "passed": all(
                (
                    record["h4_known_at_time"] is None
                    or record["h4_known_at_time"]
                    <= record["decision_time"]
                )
                and (
                    record["h1_known_at_time"] is None
                    or record["h1_known_at_time"]
                    <= record["decision_time"]
                )
                for record in event_records
            ),
            "details": (
                "Selected H4/H1 snapshots were known by "
                "the BOS decision time."
            ),
        },
        {
            "check": "permitted_direction_matches_bos",
            "passed": all(
                record["qualification"] != "PERMITTED"
                or (
                    record["directional_permission"]
                    and record["permitted_direction"]
                    == record["direction"]
                )
                for record in event_records
            ),
            "details": (
                "Every permitted BOS matches the H4/H1 "
                "permitted direction."
            ),
        },
    ]

    audit_df = pd.DataFrame(audit_records)

    # --------------------------------------------------------
    # 12. SAVE REPORTS
    # --------------------------------------------------------

    print("\n========== SAVING M15 BOS REPORTS ==========")

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")

    event_path = storage.save_report(
        event_df,
        f"EURUSD_M15_BOS_all_events_{timestamp}",
        overwrite=False,
    )

    permitted_path = storage.save_report(
        permitted_df,
        f"EURUSD_M15_BOS_permitted_events_{timestamp}",
        overwrite=False,
    )

    summary_path = storage.save_report(
        summary_df,
        f"EURUSD_M15_BOS_summary_{timestamp}",
        overwrite=False,
    )

    audit_path = storage.save_report(
        audit_df,
        f"EURUSD_M15_BOS_audit_{timestamp}",
        overwrite=False,
    )

    # --------------------------------------------------------
    # 13. DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n========== M15 BOS RESULTS ==========")

    print(f"Total BOS events: {len(bos_events)}")
    print(f"Permitted BOS events: {permission_granted}")
    print(f"Rejected BOS events: {permission_denied}")
    print(f"Outside alignment history: {outside_history}")

    print("\n========== BOS DIRECTION SUMMARY ==========")
    print(direction_summary.to_string(index=False))

    print("\n========== BOS QUALIFICATION SUMMARY ==========")
    print(qualification_summary.to_string(index=False))

    if not permitted_df.empty:

        print("\n========== LATEST PERMITTED BOS EVENTS ==========")

        print(
            permitted_df.tail(15).to_string(index=False)
        )

    else:

        print("\nNo directionally permitted BOS events were found.")

    print("\n========== AUDIT RESULTS ==========")
    print(audit_df.to_string(index=False))

    if not audit_df["passed"].all():
        raise RuntimeError(
            "M15 BOS analysis audit failed. Review the audit report."
        )

    print("\n========== SAVED REPORTS ==========")
    print(event_path)
    print(permitted_path)
    print(summary_path)
    print(audit_path)

    print("\nM15 BOS ANALYSIS COMPLETED SUCCESSFULLY.")

    log.info(
        "M15 BOS analysis completed | "
        "Events=%s | Permitted=%s | Rejected=%s",
        len(bos_events),
        permission_granted,
        permission_denied,
    )


if __name__ == "__main__":
    main()
