
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "data" / "historical" / "reports"

ALIGNMENT_FILE = (
    REPORT_DIR / "EURUSD_historical_alignment_validation.csv"
)

H1_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "raw"
    / "EURUSD"
    / "EURUSD_H1.parquet"
)

OUTPUT_PREFIX = "EURUSD_alignment_v2"

H1_DURATION = pd.Timedelta(hours=1)

REQUIRED_ALIGNMENT_COLUMNS = [
    "assessment_number",
    "assessment_time",
    "alignment_state",
    "directional_permission",
    "permitted_direction",
    "h4_trend",
    "h4_snapshot_as_of",
    "h4_known_at",
    "h1_trend",
    "h1_snapshot_as_of",
    "h1_known_at",
]

REQUIRED_H1_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "is_forming_candle",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_boolean(value: Any) -> bool:
    """Convert common CSV/Parquet boolean representations."""

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        raise ValueError("Unexpected missing boolean value.")

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def clean_text(value: Any) -> str:
    """Convert missing values into empty strings."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def utc_hours_between(
    earlier: pd.Timestamp,
    later: pd.Timestamp,
) -> float:
    """Calculate elapsed hours between UTC timestamps."""

    return (later - earlier).total_seconds() / 3600


# ============================================================
# LOAD ALIGNMENT REPORT
# ============================================================

def load_alignment_report() -> pd.DataFrame:

    print("\n[1] Loading historical alignment report")

    if not ALIGNMENT_FILE.exists():
        raise FileNotFoundError(
            f"Alignment report not found:\n{ALIGNMENT_FILE}"
        )

    df = pd.read_csv(ALIGNMENT_FILE)

    missing = [
        column
        for column in REQUIRED_ALIGNMENT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing alignment columns: {missing}"
        )

    if df.empty:
        raise ValueError("Alignment report is empty.")

    df["assessment_number"] = pd.to_numeric(
        df["assessment_number"],
        errors="raise",
    )

    df["assessment_time"] = pd.to_datetime(
        df["assessment_time"],
        utc=True,
        errors="raise",
    )

    df["directional_permission"] = (
        df["directional_permission"].apply(parse_boolean)
    )

    text_columns = [
        "alignment_state",
        "permitted_direction",
        "h4_trend",
        "h1_trend",
    ]

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    if df["assessment_time"].duplicated().any():
        raise ValueError(
            "Duplicate assessment timestamps detected."
        )

    if not df["assessment_time"].is_monotonic_increasing:
        raise ValueError(
            "Assessment timestamps are not chronological."
        )

    if df["alignment_state"].eq("").any():
        raise ValueError(
            "Missing alignment states detected."
        )

    invalid_permission = df[
        (
            df["directional_permission"]
            & ~df["permitted_direction"].isin(
                ["BULLISH", "BEARISH"]
            )
        )
        |
        (
            ~df["directional_permission"]
            & df["permitted_direction"].ne("")
        )
    ]

    if not invalid_permission.empty:
        raise ValueError(
            "Directional permission and direction are inconsistent."
        )

    print(f"Assessments loaded: {len(df):,}")
    print(f"First assessment: {df['assessment_time'].iloc[0]}")
    print(f"Last assessment:  {df['assessment_time'].iloc[-1]}")

    return df


# ============================================================
# LOAD ACTUAL H1 MARKET OBSERVATIONS
# ============================================================

def load_h1_market_observations() -> pd.DataFrame:

    print("\n[2] Loading actual H1 historical candles")

    if not H1_DATA_FILE.exists():
        raise FileNotFoundError(
            f"H1 historical dataset not found:\n{H1_DATA_FILE}"
        )

    h1 = pd.read_parquet(H1_DATA_FILE)

    missing = [
        column
        for column in REQUIRED_H1_COLUMNS
        if column not in h1.columns
    ]

    if missing:
        raise ValueError(
            f"Missing H1 dataset columns: {missing}"
        )

    if h1.empty:
        raise ValueError("H1 dataset is empty.")

    h1["time"] = pd.to_datetime(
        h1["time"],
        utc=True,
        errors="raise",
    )

    if h1["time"].duplicated().any():
        raise ValueError(
            "Duplicate H1 candle timestamps detected."
        )

    if not h1["time"].is_monotonic_increasing:
        h1 = h1.sort_values("time").reset_index(drop=True)

    h1["is_forming_candle"] = (
        h1["is_forming_candle"].apply(parse_boolean)
    )

    h1 = h1[
        ~h1["is_forming_candle"]
    ].copy()

    if h1.empty:
        raise ValueError(
            "No completed H1 candles are available."
        )

    # Each completed H1 candle represents an observed
    # one-hour interval ending one hour after its open.
    h1["candle_close_time"] = h1["time"] + H1_DURATION

    if h1["candle_close_time"].duplicated().any():
        raise ValueError(
            "Duplicate H1 candle close times detected."
        )

    h1 = h1.reset_index(drop=True)

    print(f"Completed H1 candles: {len(h1):,}")
    print(f"First candle open: {h1['time'].iloc[0]}")
    print(f"Last candle open:  {h1['time'].iloc[-1]}")

    return h1


# ============================================================
# BUILD OBSERVED ALIGNMENT TIMELINE
# ============================================================

def build_observed_timeline(
    alignment: pd.DataFrame,
    h1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("\n[3] Matching alignment assessments to H1 candles")

    h1_close_times = set(h1["candle_close_time"])

    observed = alignment[
        alignment["assessment_time"].isin(h1_close_times)
    ].copy()

    if observed.empty:
        raise ValueError(
            "No alignment assessments matched completed H1 candles."
        )

    close_to_open = dict(
        zip(
            h1["candle_close_time"],
            h1["time"],
        )
    )

    observed["h1_candle_open"] = (
        observed["assessment_time"].map(close_to_open)
    )

    observed = observed.reset_index(drop=True)

    unmatched = alignment[
        ~alignment["assessment_time"].isin(h1_close_times)
    ].copy()

    matched_count = len(observed)
    unmatched_count = len(unmatched)

    match_percentage = (
        matched_count / len(alignment) * 100
    )

    print(f"Matched assessments: {matched_count:,}")
    print(f"Unmatched assessments: {unmatched_count:,}")
    print(f"Observed coverage: {match_percentage:.2f}%")

    print(
        f"First observed assessment: "
        f"{observed['assessment_time'].iloc[0]}"
    )

    print(
        f"Last observed assessment: "
        f"{observed['assessment_time'].iloc[-1]}"
    )

    if len(observed) < 2:
        raise ValueError(
            "Insufficient matched observations for transition analysis."
        )

    return observed, unmatched


# ============================================================
# OBSERVATION GAP ANALYSIS
# ============================================================

def build_observation_gaps(
    observed: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for index in range(1, len(observed)):

        previous = observed.iloc[index - 1]
        current = observed.iloc[index]

        previous_time = previous["assessment_time"]
        current_time = current["assessment_time"]

        elapsed_hours = utc_hours_between(
            previous_time,
            current_time,
        )

        if elapsed_hours <= 1:
            continue

        records.append({
            "previous_assessment": previous_time,
            "next_assessment": current_time,
            "elapsed_calendar_hours": elapsed_hours,
            "unobserved_hours": elapsed_hours - 1,
            "previous_state": previous["alignment_state"],
            "next_state": current["alignment_state"],
            "gap_type": (
                "WEEKEND_OR_SESSION_OR_DATA_GAP"
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# RUN IDENTIFICATION
# ============================================================

def build_runs(
    observed: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """
    Identify runs using actual observed H1 candles.

    State runs:
        Consecutive observed candles with the same alignment state.

    Directional runs:
        Consecutive observed candles with the same permitted
        direction. A blocked assessment ends a directional run.

    Weekend and data gaps do not add observed H1 hours.
    """

    if mode not in {"state", "directional"}:
        raise ValueError(f"Unsupported run mode: {mode}")

    records = []
    current_rows = []
    current_key = None

    def save_run() -> None:
        nonlocal current_rows

        if not current_rows:
            return

        first = current_rows[0]
        last = current_rows[-1]

        timestamps = [
            row["assessment_time"]
            for row in current_rows
        ]

        gaps = [
            utc_hours_between(
                timestamps[i - 1],
                timestamps[i],
            )
            for i in range(1, len(timestamps))
        ]

        observed_hours = len(current_rows)

        calendar_span = (
            utc_hours_between(
                timestamps[0],
                timestamps[-1],
            ) + 1
        )

        record = {
            "start_time": timestamps[0],
            "end_time": timestamps[-1],
            "observed_h1_hours": observed_hours,
            "calendar_span_hours": calendar_span,
            "unobserved_hours_within_run": (
                max(0, calendar_span - observed_hours)
            ),
            "market_gaps_crossed": sum(
                gap > 1 for gap in gaps
            ),
            "maximum_observation_gap_hours": (
                max(gaps) if gaps else 0
            ),
        }

        if mode == "state":
            record["alignment_state"] = current_key
        else:
            record["direction"] = current_key

        records.append(record)
        current_rows = []

    for _, row in observed.iterrows():

        if mode == "state":
            eligible = True
            key = row["alignment_state"]

        else:
            eligible = row["directional_permission"]
            key = row["permitted_direction"]

        if not eligible:
            save_run()
            current_key = None
            continue

        if current_rows and key != current_key:
            save_run()

        if not current_rows:
            current_key = key

        current_rows.append(row)

    save_run()

    return pd.DataFrame(records)


# ============================================================
# STATE TRANSITIONS
# ============================================================

def build_transition_analysis(
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("\n[4] Analysing observed state transitions")

    records = []

    for index in range(1, len(observed)):

        previous = observed.iloc[index - 1]
        current = observed.iloc[index]

        elapsed = utc_hours_between(
            previous["assessment_time"],
            current["assessment_time"],
        )

        previous_state = previous["alignment_state"]
        current_state = current["alignment_state"]

        if previous_state == current_state:
            continue

        records.append({
            "transition_time": current["assessment_time"],
            "previous_state": previous_state,
            "current_state": current_state,
            "calendar_gap_hours": elapsed - 1,
            "transition_after_gap": elapsed > 1,
            "previous_direction": previous["permitted_direction"],
            "current_direction": current["permitted_direction"],
            "previous_permission": previous["directional_permission"],
            "current_permission": current["directional_permission"],
        })

    transitions = pd.DataFrame(records)

    matrix = pd.crosstab(
        pd.Series(
            [
                observed.iloc[i - 1]["alignment_state"]
                for i in range(1, len(observed))
            ],
            name="previous_state",
        ),
        pd.Series(
            [
                observed.iloc[i]["alignment_state"]
                for i in range(1, len(observed))
            ],
            name="current_state",
        ),
    )

    print(f"Observed state changes: {len(transitions):,}")

    return transitions, matrix


# ============================================================
# DIRECTIONAL PERMISSION LOSS
# ============================================================

def build_permission_loss_events(
    observed: pd.DataFrame,
) -> pd.DataFrame:

    print("\n[5] Analysing directional permission losses")

    records = []

    for index in range(1, len(observed)):

        previous = observed.iloc[index - 1]
        current = observed.iloc[index]

        elapsed = utc_hours_between(
            previous["assessment_time"],
            current["assessment_time"],
        )

        previous_direction = previous["permitted_direction"]

        if not previous["directional_permission"]:
            continue

        if (
            current["directional_permission"]
            and current["permitted_direction"] == previous_direction
        ):
            continue

        if current["directional_permission"]:
            event_type = "DIRECT_DIRECTIONAL_REVERSAL"
        else:
            event_type = "DIRECTIONAL_PERMISSION_LOST"

        h4_retains = current["h4_trend"] == previous_direction
        h1_retains = current["h1_trend"] == previous_direction

        if not h4_retains and not h1_retains:
            timeframe_result = "BOTH_LOST_DIRECTION"

        elif not h4_retains:
            timeframe_result = "H4_LOST_DIRECTION"

        elif not h1_retains:
            timeframe_result = "H1_LOST_DIRECTION"

        else:
            timeframe_result = "BOTH_STILL_SUPPORT_DIRECTION"

        records.append({
            "event_time": current["assessment_time"],
            "event_type": event_type,
            "previous_direction": previous_direction,
            "current_state": current["alignment_state"],
            "current_direction": current["permitted_direction"],
            "h4_current_trend": current["h4_trend"],
            "h1_current_trend": current["h1_trend"],
            "timeframe_result": timeframe_result,
            "calendar_gap_hours": elapsed - 1,
            "transition_after_gap": elapsed > 1,
            "h4_snapshot_as_of": current["h4_snapshot_as_of"],
            "h4_known_at": current["h4_known_at"],
            "h1_snapshot_as_of": current["h1_snapshot_as_of"],
            "h1_known_at": current["h1_known_at"],
        })

    result = pd.DataFrame(records)

    print(f"Permission loss/reversal events: {len(result):,}")

    return result


# ============================================================
# DIRECTIONAL REVERSALS
# ============================================================

def build_reversal_analysis(
    directional_runs: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    if directional_runs.empty:
        return pd.DataFrame()

    for index in range(1, len(directional_runs)):

        previous = directional_runs.iloc[index - 1]
        current = directional_runs.iloc[index]

        if previous["direction"] == current["direction"]:
            continue

        elapsed = utc_hours_between(
            previous["end_time"],
            current["start_time"],
        )

        records.append({
            "previous_direction": previous["direction"],
            "previous_run_start": previous["start_time"],
            "previous_run_end": previous["end_time"],
            "next_direction": current["direction"],
            "next_run_start": current["start_time"],
            "calendar_hours_between_runs": elapsed,
            "transition_type": (
                f"{previous['direction']}_TO_{current['direction']}"
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# DATA AVAILABILITY SUMMARY
# ============================================================

def build_data_availability_summary(
    alignment: pd.DataFrame,
    observed: pd.DataFrame,
    unmatched: pd.DataFrame,
) -> pd.DataFrame:

    state_counts = (
        alignment["alignment_state"]
        .value_counts()
        .rename_axis("alignment_state")
        .reset_index(name="assessment_count")
    )

    state_counts["observed_h1_assessments"] = (
        state_counts["alignment_state"].map(
            observed["alignment_state"].value_counts()
        ).fillna(0).astype(int)
    )

    state_counts["unmatched_assessments"] = (
        state_counts["assessment_count"]
        - state_counts["observed_h1_assessments"]
    )

    state_counts["observation_type"] = (
        state_counts["alignment_state"].map(
            lambda state: (
                "DATA_AVAILABILITY_OR_WARMUP"
                if state in {"H4_MISSING", "H4_UNCONFIRMED"}
                else "MARKET_STRUCTURE_STATE"
            )
        )
    )

    return state_counts


# ============================================================
# SESSION ANALYSIS
# ============================================================

def classify_utc_session(hour: int) -> str:
    """
    Approximate fixed UTC session buckets.

    These boundaries are not DST-adjusted.
    """

    if 0 <= hour <= 6:
        return "ASIA"

    if 7 <= hour <= 12:
        return "LONDON"

    if 13 <= hour <= 16:
        return "LONDON_NY_OVERLAP"

    if 17 <= hour <= 21:
        return "NEW_YORK"

    return "LATE_ROLLOVER"


def build_session_analysis(
    observed: pd.DataFrame,
) -> pd.DataFrame:

    print("\n[6] Analysing observed weekday sessions")

    # Use day_of_week instead of deprecated dayofweek.
    weekday = observed[
        observed["assessment_time"].dt.day_of_week < 5
    ].copy()

    weekday["utc_hour"] = (
        weekday["assessment_time"].dt.hour
    )

    weekday["session"] = (
        weekday["utc_hour"].apply(classify_utc_session)
    )

    summary = (
        weekday
        .groupby("session", sort=False)
        .agg(
            assessment_points=("assessment_number", "count"),
            permission_points=("directional_permission", "sum"),
        )
        .reset_index()
    )

    bullish = (
        weekday[
            weekday["directional_permission"]
            & (weekday["permitted_direction"] == "BULLISH")
        ]
        .groupby("session")
        .size()
    )

    bearish = (
        weekday[
            weekday["directional_permission"]
            & (weekday["permitted_direction"] == "BEARISH")
        ]
        .groupby("session")
        .size()
    )

    summary["bullish_points"] = (
        summary["session"].map(bullish).fillna(0).astype(int)
    )

    summary["bearish_points"] = (
        summary["session"].map(bearish).fillna(0).astype(int)
    )

    summary["permission_percentage"] = (
        summary["permission_points"]
        / summary["assessment_points"]
        * 100
    )

    summary["bullish_percentage"] = (
        summary["bullish_points"]
        / summary["assessment_points"]
        * 100
    )

    summary["bearish_percentage"] = (
        summary["bearish_points"]
        / summary["assessment_points"]
        * 100
    )

    return summary


# ============================================================
# SUMMARY OUTPUT
# ============================================================

def print_summary(
    alignment: pd.DataFrame,
    observed: pd.DataFrame,
    unmatched: pd.DataFrame,
    gaps: pd.DataFrame,
    state_runs: pd.DataFrame,
    directional_runs: pd.DataFrame,
    transitions: pd.DataFrame,
    losses: pd.DataFrame,
    reversals: pd.DataFrame,
    sessions: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:

    print("\n" + "=" * 68)
    print("PA-MS HISTORICAL ALIGNMENT TRANSITION ANALYSIS V2")
    print("=" * 68)

    print("\nOBSERVATION COVERAGE")

    print(f"Original assessments: {len(alignment):,}")
    print(f"Observed H1 assessments: {len(observed):,}")
    print(f"Unmatched assessments: {len(unmatched):,}")

    print(
        "Observed coverage: "
        f"{len(observed) / len(alignment) * 100:.2f}%"
    )

    print(f"Observation gaps: {len(gaps):,}")

    if not gaps.empty:
        print(
            "Total unobserved hours between observations: "
            f"{gaps['unobserved_hours'].sum():.0f}"
        )

    print("\nALIGNMENT STATE DURATIONS")

    if not state_runs.empty:

        state_summary = (
            state_runs
            .groupby("alignment_state")
            .agg(
                run_count=("alignment_state", "count"),
                average_observed_hours=("observed_h1_hours", "mean"),
                median_observed_hours=("observed_h1_hours", "median"),
                maximum_observed_hours=("observed_h1_hours", "max"),
                average_calendar_span=("calendar_span_hours", "mean"),
            )
            .round(2)
        )

        print(state_summary.to_string())

    print("\nDIRECTIONAL PERMISSION DURATIONS")

    if not directional_runs.empty:

        direction_summary = (
            directional_runs
            .groupby("direction")
            .agg(
                run_count=("direction", "count"),
                average_observed_hours=("observed_h1_hours", "mean"),
                median_observed_hours=("observed_h1_hours", "median"),
                maximum_observed_hours=("observed_h1_hours", "max"),
                average_calendar_span=("calendar_span_hours", "mean"),
            )
            .round(2)
        )

        print(direction_summary.to_string())

    print("\nSTATE TRANSITIONS")
    print(f"Observed state changes: {len(transitions):,}")

    if not transitions.empty:
        print(
            transitions
            .groupby(["previous_state", "current_state"])
            .size()
            .sort_values(ascending=False)
            .head(15)
            .to_string()
        )

        print(
            "\nTransitions following observation gaps: "
            f"{transitions['transition_after_gap'].sum()}"
        )

    print("\nDIRECTIONAL PERMISSION LOSSES")

    print(f"Total events: {len(losses):,}")

    if not losses.empty:

        print("\nEvent types:")
        print(losses["event_type"].value_counts().to_string())

        print("\nTimeframe observations:")
        print(losses["timeframe_result"].value_counts().to_string())

        print(
            "\nEvents following observation gaps: "
            f"{losses['transition_after_gap'].sum()}"
        )

    print("\nDIRECTIONAL REVERSALS")

    print(f"Opposite-direction run transitions: {len(reversals):,}")

    if not reversals.empty:
        print(
            reversals["transition_type"]
            .value_counts()
            .to_string()
        )

    print("\nDATA AVAILABILITY AND WARM-UP")

    print(
        availability[
            [
                "alignment_state",
                "assessment_count",
                "observed_h1_assessments",
                "unmatched_assessments",
                "observation_type",
            ]
        ].to_string(index=False)
    )

    print("\nWEEKDAY SESSION DISTRIBUTION")

    if not sessions.empty:
        print(sessions.round(2).to_string(index=False))

    print("\n" + "=" * 68)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    alignment = load_alignment_report()

    h1 = load_h1_market_observations()

    observed, unmatched = build_observed_timeline(
        alignment,
        h1,
    )

    gaps = build_observation_gaps(observed)

    print("\n[4] Calculating observed alignment runs")

    state_runs = build_runs(
        observed,
        mode="state",
    )

    directional_runs = build_runs(
        observed,
        mode="directional",
    )

    transitions, transition_matrix = (
        build_transition_analysis(observed)
    )

    losses = build_permission_loss_events(observed)

    reversals = build_reversal_analysis(
        directional_runs
    )

    availability = build_data_availability_summary(
        alignment,
        observed,
        unmatched,
    )

    sessions = build_session_analysis(observed)

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    print("\n[7] Saving V2 reports")

    outputs = {
        "observed_h1_timeline": observed,
        "observation_gaps": gaps,
        "state_runs": state_runs,
        "directional_runs": directional_runs,
        "state_transitions": transitions,
        "transition_matrix": transition_matrix,
        "permission_loss_events": losses,
        "directional_reversals": reversals,
        "data_availability": availability,
        "session_summary": sessions,
    }

    for name, report in outputs.items():

        output_path = (
            REPORT_DIR
            / f"{OUTPUT_PREFIX}_{name}.csv"
        )

        report.to_csv(
            output_path,
            index=(name == "transition_matrix"),
        )

        print(f"Saved: {output_path.name}")

    print_summary(
        alignment=alignment,
        observed=observed,
        unmatched=unmatched,
        gaps=gaps,
        state_runs=state_runs,
        directional_runs=directional_runs,
        transitions=transitions,
        losses=losses,
        reversals=reversals,
        sessions=sessions,
        availability=availability,
    )

    print("\nHistorical alignment transition analysis V2 completed.")


if __name__ == "__main__":
    main()
