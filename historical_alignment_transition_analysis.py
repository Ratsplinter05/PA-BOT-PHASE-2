
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "reports"
)

INPUT_FILE = (
    REPORT_DIR
    / "EURUSD_historical_alignment_validation.csv"
)

OUTPUT_PREFIX = "EURUSD_alignment"

ASSESSMENT_INTERVAL = pd.Timedelta(hours=1)

REQUIRED_COLUMNS = [
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


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def parse_boolean(value: Any) -> bool:
    """Convert CSV boolean values into actual Python booleans."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Invalid directional_permission value: {value!r}"
    )


def load_alignment_report() -> pd.DataFrame:
    """Load and validate the historical alignment report."""

    print("\n[1] Loading historical alignment report")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Historical alignment report not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The historical alignment report is empty.")

    df["assessment_number"] = pd.to_numeric(
        df["assessment_number"],
        errors="raise"
    )

    df["assessment_time"] = pd.to_datetime(
        df["assessment_time"],
        utc=True,
        errors="raise"
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
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
        )

    if df["assessment_time"].duplicated().any():
        raise ValueError(
            "Duplicate assessment timestamps detected."
        )

    if not df["assessment_time"].is_monotonic_increasing:
        raise ValueError(
            "Assessment timestamps are not chronological."
        )

    if df["alignment_state"].isna().any():
        raise ValueError(
            "Missing alignment_state values detected."
        )

    # Verify that permission and direction are consistent.
    permitted_rows = df["directional_permission"]

    invalid_permitted = (
        permitted_rows
        & ~df["permitted_direction"].isin(
            ["BULLISH", "BEARISH"]
        )
    )

    invalid_blocked = (
        ~permitted_rows
        & df["permitted_direction"].notna()
    )

    if invalid_permitted.any() or invalid_blocked.any():
        raise ValueError(
            "Directional permission and permitted direction "
            "are inconsistent."
        )

    time_differences = df["assessment_time"].diff().dropna()

    interval_gaps = time_differences[
        time_differences != ASSESSMENT_INTERVAL
    ]

    print(f"Loaded assessments: {len(df):,}")
    print(f"First assessment: {df['assessment_time'].iloc[0]}")
    print(f"Last assessment:  {df['assessment_time'].iloc[-1]}")
    print(f"Non-hourly intervals: {len(interval_gaps)}")

    if not interval_gaps.empty:
        print(
            "WARNING: Non-hourly intervals exist. "
            "Transitions across these gaps will not be counted."
        )

    return df


# ============================================================
# RUN IDENTIFICATION
# ============================================================

def build_runs(
    df: pd.DataFrame,
    mode: str
) -> pd.DataFrame:
    """
    Identify consecutive runs.

    mode:
        state       -> consecutive alignment states
        directional -> consecutive directional permission periods
    """

    if mode not in {"state", "directional"}:
        raise ValueError(f"Unsupported run mode: {mode}")

    runs: list[dict[str, Any]] = []

    current_rows: list[pd.Series] = []
    current_key: Any = None

    def save_current_run() -> None:
        nonlocal current_rows

        if not current_rows:
            return

        first = current_rows[0]
        last = current_rows[-1]

        start_time = first["assessment_time"]
        end_time = last["assessment_time"]

        duration_hours = (
            (end_time - start_time).total_seconds() / 3600
        ) + 1

        record = {
            "start_time": start_time,
            "end_time": end_time,
            "assessment_count": len(current_rows),
            "duration_hours": duration_hours,
            "duration_days": duration_hours / 24,
        }

        if mode == "state":
            record["alignment_state"] = first["alignment_state"]

        else:
            record["direction"] = first["permitted_direction"]

        runs.append(record)
        current_rows = []

    for _, row in df.iterrows():

        if mode == "state":
            eligible = True
            key = row["alignment_state"]

        else:
            eligible = bool(row["directional_permission"])
            key = row["permitted_direction"]

        if not eligible:
            save_current_run()
            current_key = None
            continue

        if current_rows:

            previous_time = current_rows[-1]["assessment_time"]

            is_contiguous = (
                row["assessment_time"] - previous_time
                == ASSESSMENT_INTERVAL
            )

            same_key = key == current_key

            if not is_contiguous or not same_key:
                save_current_run()

        if not current_rows:
            current_key = key

        current_rows.append(row)

    save_current_run()

    return pd.DataFrame(runs)


# ============================================================
# STATE TRANSITIONS
# ============================================================

def build_transition_analysis(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build transition events and a state transition matrix."""

    print("\n[2] Analysing alignment state transitions")

    transition_records: list[dict[str, Any]] = []

    for index in range(1, len(df)):

        previous = df.iloc[index - 1]
        current = df.iloc[index]

        elapsed = (
            current["assessment_time"]
            - previous["assessment_time"]
        )

        if elapsed != ASSESSMENT_INTERVAL:
            continue

        previous_state = previous["alignment_state"]
        current_state = current["alignment_state"]

        if previous_state == current_state:
            continue

        transition_records.append({
            "transition_time": current["assessment_time"],
            "previous_state": previous_state,
            "current_state": current_state,
            "previous_direction": previous["permitted_direction"],
            "current_direction": current["permitted_direction"],
            "previous_permission": previous["directional_permission"],
            "current_permission": current["directional_permission"],
        })

    transitions = pd.DataFrame(transition_records)

    # Build a matrix using only consecutive hourly observations.
    matrix_rows = []

    for index in range(1, len(df)):

        previous = df.iloc[index - 1]
        current = df.iloc[index]

        if (
            current["assessment_time"]
            - previous["assessment_time"]
            != ASSESSMENT_INTERVAL
        ):
            continue

        matrix_rows.append({
            "previous_state": previous["alignment_state"],
            "current_state": current["alignment_state"],
        })

    if matrix_rows:
        matrix_data = pd.DataFrame(matrix_rows)

        matrix = pd.crosstab(
            matrix_data["previous_state"],
            matrix_data["current_state"]
        )

        matrix.index.name = "previous_state"
        matrix.columns.name = "current_state"

    else:
        matrix = pd.DataFrame()

    print(f"State transitions identified: {len(transitions):,}")

    return transitions, matrix


# ============================================================
# DIRECTIONAL PERMISSION LOSS
# ============================================================

def build_permission_loss_events(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify when an aligned directional period ends.

    Determines whether H4, H1, or both no longer support
    the previous permitted direction at the next assessment.
    """

    print("\n[3] Analysing directional permission losses")

    records: list[dict[str, Any]] = []

    for index in range(1, len(df)):

        previous = df.iloc[index - 1]
        current = df.iloc[index]

        elapsed = (
            current["assessment_time"]
            - previous["assessment_time"]
        )

        if elapsed != ASSESSMENT_INTERVAL:
            continue

        if not previous["directional_permission"]:
            continue

        previous_direction = previous["permitted_direction"]

        if current["directional_permission"]:
            if current["permitted_direction"] == previous_direction:
                continue

            event_type = "DIRECT_DIRECTIONAL_REVERSAL"

        else:
            event_type = "DIRECTIONAL_PERMISSION_LOST"

        h4_retains_direction = (
            previous_direction == current["h4_trend"]
        )

        h1_retains_direction = (
            previous_direction == current["h1_trend"]
        )

        if not h4_retains_direction and not h1_retains_direction:
            timeframe_result = "BOTH_LOST_DIRECTION"

        elif not h4_retains_direction:
            timeframe_result = "H4_LOST_DIRECTION"

        elif not h1_retains_direction:
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
            "h4_snapshot_as_of": current["h4_snapshot_as_of"],
            "h4_known_at": current["h4_known_at"],
            "h1_snapshot_as_of": current["h1_snapshot_as_of"],
            "h1_known_at": current["h1_known_at"],
            "reason": current["reason"] if "reason" in df.columns else "",
        })

    result = pd.DataFrame(records)

    print(f"Permission loss/reversal events: {len(result):,}")

    return result


# ============================================================
# DIRECTIONAL REVERSAL ANALYSIS
# ============================================================

def build_reversal_analysis(
    directional_runs: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify changes from one directional permission run
    to the next, including reversals separated by blocked periods.
    """

    records: list[dict[str, Any]] = []

    if directional_runs.empty:
        return pd.DataFrame()

    for index in range(1, len(directional_runs)):

        previous = directional_runs.iloc[index - 1]
        current = directional_runs.iloc[index]

        previous_direction = previous["direction"]
        current_direction = current["direction"]

        if previous_direction == current_direction:
            continue

        previous_end = previous["end_time"]
        next_start = current["start_time"]

        blocked_hours = (
            next_start - previous_end
        ).total_seconds() / 3600 - 1

        records.append({
            "previous_direction": previous_direction,
            "previous_run_start": previous["start_time"],
            "previous_run_end": previous_end,
            "next_direction": current_direction,
            "next_run_start": next_start,
            "blocked_hours_between": max(0, blocked_hours),
            "transition_type": (
                f"{previous_direction}_TO_{current_direction}"
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# SESSION ANALYSIS
# ============================================================

def classify_utc_session(hour: int) -> str:
    """
    Approximate UTC session buckets.

    These are fixed UTC windows, not DST-adjusted exchange
    session definitions.
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
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyse directional permission by weekday UTC session.

    Saturday and Sunday are excluded. Session boundaries are
    approximate and do not adjust for daylight saving time.
    """

    print("\n[4] Analysing weekday session distribution")

    weekday_data = df[
        df["assessment_time"].dt.dayofweek < 5
    ].copy()

    weekday_data["utc_hour"] = (
        weekday_data["assessment_time"].dt.hour
    )

    weekday_data["session"] = (
        weekday_data["utc_hour"].apply(classify_utc_session)
    )

    summary = (
        weekday_data
        .groupby("session", sort=False)
        .agg(
            assessment_points=("assessment_number", "count"),
            permission_points=("directional_permission", "sum"),
        )
        .reset_index()
    )

    summary["bullish_points"] = (
        weekday_data[
            weekday_data["directional_permission"]
            & (weekday_data["permitted_direction"] == "BULLISH")
        ]
        .groupby("session")
        .size()
        .reindex(summary["session"], fill_value=0)
        .to_numpy()
    )

    summary["bearish_points"] = (
        weekday_data[
            weekday_data["directional_permission"]
            & (weekday_data["permitted_direction"] == "BEARISH")
        ]
        .groupby("session")
        .size()
        .reindex(summary["session"], fill_value=0)
        .to_numpy()
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
# SUMMARY REPORT
# ============================================================

def print_duration_summary(
    state_runs: pd.DataFrame,
    directional_runs: pd.DataFrame,
    transitions: pd.DataFrame,
    losses: pd.DataFrame,
    reversals: pd.DataFrame,
    sessions: pd.DataFrame,
) -> None:

    print("\n" + "=" * 65)
    print("PA-MS HISTORICAL ALIGNMENT TRANSITION ANALYSIS")
    print("=" * 65)

    print("\nALIGNMENT STATE DURATIONS")

    if state_runs.empty:
        print("No state runs found.")

    else:
        state_summary = (
            state_runs
            .groupby("alignment_state")
            .agg(
                run_count=("alignment_state", "count"),
                average_hours=("duration_hours", "mean"),
                median_hours=("duration_hours", "median"),
                maximum_hours=("duration_hours", "max"),
            )
            .round(2)
        )

        print(state_summary.to_string())

    print("\nDIRECTIONAL PERMISSION DURATIONS")

    if directional_runs.empty:
        print("No directional permission runs found.")

    else:
        direction_summary = (
            directional_runs
            .groupby("direction")
            .agg(
                run_count=("direction", "count"),
                average_hours=("duration_hours", "mean"),
                median_hours=("duration_hours", "median"),
                maximum_hours=("duration_hours", "max"),
            )
            .round(2)
        )

        print(direction_summary.to_string())

    print("\nSTATE TRANSITIONS")
    print(f"Total state changes: {len(transitions):,}")

    if not transitions.empty:
        print(
            transitions
            .groupby(
                ["previous_state", "current_state"]
            )
            .size()
            .sort_values(ascending=False)
            .head(15)
            .to_string()
        )

    print("\nDIRECTIONAL PERMISSION LOSSES AND REVERSALS")
    print(f"Total events: {len(losses):,}")

    if not losses.empty:
        print("\nEvent types:")
        print(losses["event_type"].value_counts().to_string())

        print("\nTimeframe observations:")
        print(losses["timeframe_result"].value_counts().to_string())

    print("\nDIRECTIONAL REVERSALS BETWEEN PERMISSION RUNS")
    print(f"Opposite-direction run transitions: {len(reversals):,}")

    if not reversals.empty:
        print(
            reversals["transition_type"]
            .value_counts()
            .to_string()
        )

    print("\nWEEKDAY SESSION DISTRIBUTION")

    if not sessions.empty:
        print(
            sessions.round(2).to_string(index=False)
        )

    print("\n" + "=" * 65)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_alignment_report()

    print("\n[2] Identifying alignment state runs")

    state_runs = build_runs(df, mode="state")

    print("\n[3] Identifying directional permission runs")

    directional_runs = build_runs(
        df,
        mode="directional"
    )

    transitions, transition_matrix = (
        build_transition_analysis(df)
    )

    losses = build_permission_loss_events(df)

    reversals = build_reversal_analysis(
        directional_runs
    )

    sessions = build_session_analysis(df)

    # --------------------------------------------------------
    # Save detailed outputs
    # --------------------------------------------------------

    print("\n[5] Saving analysis reports")

    outputs = {
        "state_runs": state_runs,
        "directional_runs": directional_runs,
        "state_transitions": transitions,
        "transition_matrix": transition_matrix,
        "permission_loss_events": losses,
        "directional_reversals": reversals,
        "session_summary": sessions,
    }

    for report_name, report_df in outputs.items():

        output_path = (
            REPORT_DIR
            / f"{OUTPUT_PREFIX}_{report_name}.csv"
        )

        report_df.to_csv(
            output_path,
            index=True if report_name == "transition_matrix" else False
        )

        print(f"Saved: {output_path.name}")

    print_duration_summary(
        state_runs=state_runs,
        directional_runs=directional_runs,
        transitions=transitions,
        losses=losses,
        reversals=reversals,
        sessions=sessions,
    )

    print("\nHistorical alignment transition analysis completed.")


if __name__ == "__main__":
    main()
