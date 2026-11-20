
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from research.swing_detection.fractal_detector import FractalSwingDetector
from research.swing_detection.structure_classifier import StructureClassifier
from research.swing_detection.trend_classifier import TrendClassifier
from research.swing_detection.bos_detector import BOSDetector
from research.swing_detection.protected_swing_tracker import ProtectedSwingTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "raw"
    / "EURUSD"
    / "EURUSD_H1.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical"
    / "reports"
)


def main():

    print("\n========== PA-MS PROTECTED SWING ANALYSIS ==========")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_FILE}"
        )

    # 1. Load historical candles
    candles = pd.read_parquet(DATA_FILE)

    print(f"\nHistorical candles loaded: {len(candles)}")
    print(f"Dataset: {DATA_FILE}")

    # 2. Detect fractal swings
    swing_detector = FractalSwingDetector(
        left_bars=2,
        right_bars=2,
    )

    swings = swing_detector.detect(candles)

    print(f"Confirmed swings: {len(swings)}")

    # 3. Classify market structure
    structure_classifier = StructureClassifier()

    classified_swings = structure_classifier.classify(swings)

    print(f"Classified swings: {len(classified_swings)}")

    # 4. Generate trend snapshots
    trend_classifier = TrendClassifier()

    trend_snapshots = trend_classifier.classify(
        classified_swings
    )

    print(f"Trend snapshots: {len(trend_snapshots)}")

    # 5. Detect Breaks of Structure
    bos_detector = BOSDetector()

    bos_events = bos_detector.detect(
        candles,
        classified_swings,
        trend_snapshots,
        timeframe="H1",
    )

    print(f"BOS events detected: {len(bos_events)}")

    # 6. Identify protected swings
    protected_tracker = ProtectedSwingTracker()

    protected_swings = protected_tracker.track(
        classified_swings,
        bos_events,
    )

    print(
        f"Protected swings identified: "
        f"{len(protected_swings)}"
    )

    # 7. Prepare detailed audit report
    records = []

    ordered_bos = sorted(
        bos_events,
        key=lambda event: (
            event.break_index,
            event.break_time,
        ),
    )

    for protected in protected_swings:

        event = protected.established_by
        swing = protected.swing.swing

        previous_opposite_events = [
            previous
            for previous in ordered_bos
            if (
                previous.break_index < event.break_index
                and previous.direction != event.direction
            )
        ]

        last_opposite_event = (
            previous_opposite_events[-1]
            if previous_opposite_events
            else None
        )

        if last_opposite_event is None:
            formed_after_opposite_bos = None
            opposite_bos_index = None
            opposite_bos_time = None

        else:
            opposite_bos_index = last_opposite_event.break_index
            opposite_bos_time = last_opposite_event.break_time

            formed_after_opposite_bos = (
                swing.confirmation_index
                > last_opposite_event.break_index
            )

        records.append({
            "trend": protected.trend,
            "protected_type": swing.swing_type,
            "classification": protected.swing.classification,
            "protected_price": protected.price,
            "swing_time": swing.swing_time,
            "confirmation_time": swing.confirmation_time,
            "swing_index": swing.swing_index,
            "confirmation_index": swing.confirmation_index,
            "bos_direction": event.direction,
            "bos_time": event.break_time,
            "bos_index": event.break_index,
            "bos_level": event.level_price,
            "bos_close": event.close_price,
            "activation_index": protected.activation_index,
            "previous_opposite_bos_time": opposite_bos_time,
            "previous_opposite_bos_index": opposite_bos_index,
            "formed_after_opposite_bos": formed_after_opposite_bos,
        })

    report = pd.DataFrame(records)

    # 8. Summarise results
    bullish_bos = sum(
        event.direction == "BULLISH"
        for event in bos_events
    )

    bearish_bos = sum(
        event.direction == "BEARISH"
        for event in bos_events
    )

    bullish_protected = sum(
        item.trend == "BULLISH"
        for item in protected_swings
    )

    bearish_protected = sum(
        item.trend == "BEARISH"
        for item in protected_swings
    )

    print("\n========== BOS SUMMARY ==========")
    print(f"Bullish BOS: {bullish_bos}")
    print(f"Bearish BOS: {bearish_bos}")

    print("\n========== PROTECTED SWING SUMMARY ==========")
    print(f"Bullish protected HLs: {bullish_protected}")
    print(f"Bearish protected LHs: {bearish_protected}")

    if not report.empty:

        stale_levels = report[
            report["formed_after_opposite_bos"] == False
        ]

        unchecked_levels = report[
            report["formed_after_opposite_bos"].isna()
        ]

        print("\n========== STRUCTURAL AUDIT ==========")
        print(f"Total protected levels: {len(report)}")
        print(f"Potentially stale levels: {len(stale_levels)}")
        print(f"Levels without prior opposite BOS: {len(unchecked_levels)}")

        print("\n========== LATEST 10 PROTECTED SWINGS ==========")

        display_columns = [
            "trend",
            "classification",
            "protected_price",
            "bos_time",
            "activation_index",
            "formed_after_opposite_bos",
        ]

        print(
            report[display_columns]
            .tail(10)
            .to_string(index=False)
        )

    # 9. Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    report_file = (
        REPORT_DIR
        / f"protected_swing_analysis_{timestamp}.csv"
    )

    report.to_csv(report_file, index=False)

    print("\n========== REPORT SAVED ==========")
    print(report_file)

    print("\nProtected swing analysis completed.")


if __name__ == "__main__":
    main()
