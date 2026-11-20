from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from research.swing_detection.fractal_detector import FractalSwingDetector
from research.swing_detection.structure_classifier import StructureClassifier
from research.swing_detection.trend_classifier import TrendClassifier
from research.swing_detection.bos_detector import BOSDetector
from research.swing_detection.protected_swing_tracker import ProtectedSwingTracker
from research.swing_detection.protected_level_registry import ProtectedLevelRegistry
from research.swing_detection.choch_detector import CHoCHDetector


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def same_swing(first, second):
    """
    Determine whether two ClassifiedSwing records refer
    to the same underlying swing.
    """

    first_swing = first.swing
    second_swing = second.swing

    return (
        first_swing.swing_type == second_swing.swing_type
        and first_swing.swing_index == second_swing.swing_index
        and first_swing.confirmation_index
        == second_swing.confirmation_index
    )


def find_existing_level(registry, protected):
    """
    Find an existing registry level using its unique
    protected swing identity.
    """

    for level in registry.levels:

        if (
            level.trend == protected.trend
            and same_swing(level.swing, protected.swing)
        ):
            return level

    return None


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\n========== PA-MS CHoCH ANALYSIS ==========")

    # --------------------------------------------------------
    # 1. LOAD HISTORICAL DATA
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_FILE}"
        )

    candles = pd.read_parquet(DATA_FILE)

    print(f"\nHistorical candles: {len(candles)}")

    if candles.empty:
        raise ValueError("Historical dataset is empty.")

    # --------------------------------------------------------
    # 2. DETECT FRACTAL SWINGS
    # --------------------------------------------------------

    swing_detector = FractalSwingDetector(
        left_bars=2,
        right_bars=2,
    )

    swings = swing_detector.detect(candles)

    # --------------------------------------------------------
    # 3. CLASSIFY SWINGS
    # --------------------------------------------------------

    structure_classifier = StructureClassifier()

    classified_swings = structure_classifier.classify(
        swings
    )

    # --------------------------------------------------------
    # 4. CLASSIFY MARKET TREND
    # --------------------------------------------------------

    trend_classifier = TrendClassifier()

    trend_snapshots = trend_classifier.classify(
        classified_swings
    )

    # --------------------------------------------------------
    # 5. DETECT BREAK OF STRUCTURE
    # --------------------------------------------------------

    bos_detector = BOSDetector()

    bos_events = bos_detector.detect(
        candles,
        classified_swings,
        trend_snapshots,
        timeframe="H1", 
    )

    # --------------------------------------------------------
    # 6. TRACK PROTECTED SWINGS
    # --------------------------------------------------------

    tracker = ProtectedSwingTracker()

    protected_swings = tracker.track(
        classified_swings,
        bos_events,
    )

    protected_swings = sorted(
        protected_swings,
        key=lambda item: (
            item.activation_index,
            item.established_by.break_index,
        ),
    )

    print(f"Confirmed swings: {len(swings)}")
    print(f"Classified swings: {len(classified_swings)}")
    print(f"Trend snapshots: {len(trend_snapshots)}")
    print(f"BOS events: {len(bos_events)}")
    print(
        "Protected swing associations: "
        f"{len(protected_swings)}"
    )

    # --------------------------------------------------------
    # 7. INITIALISE CHOCH PROCESSING
    # --------------------------------------------------------

    registry = ProtectedLevelRegistry()
    choch_detector = CHoCHDetector()

    protected_pointer = 0

    choch_events = []

    registration_count = 0
    skipped_retired_associations = 0
    retirement_count = 0

    # --------------------------------------------------------
    # 8. PROCESS HISTORICAL CANDLES CHRONOLOGICALLY
    # --------------------------------------------------------

    print("\nProcessing candles chronologically...")

    for candle_index in range(1, len(candles)):

        # Register protected levels only after their
        # activation candle has arrived.

        while (
            protected_pointer < len(protected_swings)
            and protected_swings[
                protected_pointer
            ].activation_index <= candle_index
        ):

            protected = protected_swings[
                protected_pointer
            ]

            existing_level = find_existing_level(
                registry,
                protected,
            )

            if (
                existing_level is not None
                and not existing_level.active
            ):

                # A previously retired level must never
                # be reactivated by a repeated BOS association.

                skipped_retired_associations += 1

            else:

                # New levels are registered.
                # Existing active levels can receive
                # additional BOS associations.

                registry.register([protected])

                registration_count += 1

            protected_pointer += 1

        # ----------------------------------------------------
        # DETECT CHOCH ON THE CURRENT CANDLE ONLY
        # ----------------------------------------------------

        # Important:
        # The detector receives only candles available
        # up to the current candle.
        #
        # This prevents future candles from influencing
        # historical CHoCH detection.

        candle_history = candles.iloc[
            : candle_index + 1
        ]

        events = choch_detector.detect(
            candle_history,
            registry,
            start_index=candle_index,
        )

        # ----------------------------------------------------
        # RETIRE BROKEN PROTECTED LEVELS IMMEDIATELY
        # ----------------------------------------------------

        for event in events:

            matching_levels = [
                level
                for level in registry.get_active_levels(
                    as_of_index=candle_index
                )
                if (
                    level.trend == event.prior_trend
                    and same_swing(
                        level.swing,
                        event.level_swing,
                    )
                )
            ]

            if not matching_levels:

                raise RuntimeError(
                    "CHoCH event has no matching active "
                    "protected level. "
                    f"Break index: {event.break_index}"
                )

            for level in matching_levels:

                registry.retire(
                    level,
                    event,
                )

                retirement_count += 1

            choch_events.append(event)

    # --------------------------------------------------------
    # 9. BUILD CHOCH EVENT REPORT
    # --------------------------------------------------------

    choch_records = []

    for event in choch_events:

        swing = event.level_swing.swing

        choch_records.append({

            "direction": event.direction,

            "break_time": event.break_time,

            "break_index": event.break_index,

            "broken_level": event.level_price,

            "close_price": event.close_price,

            "prior_trend": event.prior_trend,

            "protected_swing_type": swing.swing_type,

            "protected_swing_classification": (
                event.level_swing.classification
            ),

            "protected_swing_price": swing.price,

            "protected_swing_index": swing.swing_index,

            "protected_confirmation_index": (
                swing.confirmation_index
            ),

        })

    choch_report = pd.DataFrame(
        choch_records
    )

    # --------------------------------------------------------
    # 10. BUILD FINAL REGISTRY AUDIT
    # --------------------------------------------------------

    registry_records = []

    for level in registry.levels:

        swing = level.swing.swing

        registry_records.append({

            "trend": level.trend,

            "swing_type": swing.swing_type,

            "classification": level.swing.classification,

            "protected_price": level.price,

            "swing_index": swing.swing_index,

            "confirmation_index": (
                swing.confirmation_index
            ),

            "activation_index": level.activation_index,

            "first_bos_index": (
                level.established_by.break_index
            ),

            "first_bos_time": (
                level.established_by.break_time
            ),

            "bos_event_count": len(
                level.bos_events
            ),

            "active_at_end": level.active,

            "retired_by_direction": (
                level.retired_by.direction
                if level.retired_by is not None
                else None
            ),

            "retirement_index": (
                level.retired_by.break_index
                if level.retired_by is not None
                else None
            ),

            "retirement_time": (
                level.retired_by.break_time
                if level.retired_by is not None
                else None
            ),

        })

    registry_report = pd.DataFrame(
        registry_records
    )

    # --------------------------------------------------------
    # 11. CALCULATE SUMMARY STATISTICS
    # --------------------------------------------------------

    bullish_choch = sum(
        event.direction == "BULLISH"
        for event in choch_events
    )

    bearish_choch = sum(
        event.direction == "BEARISH"
        for event in choch_events
    )

    active_levels = registry.get_active_levels()

    retired_levels = registry.get_retired_levels()

    unprocessed_associations = (
        len(protected_swings) - protected_pointer
    )

    # --------------------------------------------------------
    # 12. DISPLAY CHOCH SUMMARY
    # --------------------------------------------------------

    print("\n========== CHOCH SUMMARY ==========")

    print(
        f"Total CHoCH events: {len(choch_events)}"
    )

    print(
        f"Bullish CHoCH: {bullish_choch}"
    )

    print(
        f"Bearish CHoCH: {bearish_choch}"
    )

    # --------------------------------------------------------
    # 13. DISPLAY REGISTRY SUMMARY
    # --------------------------------------------------------

    print("\n========== REGISTRY SUMMARY ==========")

    print(
        "Registered associations: "
        f"{registration_count}"
    )

    print(
        f"Unique registered levels: {len(registry.levels)}"
    )

    print(
        f"Retirement operations: {retirement_count}"
    )

    print(
        "Skipped retired-level associations: "
        f"{skipped_retired_associations}"
    )

    print(
        f"Active levels at dataset end: {len(active_levels)}"
    )

    print(
        f"Retired levels: {len(retired_levels)}"
    )

    print(
        "Unprocessed associations: "
        f"{unprocessed_associations}"
    )

    # --------------------------------------------------------
    # 14. DISPLAY LATEST CHOCH EVENTS
    # --------------------------------------------------------

    if not choch_report.empty:

        print(
            "\n========== LATEST 15 CHOCH EVENTS =========="
        )

        print(
            choch_report.tail(15).to_string(
                index=False
            )
        )

    else:

        print("\nNo CHoCH events were detected.")

    # --------------------------------------------------------
    # 15. DISPLAY FINAL REGISTRY RECORDS
    # --------------------------------------------------------

    if not registry_report.empty:

        print(
            "\n========== LATEST 10 REGISTRY RECORDS =========="
        )

        print(
            registry_report.tail(10).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 16. SAVE REPORTS
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    choch_file = (
        REPORT_DIR
        / f"choch_analysis_{timestamp}.csv"
    )

    registry_file = (
        REPORT_DIR
        / f"protected_registry_final_{timestamp}.csv"
    )

    choch_report.to_csv(
        choch_file,
        index=False,
    )

    registry_report.to_csv(
        registry_file,
        index=False,
    )

    # --------------------------------------------------------
    # 17. FINAL OUTPUT
    # --------------------------------------------------------

    print("\n========== REPORTS SAVED ==========")

    print(choch_file)

    print(registry_file)

    print("\nCHoCH analysis completed successfully.")


if __name__ == "__main__":

    main()
