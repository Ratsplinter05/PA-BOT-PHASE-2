from datetime import datetime, timedelta, timezone

from src.config import load_settings
from src.logger import get_logger

from research.historical_storage import HistoricalDataStorage
from research.swing_detection.fractal_detector import FractalSwingDetector
from research.swing_detection.structure_classifier import StructureClassifier
from research.swing_detection.trend_classifier import TrendClassifier


log = get_logger("H1_TREND_ANALYSIS")


def swing_details(classified_swing):
    if classified_swing is None:
        return {
            "type": None,
            "classification": None,
            "price": None,
            "swing_time": None,
            "confirmation_candle_open_time": None,
            "known_at_time": None,
            "swing_index": None,
            "confirmation_index": None,
        }

    swing = classified_swing.swing

    return {
        "type": swing.swing_type,
        "classification": classified_swing.classification,
        "price": swing.price,
        "swing_time": swing.swing_time,
        "confirmation_candle_open_time": swing.confirmation_time,
        "known_at_time": (
            swing.confirmation_time + timedelta(hours=1)
        ),
        "swing_index": swing.swing_index,
        "confirmation_index": swing.confirmation_index,
    }


def main():

    settings = load_settings()

    trading = settings.get("trading", {})
    market = settings.get("market", {})
    research = settings.get("research", {})
    historical = research.get("historical_data", {})

    # SAFETY CHECKS

    if trading.get("execution_enabled") is not False:
        raise RuntimeError(
            "H1 analysis blocked: trade execution must remain disabled."
        )

    if historical.get("enabled") is not True:
        raise RuntimeError("Historical research is disabled.")

    if "H1" not in market.get("timeframes", {}):
        raise RuntimeError("H1 is not configured in settings.yaml.")

    instrument = "EURUSD"
    timeframe = "H1"

    storage = HistoricalDataStorage()

    print("\n========== H1 HISTORICAL DATA ==========")

    candles = storage.load_dataset(instrument, timeframe)

    if candles.empty:
        raise RuntimeError("H1 historical dataset is empty.")

    if "is_forming_candle" in candles.columns:
        if candles["is_forming_candle"].fillna(False).any():
            raise RuntimeError(
                "Dataset contains forming candles. Analysis blocked."
            )

    if len(candles) < 10:
        raise RuntimeError("Insufficient H1 candles for swing analysis.")

    # The last retained candle is closed.
    # Its closing time is one hour after its opening timestamp.

    dataset_as_of_time = (
        candles["time"].iloc[-1].to_pydatetime()
        + timedelta(hours=1)
    )

    print(f"Instrument: {instrument}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles: {len(candles)}")
    print(f"First candle: {candles['time'].iloc[0]}")
    print(f"Last candle: {candles['time'].iloc[-1]}")
    print(f"Analysis as-of time: {dataset_as_of_time}")

    # STEP 1: DETECT CONFIRMED SWINGS

    print("\n========== H1 SWING DETECTION ==========")

    swing_detector = FractalSwingDetector(
        left_bars=2,
        right_bars=2,
    )

    swings = swing_detector.detect(candles)

    print(f"Confirmed swing points: {len(swings)}")

    # STEP 2: CLASSIFY MARKET STRUCTURE

    print("\n========== H1 STRUCTURE CLASSIFICATION ==========")

    structure_classifier = StructureClassifier(
        price_tolerance=0.0,
    )

    classified_swings = structure_classifier.classify(swings)

    structure_counts = {}

    for item in classified_swings:
        label = item.classification or "UNCLASSIFIED"
        structure_counts[label] = (
            structure_counts.get(label, 0) + 1
        )

    print(f"Classified swing records: {len(classified_swings)}")

    for label, count in sorted(structure_counts.items()):
        print(f"{label}: {count}")

    # STEP 3: CLASSIFY TREND

    print("\n========== H1 TREND CLASSIFICATION ==========")

    trend_classifier = TrendClassifier()

    trend_snapshots = trend_classifier.classify(
        classified_swings
    )

    print(f"Trend snapshots generated: {len(trend_snapshots)}")

    # STEP 4: CONVERT SNAPSHOTS TO ACTUAL KNOWABLE TIMES

    # TrendClassifier.as_of_time stores the confirmation
    # candle's opening time.
    #
    # The swing becomes knowable after that H1 candle closes.

    trend_records = []

    for snapshot in trend_snapshots:

        known_at_time = (
            snapshot.as_of_time + timedelta(hours=1)
        )

        high = swing_details(snapshot.latest_high)
        low = swing_details(snapshot.latest_low)

        trend_records.append({
            "instrument": instrument,
            "timeframe": timeframe,

            "confirmation_candle_open_time": snapshot.as_of_time,
            "known_at_time": known_at_time,

            "trend": snapshot.trend,
            "reason": snapshot.reason,

            "latest_high_classification": high["classification"],
            "latest_high_price": high["price"],
            "latest_high_swing_time": high["swing_time"],
            "latest_high_confirmation_time": (
                high["confirmation_candle_open_time"]
            ),

            "latest_low_classification": low["classification"],
            "latest_low_price": low["price"],
            "latest_low_swing_time": low["swing_time"],
            "latest_low_confirmation_time": (
                low["confirmation_candle_open_time"]
            ),
        })

    # Only snapshots genuinely available by the latest
    # dataset candle close can contribute to the latest bias.

    eligible_snapshots = [
        record
        for record in trend_records
        if record["known_at_time"] <= dataset_as_of_time
    ]

    if eligible_snapshots:

        latest_snapshot = max(
            eligible_snapshots,
            key=lambda record: record["known_at_time"],
        )

    else:
        latest_snapshot = None

    if latest_snapshot is None:

        latest_bias = "UNCONFIRMED"

        latest_reason = (
            "No trend snapshot was confirmed and available "
            "by the analysis as-of time."
        )

        latest_known_at = None
        latest_high = None
        latest_low = None

    else:

        latest_bias = latest_snapshot["trend"]
        latest_reason = latest_snapshot["reason"]
        latest_known_at = latest_snapshot["known_at_time"]

        latest_high = (
            latest_snapshot["latest_high_classification"]
        )

        latest_low = (
            latest_snapshot["latest_low_classification"]
        )

    print("\n========== LATEST H1 DIRECTIONAL BIAS ==========")

    print(f"Analysis as-of: {dataset_as_of_time}")
    print(f"Latest known bias: {latest_bias}")
    print(f"Last bias confirmation: {latest_known_at}")
    print(f"Latest high classification: {latest_high}")
    print(f"Latest low classification: {latest_low}")
    print(f"Reason: {latest_reason}")

    # STEP 5: PREPARE STRUCTURE REPORT

    structure_records = []

    for item in classified_swings:

        details = swing_details(item)

        structure_records.append({
            "instrument": instrument,
            "timeframe": timeframe,
            **details,
            "previous_price": item.previous_price,
            "previous_swing_time": item.previous_swing_time,
        })

    # STEP 6: PREPARE LATEST BIAS REPORT

    bias_record = {
        "instrument": instrument,
        "timeframe": timeframe,
        "analysis_as_of_time": dataset_as_of_time,
        "latest_bias": latest_bias,
        "last_bias_known_at": latest_known_at,
        "latest_high_classification": latest_high,
        "latest_low_classification": latest_low,
        "reason": latest_reason,
        "candles_analyzed": len(candles),
        "confirmed_swings": len(swings),
        "trend_snapshots": len(trend_snapshots),
    }

    # STEP 7: SAVE REPORTS

    print("\n========== SAVING H1 REPORTS ==========")

    report_timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")

    structure_path = storage.save_report(
        structure_records,
        f"EURUSD_H1_structure_{report_timestamp}",
        overwrite=False,
    )

    trend_path = storage.save_report(
        trend_records,
        f"EURUSD_H1_trend_history_{report_timestamp}",
        overwrite=False,
    )

    bias_path = storage.save_report(
        [bias_record],
        f"EURUSD_H1_latest_bias_{report_timestamp}",
        overwrite=False,
    )

    print(f"Structure report: {structure_path}")
    print(f"Trend history: {trend_path}")
    print(f"Latest bias: {bias_path}")

    # STEP 8: FINAL SUMMARY

    print("\n========== H1 ANALYSIS SUMMARY ==========")

    print(f"Instrument: {instrument}")
    print(f"Timeframe: {timeframe}")
    print(f"Historical candles: {len(candles)}")
    print(f"Confirmed swings: {len(swings)}")
    print(f"Trend snapshots: {len(trend_snapshots)}")
    print(f"Latest H1 bias: {latest_bias}")
    print(f"Analysis as-of time: {dataset_as_of_time}")
    print(
        f"Execution enabled: "
        f"{trading.get('execution_enabled')}"
    )

    print("\nH1 TREND ANALYSIS COMPLETED.")


if __name__ == "__main__":
    main()
