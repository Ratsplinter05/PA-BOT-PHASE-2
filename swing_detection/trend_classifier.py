
from datetime import datetime
from typing import Iterable

from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.trend_models import TrendSnapshot


class TrendClassifier:

    def classify(
        self,
        swings: Iterable[ClassifiedSwing]
    ) -> list[TrendSnapshot]:

        records = list(swings)

        valid_classifications = {
            "HIGH": ("HH", "LH", "EQH"),
            "LOW": ("HL", "LL", "EQL"),
        }

        for item in records:
            if not isinstance(item, ClassifiedSwing):
                raise TypeError(
                    "All items must be ClassifiedSwing instances."
                )

            swing = item.swing

            if swing.swing_type not in ("HIGH", "LOW"):
                raise ValueError(
                    f"Invalid swing type: {swing.swing_type}"
                )

            if (
                item.classification is not None
                and item.classification not in (
                    "HH", "HL", "LH", "LL", "EQH", "EQL"
                )
            ):
                raise ValueError(
                    f"Invalid structure classification: "
                    f"{item.classification}"
                )

            if (
                item.classification is not None
                and item.classification not in
                valid_classifications[swing.swing_type]
            ):
                raise ValueError(
                    f"Invalid classification "
                    f"'{item.classification}' for "
                    f"{swing.swing_type} swing."
                )

        records.sort(
            key=lambda item: (
                item.swing.confirmation_time,
                item.swing.confirmation_index,
                item.swing.swing_index,
                item.swing.swing_type,
            )
        )

        latest_high = None
        latest_low = None

        snapshots = []

        # Process all swings confirmed at the same timestamp together.
        # This prevents temporary trend states when a high and low
        # are confirmed at the same time.

        grouped_records = {}

        for item in records:
            confirmation_time = item.swing.confirmation_time

            grouped_records.setdefault(
                confirmation_time, []
            ).append(item)

        for confirmation_time in sorted(grouped_records):

            confirmed_group = grouped_records[confirmation_time]

            for item in confirmed_group:

                if item.swing.swing_type == "HIGH":
                    latest_high = item

                elif item.swing.swing_type == "LOW":
                    latest_low = item

            high_label = (
                latest_high.classification
                if latest_high is not None
                else None
            )

            low_label = (
                latest_low.classification
                if latest_low is not None
                else None
            )

            # Insufficient structure or unclassified swings
            if (
                latest_high is None
                or latest_low is None
                or high_label not in ("HH", "LH", "EQH")
                or low_label not in ("HL", "LL", "EQL")
            ):

                trend = "UNCONFIRMED"
                reason = (
                    "Insufficient confirmed and classified "
                    "swing highs and lows."
                )

            # Equal levels indicate uncertain directional structure
            elif high_label == "EQH" or low_label == "EQL":

                trend = "MIXED"
                reason = (
                    "Equal highs or lows prevent a clear "
                    "directional classification."
                )

            # Bullish market structure
            elif high_label == "HH" and low_label == "HL":

                trend = "BULLISH"
                reason = (
                    "Latest confirmed high is HH and "
                    "latest confirmed low is HL."
                )

            # Bearish market structure
            elif high_label == "LH" and low_label == "LL":

                trend = "BEARISH"
                reason = (
                    "Latest confirmed high is LH and "
                    "latest confirmed low is LL."
                )

            # Conflicting structural signals
            else:

                trend = "MIXED"
                reason = (
                    "Latest confirmed high and low "
                    "show conflicting structural signals."
                )

            snapshots.append(
                TrendSnapshot(
                    as_of_time=confirmation_time,
                    trend=trend,
                    latest_high=latest_high,
                    latest_low=latest_low,
                    reason=reason,
                )
            )

        return snapshots
