
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from research.swing_detection.alignment_models import AlignmentResult
from research.swing_detection.trend_models import TrendSnapshot


class AlignmentEngine:

    H4_DURATION = timedelta(hours=4)
    H1_DURATION = timedelta(hours=1)

    VALID_TRENDS = {
        "BULLISH",
        "BEARISH",
        "MIXED",
        "UNCONFIRMED",
    }

    def _validate_timestamp(
        self,
        value: datetime,
        field_name: str,
    ) -> None:

        if not isinstance(value, datetime):
            raise TypeError(
                f"{field_name} must be a datetime."
            )

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{field_name} must be timezone-aware."
            )

    def _validate_trend_state(
        self,
        trend: str,
        timeframe_name: str,
    ) -> None:

        if (
            not isinstance(trend, str)
            or trend not in self.VALID_TRENDS
        ):
            raise ValueError(
                f"Unsupported {timeframe_name} trend state: "
                f"{trend}"
            )

    def _known_at(
        self,
        snapshot: TrendSnapshot,
        duration: timedelta,
    ) -> datetime:

        self._validate_timestamp(
            snapshot.as_of_time,
            "snapshot.as_of_time",
        )

        return snapshot.as_of_time + duration

    def _select_latest_snapshot(
        self,
        snapshots: Iterable[TrendSnapshot],
        duration: timedelta,
        assessment_time: datetime,
        timeframe_name: str,
    ) -> tuple[
        Optional[TrendSnapshot],
        Optional[datetime],
    ]:

        eligible = []
        seen_timestamps = set()

        for snapshot in snapshots:

            if not isinstance(snapshot, TrendSnapshot):
                raise TypeError(
                    "All snapshot inputs must be TrendSnapshot objects."
                )

            self._validate_trend_state(
                snapshot.trend,
                timeframe_name,
            )

            known_at = self._known_at(
                snapshot,
                duration,
            )

            # Normalize timestamps to UTC before checking duplicates.
            normalized_time = snapshot.as_of_time.astimezone(
                timezone.utc
            )

            if normalized_time in seen_timestamps:
                raise ValueError(
                    f"duplicate {timeframe_name} snapshot timestamp: "
                    f"{normalized_time.isoformat()}"
                )

            seen_timestamps.add(normalized_time)

            if known_at <= assessment_time:
                eligible.append((known_at, snapshot))

        if not eligible:
            return None, None

        eligible.sort(
            key=lambda item: (
                item[0],
                item[1].as_of_time.astimezone(timezone.utc),
            )
        )

        known_at, snapshot = eligible[-1]

        return snapshot, known_at

    def _classify_alignment(
        self,
        h4_trend: Optional[str],
        h1_trend: Optional[str],
    ) -> tuple[str, bool, Optional[str], str]:

        if h4_trend is None:
            return (
                "H4_MISSING",
                False,
                None,
                "No H4 trend snapshot was available at the assessment time.",
            )

        if h1_trend is None:
            return (
                "H1_MISSING",
                False,
                None,
                "No H1 trend snapshot was available at the assessment time.",
            )

        if h4_trend == "UNCONFIRMED":
            return (
                "H4_UNCONFIRMED",
                False,
                None,
                "H4 structure is not sufficiently confirmed.",
            )

        if h4_trend == "MIXED":
            return (
                "H4_MIXED",
                False,
                None,
                "H4 structure is mixed. Directional permission is withheld.",
            )

        if h1_trend == "UNCONFIRMED":
            return (
                "H1_UNCONFIRMED",
                False,
                None,
                "H1 structure is not sufficiently confirmed.",
            )

        if h1_trend == "MIXED":
            return (
                "H1_MIXED",
                False,
                None,
                "H1 structure is mixed. Directional permission is withheld.",
            )

        if h4_trend == h1_trend == "BULLISH":
            return (
                "ALIGNED_BULLISH",
                True,
                "BULLISH",
                "H4 and H1 structures are aligned bullish.",
            )

        if h4_trend == h1_trend == "BEARISH":
            return (
                "ALIGNED_BEARISH",
                True,
                "BEARISH",
                "H4 and H1 structures are aligned bearish.",
            )

        return (
            "CONFLICT",
            False,
            None,
            "H4 and H1 directional structures conflict.",
        )

    def evaluate(
        self,
        h4_snapshots: Iterable[TrendSnapshot],
        h1_snapshots: Iterable[TrendSnapshot],
        assessment_time: datetime,
    ) -> AlignmentResult:

        self._validate_timestamp(
            assessment_time,
            "assessment_time",
        )

        if assessment_time.utcoffset() != timedelta(0):
            assessment_time = assessment_time.astimezone(
                timezone.utc
            )

        h4_snapshot, h4_known_at = self._select_latest_snapshot(
            h4_snapshots,
            self.H4_DURATION,
            assessment_time,
            "H4",
        )

        h1_snapshot, h1_known_at = self._select_latest_snapshot(
            h1_snapshots,
            self.H1_DURATION,
            assessment_time,
            "H1",
        )

        h4_trend = (
            h4_snapshot.trend
            if h4_snapshot is not None
            else None
        )

        h1_trend = (
            h1_snapshot.trend
            if h1_snapshot is not None
            else None
        )

        (
            alignment_state,
            directional_permission,
            permitted_direction,
            reason,
        ) = self._classify_alignment(
            h4_trend,
            h1_trend,
        )

        return AlignmentResult(
            assessment_time=assessment_time,
            alignment_state=alignment_state,
            directional_permission=directional_permission,
            permitted_direction=permitted_direction,
            h4_snapshot=h4_snapshot,
            h1_snapshot=h1_snapshot,
            h4_known_at_time=h4_known_at,
            h1_known_at_time=h1_known_at,
            reason=reason,
        )
