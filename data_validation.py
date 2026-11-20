
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


REQUIRED_COLUMNS = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
}


class CandleValidator:
    """
    Validates completed market candles before strategy analysis.
    """

    @staticmethod
    def _validate_forming_candle_flags(flags):
        """
        Require genuine Boolean forming-candle flags.

        BooleanDtype is accepted when no values are missing.
        Strings, numbers and other non-Boolean types are rejected.
        """

        if flags.isna().any():
            return "Forming-candle status contains missing values."

        if not is_bool_dtype(flags.dtype):
            return (
                "Forming-candle status must contain Boolean values only."
            )

        if flags.any():
            return "Forming candles must not enter strategy analysis."

        return None

    def validate(self, candles, timeframe=None, check_future=True):

        errors = []
        warnings = []

        if not isinstance(candles, pd.DataFrame):
            return self._result(
                False,
                ["Input must be a pandas DataFrame."],
                warnings
            )

        if candles.empty:
            return self._result(
                False,
                ["Candle dataset is empty."],
                warnings
            )

        missing_columns = REQUIRED_COLUMNS - set(candles.columns)

        if missing_columns:
            errors.append(
                f"Missing required columns: {sorted(missing_columns)}"
            )

            return self._result(False, errors, warnings)

        df = candles.copy()

        # --------------------------------------------------
        # TIMESTAMP VALIDATION
        # --------------------------------------------------

        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            errors.append("Candle timestamps must be datetime values.")

        else:

            if df["time"].dt.tz is None:
                errors.append("Candle timestamps must be timezone-aware.")

            elif str(df["time"].dt.tz) != "UTC":
                errors.append("Candle timestamps must use UTC.")

        if df["time"].isna().any():
            errors.append("Missing candle timestamps detected.")

        if df["time"].duplicated().any():
            errors.append("Duplicate candle timestamps detected.")

        if not df["time"].is_monotonic_increasing:
            errors.append("Candle timestamps are not chronological.")

        # --------------------------------------------------
        # NUMERIC DATA VALIDATION
        # --------------------------------------------------

        price_columns = ["open", "high", "low", "close"]

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]

        for column in numeric_columns:

            if not pd.api.types.is_numeric_dtype(df[column]):
                errors.append(
                    f"{column} must contain numeric values."
                )
                continue

            values = df[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                errors.append(
                    f"{column} contains missing or non-finite values."
                )

        if not errors:

            for column in price_columns:

                if (df[column] <= 0).any():
                    errors.append(
                        f"{column} contains non-positive prices."
                    )

            invalid_ohlc = (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )

            if invalid_ohlc.any():
                errors.append("Invalid OHLC relationships detected.")

            for column in ["tick_volume", "spread", "real_volume"]:

                if (df[column] < 0).any():
                    errors.append(
                        f"{column} contains negative values."
                    )

        # --------------------------------------------------
        # FORMING CANDLE VALIDATION
        # --------------------------------------------------

        if "is_forming_candle" in df.columns:

            forming_error = self._validate_forming_candle_flags(
                df["is_forming_candle"]
            )

            if forming_error is not None:
                errors.append(forming_error)

        # --------------------------------------------------
        # FUTURE TIMESTAMP VALIDATION
        # --------------------------------------------------

        if check_future and pd.api.types.is_datetime64_any_dtype(
            df["time"]
        ):

            if df["time"].dt.tz is not None:

                now = pd.Timestamp(
                    datetime.now(timezone.utc)
                )

                if (df["time"] > now).any():
                    errors.append(
                        "Future candle timestamps detected."
                    )

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        return self._result(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            candle_count=len(df),
            timeframe=timeframe
        )

    @staticmethod
    def _result(
        passed,
        errors,
        warnings,
        candle_count=0,
        timeframe=None
    ):

        return {
            "status": "PASS" if passed else "FAIL",
            "valid": passed,
            "candle_count": candle_count,
            "timeframe": timeframe,
            "errors": errors,
            "warnings": warnings,
        }

