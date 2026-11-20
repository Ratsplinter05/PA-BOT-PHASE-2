
import math

import pandas as pd
from pandas.api.types import is_bool_dtype

class WilderATR:
    """
    Calculate Average True Range using Wilder's smoothing method.

    The first ATR value is seeded using the arithmetic mean of
    the first `period` True Range values.

    Subsequent values use:
        ATR = ((previous ATR * (period - 1)) + current TR) / period

    The output preserves the original DataFrame index.
    The first period - 1 values are NaN.
    """

    REQUIRED_COLUMNS = {
        "time",
        "high",
        "low",
        "close",
    }

    def __init__(self, period: int = 14):
        if isinstance(period, bool) or not isinstance(period, int):
            raise TypeError("period must be an integer.")

        if period < 1:
            raise ValueError("period must be at least 1.")

        self.period = period

    def _validate_candles(
        self,
        candles: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(candles, pd.DataFrame):
            raise TypeError("candles must be a pandas DataFrame.")

        missing = self.REQUIRED_COLUMNS.difference(candles.columns)

        if missing:
            raise ValueError(
                f"Missing required candle columns: {sorted(missing)}"
            )

        if candles.empty:
            raise ValueError("candles cannot be empty.")

        data = candles.copy()

        # Reject forming candles.
        if "is_forming_candle" in data.columns:

            forming = data["is_forming_candle"]

            if forming.isna().any():
                raise ValueError(
                    "is_forming_candle contains missing values."
                )

            if not is_bool_dtype(forming.dtype):
                raise TypeError(
                    "is_forming_candle must contain boolean values."
                )

            if forming.any():
                raise ValueError(
                    "Forming candles are not permitted."
                )

        # Validate UTC timestamps.
        parsed_time = pd.to_datetime(
            data["time"],
            errors="coerce",
        )

        if parsed_time.isna().any():
            raise ValueError(
                "Candle timestamps contain invalid values."
            )

        if not isinstance(parsed_time.dtype, pd.DatetimeTZDtype):
            raise ValueError(
                "Candle timestamps must be timezone-aware."
            )

        if str(parsed_time.dt.tz) != "UTC":
            raise ValueError(
                "Candle timestamps must use UTC timezone."
            )

        if parsed_time.duplicated().any():
            raise ValueError(
                "Duplicate candle timestamps detected."
            )

        if not parsed_time.is_monotonic_increasing:
            raise ValueError(
                "Candle timestamps must be chronological."
            )

        data["time"] = parsed_time

        # Validate prices.
        for column in ("high", "low", "close"):

            converted = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            if converted.isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing or "
                    "non-numeric values."
                )

            if not all(
                math.isfinite(float(value))
                for value in converted
            ):
                raise ValueError(
                    f"Column '{column}' contains non-finite values."
                )

            if (converted <= 0).any():
                raise ValueError(
                    f"Column '{column}' must contain positive prices."
                )

            data[column] = converted.astype(float)

        # Validate OHLC consistency.
        if (
            (data["high"] < data["low"]).any()
            or (data["high"] < data["close"]).any()
            or (data["low"] > data["close"]).any()
        ):
            raise ValueError(
                "Invalid OHLC candle structure detected."
            )

        return data

    def calculate(
        self,
        candles: pd.DataFrame,
    ) -> pd.Series:
        """
        Return a Wilder ATR series aligned with the input candles.
        """

        data = self._validate_candles(candles)

        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)

        total = len(data)

        true_ranges = []

        for index in range(total):

            high = highs[index]
            low = lows[index]

            if index == 0:
                true_range = high - low

            else:
                previous_close = closes[index - 1]

                true_range = max(
                    high - low,
                    abs(high - previous_close),
                    abs(low - previous_close),
                )

            true_ranges.append(true_range)

        atr_values = [float("nan")] * total

        # Insufficient history: no ATR value can be produced.
        if total < self.period:
            return pd.Series(
                atr_values,
                index=candles.index,
                name=f"atr_{self.period}",
                dtype=float,
            )

        # Initial ATR uses the arithmetic mean of the first
        # `period` True Range values.
        initial_atr = (
            sum(true_ranges[:self.period]) / self.period
        )

        seed_index = self.period - 1
        atr_values[seed_index] = initial_atr

        # Apply Wilder's recursive smoothing.
        previous_atr = initial_atr

        for index in range(self.period, total):

            current_atr = (
                (
                    previous_atr * (self.period - 1)
                    + true_ranges[index]
                )
                / self.period
            )

            atr_values[index] = current_atr
            previous_atr = current_atr

        return pd.Series(
            atr_values,
            index=candles.index,
            name=f"atr_{self.period}",
            dtype=float,
        )
