
import math
from numbers import Integral

import pandas as pd
from pandas.api.types import is_bool_dtype

from research.swing_detection.swing_models import SwingPoint


class FractalSwingDetector:
    """
    Detects confirmed swing highs and lows using a fractal method.

    A swing high must have a strictly lower high on every candle
    within the configured left and right windows.

    A swing low must have a strictly higher low on every candle
    within the configured left and right windows.

    Only completed candles are accepted.
    """

    REQUIRED_COLUMNS = {
        "time",
        "open",
        "high",
        "low",
        "close",
    }

    PRICE_COLUMNS = ("open", "high", "low", "close")

    def __init__(self, left_bars: int = 2, right_bars: int = 2):
        self._validate_window(left_bars, "left_bars")
        self._validate_window(right_bars, "right_bars")

        self.left_bars = left_bars
        self.right_bars = right_bars

    @staticmethod
    def _validate_window(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")

        if value < 1:
            raise ValueError(f"{name} must be at least 1.")

    @staticmethod
    def _validate_forming_candle_flags(flags: pd.Series) -> None:
        """
        Require genuine Boolean forming-candle flags.

        BooleanDtype is accepted, provided there are no missing values.
        Strings and numeric values are rejected rather than coerced.
        """

        if flags.isna().any():
            raise ValueError(
                "is_forming_candle contains missing values."
            )

        if not is_bool_dtype(flags.dtype):
            raise TypeError(
                "is_forming_candle must contain Boolean values only."
            )

        if flags.any():
            raise ValueError(
                "Forming candles are not allowed. "
                "Supply completed candles only."
            )

    def _validate_candles(self, candles: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(candles, pd.DataFrame):
            raise TypeError("candles must be a pandas DataFrame.")

        missing = self.REQUIRED_COLUMNS.difference(candles.columns)

        if missing:
            raise ValueError(
                f"Missing required candle columns: {sorted(missing)}"
            )

        if candles.empty:
            return candles.copy()

        data = candles.copy()

        # Reject malformed or forming-candle flags.
        if "is_forming_candle" in data.columns:
            self._validate_forming_candle_flags(
                data["is_forming_candle"]
            )

        # Require timezone-aware UTC timestamps.
        parsed_time = pd.to_datetime(data["time"], errors="coerce")

        if parsed_time.isna().any():
            raise ValueError("Candle timestamps contain invalid values.")

        if not isinstance(parsed_time.dtype, pd.DatetimeTZDtype):
            raise ValueError(
                "Candle timestamps must be timezone-aware UTC values."
            )

        if str(parsed_time.dt.tz) != "UTC":
            raise ValueError("Candle timestamps must use UTC timezone.")

        if parsed_time.duplicated().any():
            raise ValueError("Duplicate candle timestamps detected.")

        if not parsed_time.is_monotonic_increasing:
            raise ValueError(
                "Candle timestamps must be in ascending chronological order."
            )

        data["time"] = parsed_time

        # Validate numeric OHLC values.
        for column in self.PRICE_COLUMNS:
            converted = pd.to_numeric(data[column], errors="coerce")

            if converted.isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing or non-numeric values."
                )

            if not all(math.isfinite(float(value)) for value in converted):
                raise ValueError(
                    f"Column '{column}' contains non-finite values."
                )

            if (converted <= 0).any():
                raise ValueError(
                    f"Column '{column}' must contain positive prices."
                )

            data[column] = converted.astype(float)

        # Validate candle structure.
        invalid_high = (
            (data["high"] < data["open"])
            | (data["high"] < data["close"])
            | (data["high"] < data["low"])
        )

        invalid_low = (
            (data["low"] > data["open"])
            | (data["low"] > data["close"])
        )

        if invalid_high.any() or invalid_low.any():
            raise ValueError("Invalid OHLC candle structure detected.")

        return data

    def detect(self, candles: pd.DataFrame) -> list[SwingPoint]:
        """
        Return confirmed swing highs and lows.

        The swing index is the positional index within the supplied
        DataFrame, not its original pandas index label.
        """

        data = self._validate_candles(candles)

        if data.empty:
            return []

        total_bars = len(data)
        required_bars = self.left_bars + self.right_bars + 1

        if total_bars < required_bars:
            return []

        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        times = data["time"].tolist()

        swings = []

        first_candidate = self.left_bars
        last_candidate = total_bars - self.right_bars

        for index in range(first_candidate, last_candidate):

            left_start = index - self.left_bars
            right_end = index + self.right_bars + 1

            left_highs = highs[left_start:index]
            right_highs = highs[index + 1:right_end]

            left_lows = lows[left_start:index]
            right_lows = lows[index + 1:right_end]

            is_swing_high = (
                highs[index] > left_highs.max()
                and highs[index] > right_highs.max()
            )

            is_swing_low = (
                lows[index] < left_lows.min()
                and lows[index] < right_lows.min()
            )

            confirmation_index = index + self.right_bars

            if is_swing_high:
                swings.append(
                    SwingPoint(
                        swing_type="HIGH",
                        price=float(highs[index]),
                        swing_time=times[index].to_pydatetime(),
                        swing_index=index,
                        confirmation_time=(
                            times[confirmation_index].to_pydatetime()
                        ),
                        confirmation_index=confirmation_index,
                        left_bars=self.left_bars,
                        right_bars=self.right_bars,
                    )
                )

            if is_swing_low:
                swings.append(
                    SwingPoint(
                        swing_type="LOW",
                        price=float(lows[index]),
                        swing_time=times[index].to_pydatetime(),
                        swing_index=index,
                        confirmation_time=(
                            times[confirmation_index].to_pydatetime()
                        ),
                        confirmation_index=confirmation_index,
                        left_bars=self.left_bars,
                        right_bars=self.right_bars,
                    )
                )

        # Deterministic ordering, including candles that qualify as both.
        swings.sort(
            key=lambda swing: (
                swing.confirmation_index,
                swing.swing_index,
                swing.swing_type,
            )
        )

        return swings
