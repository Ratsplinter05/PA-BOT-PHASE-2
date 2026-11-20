
from decimal import Decimal

import numpy as np
import pandas as pd
import unittest

from research.swing_detection.atr_calculator import WilderATR


def make_reference_candles(count=40):
    """
    Generate deterministic OHLC candles with price gaps.

    The dataset is independent of the ATR calculator and is
    used exclusively for numerical reference validation.
    """

    times = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=count,
        freq="15min",
        tz="UTC",
    )

    opens = []
    highs = []
    lows = []
    closes = []

    previous_close = 100.0

    for index in range(count):

        # Introduce deterministic upward and downward gaps.
        if index % 7 == 0:
            open_price = previous_close + 1.5

        elif index % 9 == 0:
            open_price = previous_close - 1.2

        else:
            open_price = previous_close + 0.15

        movement = (
            0.8 if index % 2 == 0 else -0.55
        )

        close_price = open_price + movement

        candle_range = 0.6 + (index % 4) * 0.2

        high_price = max(
            open_price,
            close_price,
        ) + candle_range

        low_price = min(
            open_price,
            close_price,
        ) - candle_range

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)

        previous_close = close_price

    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "is_forming_candle": [False] * count,
    })


def calculate_decimal_reference(candles, period):
    """
    Independently calculate Wilder ATR using Decimal arithmetic.

    This reference implementation does not call WilderATR
    or use its internal methods.
    """

    true_ranges = []

    previous_close = None

    for row in candles.itertuples(index=False):

        high = Decimal(str(row.high))
        low = Decimal(str(row.low))
        close = Decimal(str(row.close))

        if previous_close is None:

            true_range = high - low

        else:

            range_1 = high - low
            range_2 = abs(high - previous_close)
            range_3 = abs(low - previous_close)

            true_range = max(
                range_1,
                range_2,
                range_3,
            )

        true_ranges.append(true_range)

        previous_close = close

    results = [None] * len(true_ranges)

    if len(true_ranges) < period:
        return results

    initial_atr = (
        sum(true_ranges[:period], Decimal("0"))
        / Decimal(period)
    )

    seed_index = period - 1

    results[seed_index] = initial_atr

    previous_atr = initial_atr

    for index in range(period, len(true_ranges)):

        current_atr = (
            previous_atr * Decimal(period - 1)
            + true_ranges[index]
        ) / Decimal(period)

        results[index] = current_atr

        previous_atr = current_atr

    return results


class TestATRReferenceValidation(unittest.TestCase):

    def test_atr_matches_independent_decimal_reference(self):

        candles = make_reference_candles()

        periods = [1, 3, 7, 14]

        for period in periods:

            with self.subTest(period=period):

                actual = WilderATR(
                    period=period
                ).calculate(candles)

                expected = calculate_decimal_reference(
                    candles,
                    period,
                )

                for index, reference_value in enumerate(expected):

                    if reference_value is None:

                        self.assertTrue(
                            np.isnan(actual.iloc[index])
                        )

                    else:

                        self.assertAlmostEqual(
                            actual.iloc[index],
                            float(reference_value),
                            places=10,
                            msg=(
                                f"ATR mismatch at index {index}, "
                                f"period {period}"
                            ),
                        )

    def test_historical_atr_matches_prefix_calculation(self):

        candles = make_reference_candles()

        period = 14
        cutoff = 25

        full_result = WilderATR(
            period=period
        ).calculate(candles)

        prefix_result = WilderATR(
            period=period
        ).calculate(candles.iloc[:cutoff].copy())

        np.testing.assert_allclose(
            full_result.iloc[:cutoff].to_numpy(),
            prefix_result.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    def test_future_candle_changes_do_not_rewrite_past_atr(self):

        candles = make_reference_candles()

        period = 14
        cutoff = 25

        original_result = WilderATR(
            period=period
        ).calculate(candles)

        modified_candles = candles.copy(deep=True)

        # Change only candles occurring after the cutoff.
        future_indices = modified_candles.index[cutoff:]

        modified_candles.loc[
            future_indices,
            "high",
        ] *= 1.15

        modified_candles.loc[
            future_indices,
            "low",
        ] *= 0.95

        modified_candles.loc[
            future_indices,
            "close",
        ] *= 1.05

        modified_result = WilderATR(
            period=period
        ).calculate(modified_candles)

        np.testing.assert_allclose(
            original_result.iloc[:cutoff].to_numpy(),
            modified_result.iloc[:cutoff].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    def test_reference_calculation_is_deterministic(self):

        candles = make_reference_candles()

        first_reference = calculate_decimal_reference(
            candles,
            period=14,
        )

        second_reference = calculate_decimal_reference(
            candles.copy(deep=True),
            period=14,
        )

        self.assertEqual(
            first_reference,
            second_reference,
        )

    def test_reference_seed_matches_initial_atr(self):

        candles = make_reference_candles()

        period = 14

        reference = calculate_decimal_reference(
            candles,
            period,
        )

        actual = WilderATR(
            period=period
        ).calculate(candles)

        seed_index = period - 1

        self.assertAlmostEqual(
            actual.iloc[seed_index],
            float(reference[seed_index]),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
