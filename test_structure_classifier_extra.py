import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_classifier import StructureClassifier

BASE = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def make_swing(swing_type, price, index, right_bars=2):
    swing_time = BASE + timedelta(hours=index * 3)
    return SwingPoint(
        swing_type=swing_type,
        price=price,
        swing_time=swing_time,
        swing_index=index,
        confirmation_time=swing_time + timedelta(hours=right_bars),
        confirmation_index=index + right_bars,
        left_bars=2,
        right_bars=right_bars,
    )


class TestStructureClassifierExtra(unittest.TestCase):

    def test_order_uses_confirmation_not_swing_time(self):
        # A formed first but was confirmed much later than B.
        a = make_swing("HIGH", 1.1000, 1, right_bars=20)
        b = make_swing("HIGH", 1.1050, 2, right_bars=2)

        result = StructureClassifier().classify([a, b])

        self.assertIs(result[0].swing, b)
        self.assertIs(result[1].swing, a)
        self.assertEqual(result[1].classification, "LH")

    def test_previous_fields_are_populated(self):
        first = make_swing("HIGH", 1.1000, 1)
        second = make_swing("HIGH", 1.1050, 2)

        result = StructureClassifier().classify([first, second])

        self.assertIsNone(result[0].previous_price)
        self.assertIsNone(result[0].previous_swing_time)
        self.assertEqual(result[1].previous_price, 1.1000)
        self.assertEqual(result[1].previous_swing_time, first.swing_time)

    def test_equal_swings_become_the_next_reference(self):
        swings = [
            make_swing("HIGH", 100.0, 1),
            make_swing("HIGH", 100.9, 2),
            make_swing("HIGH", 101.8, 3),
        ]

        result = StructureClassifier(price_tolerance=1.0).classify(swings)

        self.assertEqual(
            [r.classification for r in result], [None, "EQH", "EQH"]
        )

    def test_empty_input(self):
        self.assertEqual(StructureClassifier().classify([]), [])

    def test_non_swing_item_rejected(self):
        with self.assertRaises(TypeError):
            StructureClassifier().classify(["not a swing"])

    def test_zero_and_negative_prices_rejected(self):
        for price in (0.0, -1.0):
            with self.assertRaises(ValueError):
                StructureClassifier().classify(
                    [make_swing("HIGH", price, 1)]
                )

    def test_invalid_swing_type_rejected(self):
        with self.assertRaises(ValueError):
            StructureClassifier().classify([make_swing("MIDDLE", 1.1, 1)])

    def test_duplicate_input_is_compared_to_itself(self):
        swing = make_swing("HIGH", 1.1000, 1)

        result = StructureClassifier().classify([swing, swing])

        self.assertEqual(result[1].classification, "EQH")


if __name__ == "__main__":
    unittest.main()