import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_classifier import StructureClassifier


class TestStructureClassifier(unittest.TestCase):

    def setUp(self):
        self.base_time = datetime(
            2026, 1, 1, 10, 0, tzinfo=timezone.utc
        )

    def make_swing(self, swing_type, price, index):
        swing_time = self.base_time + timedelta(hours=index * 3)

        return SwingPoint(
            swing_type=swing_type,
            price=price,
            swing_time=swing_time,
            swing_index=index,
            confirmation_time=swing_time + timedelta(hours=2),
            confirmation_index=index + 2,
            left_bars=2,
            right_bars=2,
        )

    def test_higher_high(self):
        swings = [
            self.make_swing("HIGH", 1.1000, 1),
            self.make_swing("HIGH", 1.1050, 2),
        ]

        result = StructureClassifier().classify(swings)

        self.assertIsNone(result[0].classification)
        self.assertEqual(result[1].classification, "HH")

    def test_lower_high(self):
        swings = [
            self.make_swing("HIGH", 1.1050, 1),
            self.make_swing("HIGH", 1.1000, 2),
        ]

        result = StructureClassifier().classify(swings)

        self.assertEqual(result[1].classification, "LH")

    def test_higher_low(self):
        swings = [
            self.make_swing("LOW", 1.0900, 1),
            self.make_swing("LOW", 1.0950, 2),
        ]

        result = StructureClassifier().classify(swings)

        self.assertEqual(result[1].classification, "HL")

    def test_lower_low(self):
        swings = [
            self.make_swing("LOW", 1.0950, 1),
            self.make_swing("LOW", 1.0900, 2),
        ]

        result = StructureClassifier().classify(swings)

        self.assertEqual(result[1].classification, "LL")

    def test_equal_high(self):
        swings = [
            self.make_swing("HIGH", 1.1000, 1),
            self.make_swing("HIGH", 1.1000, 2),
        ]

        result = StructureClassifier().classify(swings)

        self.assertEqual(result[1].classification, "EQH")

    def test_equal_low(self):
        swings = [
            self.make_swing("LOW", 1.0900, 1),
            self.make_swing("LOW", 1.0900, 2),
        ]

        result = StructureClassifier().classify(swings)

        self.assertEqual(result[1].classification, "EQL")

    def test_price_tolerance(self):
        swings = [
            self.make_swing("HIGH", 1.10000, 1),
            self.make_swing("HIGH", 1.10005, 2),
        ]

        result = StructureClassifier(
            price_tolerance=0.0001
        ).classify(swings)

        self.assertEqual(result[1].classification, "EQH")

    def test_different_swing_types_are_compared_separately(self):
        swings = [
            self.make_swing("HIGH", 1.1000, 1),
            self.make_swing("LOW", 1.0900, 2),
            self.make_swing("HIGH", 1.1050, 3),
            self.make_swing("LOW", 1.0850, 4),
        ]

        result = StructureClassifier().classify(swings)

        self.assertEqual(result[2].classification, "HH")
        self.assertEqual(result[3].classification, "LL")

    def test_swing_order_uses_confirmation_time(self):
        first = self.make_swing("HIGH", 1.1000, 1)
        second = self.make_swing("HIGH", 1.1050, 2)

        result = StructureClassifier().classify(
            [second, first]
        )

        self.assertEqual(result[0].swing, first)
        self.assertEqual(result[1].swing, second)
        self.assertEqual(result[1].classification, "HH")

    def test_original_swing_objects_are_preserved(self):
        swing = self.make_swing("HIGH", 1.1000, 1)

        result = StructureClassifier().classify([swing])

        self.assertIs(result[0].swing, swing)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            StructureClassifier(price_tolerance=-0.1)

    def test_non_numeric_tolerance_rejected(self):
        with self.assertRaises(TypeError):
            StructureClassifier(price_tolerance="0.0001")

    def test_non_finite_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            StructureClassifier(price_tolerance=float("inf"))

    def test_invalid_swing_price_rejected(self):
        swing = self.make_swing("HIGH", float("nan"), 1)

        with self.assertRaises(ValueError):
            StructureClassifier().classify([swing])

    def test_invalid_confirmation_time_rejected(self):
        swing = self.make_swing("HIGH", 1.1000, 1)

        invalid_swing = SwingPoint(
            swing_type="HIGH",
            price=swing.price,
            swing_time=swing.confirmation_time,
            swing_index=swing.swing_index,
            confirmation_time=swing.swing_time,
            confirmation_index=swing.confirmation_index,
            left_bars=2,
            right_bars=2,
        )

        with self.assertRaises(ValueError):
            StructureClassifier().classify([invalid_swing])


if __name__ == "__main__":
    unittest.main()
