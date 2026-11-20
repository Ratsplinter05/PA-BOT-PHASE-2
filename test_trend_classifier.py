
import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.trend_classifier import TrendClassifier


class TestTrendClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = TrendClassifier()
        self.base_time = datetime(
            2026, 1, 1, 0, 0, tzinfo=timezone.utc
        )

    def make_swing(
        self,
        swing_type,
        classification,
        offset,
        price,
        index
    ):
        swing_time = self.base_time + timedelta(hours=offset)
        confirmation_time = swing_time + timedelta(hours=2)

        swing = SwingPoint(
            swing_type=swing_type,
            price=price,
            swing_time=swing_time,
            swing_index=index,
            confirmation_time=confirmation_time,
            confirmation_index=index + 2,
            left_bars=2,
            right_bars=2,
        )

        return ClassifiedSwing(
            swing=swing,
            classification=classification,
            previous_price=None,
            previous_swing_time=None,
        )

    def test_bullish_structure(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1000, 0),
            self.make_swing("LOW", None, 1, 1.0900, 1),
            self.make_swing("HIGH", "HH", 3, 1.1100, 3),
            self.make_swing("LOW", "HL", 4, 1.0950, 4),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "BULLISH")

    def test_bearish_structure(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1200, 0),
            self.make_swing("LOW", None, 1, 1.1100, 1),
            self.make_swing("HIGH", "LH", 3, 1.1150, 3),
            self.make_swing("LOW", "LL", 4, 1.1000, 4),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "BEARISH")

    def test_conflicting_structure_is_mixed(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1000, 0),
            self.make_swing("LOW", None, 1, 1.0900, 1),
            self.make_swing("HIGH", "HH", 3, 1.1100, 3),
            self.make_swing("LOW", "LL", 4, 1.0800, 4),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "MIXED")

    def test_equal_high_produces_mixed_structure(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1000, 0),
            self.make_swing("LOW", None, 1, 1.0900, 1),
            self.make_swing("HIGH", "EQH", 3, 1.1000, 3),
            self.make_swing("LOW", "HL", 4, 1.0950, 4),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "MIXED")

    def test_insufficient_structure_is_unconfirmed(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1000, 0),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "UNCONFIRMED")

    def test_simultaneous_confirmations_create_one_snapshot(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1000, 0),
            self.make_swing("LOW", None, 0, 1.0900, 1),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].trend, "UNCONFIRMED")

    def test_input_is_processed_chronologically(self):
        swings = [
            self.make_swing("LOW", "HL", 4, 1.0950, 4),
            self.make_swing("HIGH", "HH", 3, 1.1100, 3),
            self.make_swing("LOW", None, 1, 1.0900, 1),
            self.make_swing("HIGH", None, 0, 1.1000, 0),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(
            results[-1].trend,
            "BULLISH"
        )

        confirmation_times = [
            item.as_of_time for item in results
        ]

        self.assertEqual(
            confirmation_times,
            sorted(confirmation_times)
        )

    # Additional tests

    def test_equal_low_produces_mixed_structure(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1000, 0),
            self.make_swing("LOW", None, 1, 1.0900, 1),
            self.make_swing("HIGH", "HH", 3, 1.1100, 3),
            self.make_swing("LOW", "EQL", 4, 1.0900, 4),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "MIXED")

    def test_simultaneous_structure_updates_create_one_snapshot(self):
        swings = [
            self.make_swing("HIGH", None, 0, 1.1000, 0),
            self.make_swing("LOW", None, 1, 1.0900, 1),
            self.make_swing("HIGH", "HH", 3, 1.1100, 3),
            self.make_swing("LOW", "HL", 4, 1.0950, 4),
            self.make_swing("HIGH", "LH", 6, 1.1050, 6),
            self.make_swing("LOW", "LL", 6, 1.0850, 7),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "BEARISH")

        self.assertEqual(
            sum(
                result.as_of_time == results[-1].as_of_time
                for result in results
            ),
            1,
        )

    def test_high_cannot_have_low_structure_classification(self):
        invalid_swing = self.make_swing(
            "HIGH", "HL", 1, 1.1000, 1
        )

        with self.assertRaisesRegex(ValueError, "classification"):
            self.classifier.classify([invalid_swing])


if __name__ == "__main__":
    unittest.main()
