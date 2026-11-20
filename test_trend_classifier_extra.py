import unittest
from datetime import datetime, timedelta, timezone

from research.swing_detection.swing_models import SwingPoint
from research.swing_detection.structure_models import ClassifiedSwing
from research.swing_detection.trend_classifier import TrendClassifier

BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def make_point(swing_type, index):
    swing_time = BASE + timedelta(hours=index)
    return SwingPoint(
        swing_type=swing_type,
        price=1.1000,
        swing_time=swing_time,
        swing_index=index,
        confirmation_time=swing_time + timedelta(hours=2),
        confirmation_index=index + 2,
        left_bars=2,
        right_bars=2,
    )


def make_classified(swing_type, classification, index):
    return ClassifiedSwing(
        swing=make_point(swing_type, index),
        classification=classification,
        previous_price=None,
        previous_swing_time=None,
    )


def bullish_then_bearish():
    return [
        make_classified("HIGH", None, 0),
        make_classified("LOW", None, 1),
        make_classified("HIGH", "HH", 3),
        make_classified("LOW", "HL", 4),
        make_classified("HIGH", "LH", 6),
        make_classified("LOW", "LL", 7),
    ]


class TestTrendClassifierExtra(unittest.TestCase):

    def setUp(self):
        self.classifier = TrendClassifier()

    def test_empty_input(self):
        self.assertEqual(self.classifier.classify([]), [])

    def test_non_classified_swing_rejected(self):
        with self.assertRaises(TypeError):
            self.classifier.classify([make_point("HIGH", 1)])

    def test_invalid_classification_string_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid structure classification"
        ):
            self.classifier.classify([make_classified("HIGH", "XX", 1)])

    def test_invalid_swing_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid swing type"):
            self.classifier.classify([make_classified("MIDDLE", None, 1)])

    def test_low_cannot_have_high_classification(self):
        with self.assertRaisesRegex(ValueError, "classification"):
            self.classifier.classify([make_classified("LOW", "HH", 1)])

    def test_one_side_unclassified_is_unconfirmed(self):
        swings = [
            make_classified("HIGH", None, 0),
            make_classified("LOW", None, 1),
            make_classified("HIGH", "HH", 3),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[-1].trend, "UNCONFIRMED")

    def test_lower_high_with_higher_low_is_mixed(self):
        swings = [
            make_classified("HIGH", None, 0),
            make_classified("LOW", None, 1),
            make_classified("HIGH", "LH", 3),
            make_classified("LOW", "HL", 4),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "MIXED")
        self.assertIn("conflicting", results[-1].reason)

    def test_equal_high_with_lower_low_is_mixed(self):
        # Records current behaviour: any EQH/EQL gives MIXED.
        swings = [
            make_classified("HIGH", None, 0),
            make_classified("LOW", None, 1),
            make_classified("HIGH", "EQH", 3),
            make_classified("LOW", "LL", 4),
        ]

        results = self.classifier.classify(swings)

        self.assertEqual(results[-1].trend, "MIXED")

    def test_trend_progression(self):
        results = self.classifier.classify(bullish_then_bearish())

        self.assertEqual(
            [r.trend for r in results],
            [
                "UNCONFIRMED",
                "UNCONFIRMED",
                "UNCONFIRMED",
                "BULLISH",
                "MIXED",
                "BEARISH",
            ],
        )

    def test_snapshots_reference_the_correct_swings(self):
        swings = bullish_then_bearish()

        results = self.classifier.classify(swings)

        self.assertIs(results[3].latest_high, swings[2])
        self.assertIs(results[3].latest_low, swings[3])
        self.assertIs(results[-1].latest_high, swings[4])
        self.assertIs(results[-1].latest_low, swings[5])

    def test_input_list_is_not_modified(self):
        swings = [
            make_classified("LOW", "HL", 4),
            make_classified("HIGH", "HH", 3),
            make_classified("LOW", None, 1),
            make_classified("HIGH", None, 0),
        ]
        original = list(swings)

        self.classifier.classify(swings)

        for current, before in zip(swings, original):
            self.assertIs(current, before)

    def test_accepts_a_generator(self):
        swings = (item for item in bullish_then_bearish())

        results = self.classifier.classify(swings)

        self.assertEqual(len(results), 6)
        self.assertEqual(results[-1].trend, "BEARISH")


if __name__ == "__main__":
    unittest.main()