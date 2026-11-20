import unittest

from src.market_data import MarketData


class MarketDataSymbolResolutionTests(unittest.TestCase):

    def setUp(self):
        self.market_data = MarketData()

    def test_exact_symbol_match(self):
        available = ["XAUUSD.0", "GBPUSD.0"]

        result = self.market_data.resolve_symbol(
            "XAUUSD.0",
            available
        )

        self.assertEqual(result, "XAUUSD.0")

    def test_case_insensitive_symbol_match(self):
        available = ["XAUUSD.0", "GBPUSD.0"]

        result = self.market_data.resolve_symbol(
            "xauusd.0",
            available
        )

        self.assertEqual(result, "XAUUSD.0")

    def test_missing_symbol_returns_none(self):
        available = ["XAUUSD.0", "GBPUSD.0"]

        result = self.market_data.resolve_symbol(
            "BTCUSD.0",
            available
        )

        self.assertIsNone(result)

    def test_empty_symbol_list_returns_none(self):
        result = self.market_data.resolve_symbol(
            "XAUUSD.0",
            []
        )

        self.assertIsNone(result)

    def test_exact_match_takes_precedence(self):
        available = ["xauusd.0", "XAUUSD.0"]

        result = self.market_data.resolve_symbol(
            "XAUUSD.0",
            available
        )

        self.assertEqual(result, "XAUUSD.0")


if __name__ == "__main__":
    unittest.main()
