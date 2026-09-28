import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coinlist import top_coins, reference_ids, price_for


def fake_universe(limit):
    """Mirrors fetch_top_cryptos: (mapping, cg_ids, is_live, note)."""
    mapping = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD",
               "NEAR Protocol (NEAR)": "NEAR-USD"}
    cg_ids = {"Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum",
              "NEAR Protocol (NEAR)": "near"}
    return dict(list(mapping.items())[:limit]), cg_ids, True, "from the API"


class TestTopCoins(unittest.TestCase):
    def test_returns_records_with_everything_needed(self):
        coins, note, live = top_coins(3, fetch=fake_universe)
        self.assertEqual(len(coins), 3)
        for key in ("label", "ticker", "id", "symbol"):
            self.assertIn(key, coins[0])

    def test_symbols_are_bare(self):
        coins, _n, _l = top_coins(3, fetch=fake_universe)
        self.assertEqual([c["symbol"] for c in coins], ["BTC", "ETH", "NEAR"])

    def test_tickers_are_preserved(self):
        coins, _n, _l = top_coins(1, fetch=fake_universe)
        self.assertEqual(coins[0]["ticker"], "BTC-USD")

    def test_limit_respected(self):
        self.assertEqual(len(top_coins(2, fetch=fake_universe)[0]), 2)

    def test_note_and_liveness_passed_through(self):
        _c, note, live = top_coins(1, fetch=fake_universe)
        self.assertEqual(note, "from the API")
        self.assertTrue(live)

    def test_missing_coingecko_id_is_none_not_a_crash(self):
        def no_ids(limit):
            return {"Mystery (XYZ)": "XYZ-USD"}, {}, False, "fallback"
        coins, _n, _l = top_coins(1, fetch=no_ids)
        self.assertIsNone(coins[0]["id"])

    def test_four_value_unpacking(self):
        """The bug this module exists to prevent: three-value unpacking."""
        result = fake_universe(3)
        self.assertEqual(len(result), 4)


class TestReferenceHelpers(unittest.TestCase):
    def test_ids_deduplicated_and_sorted(self):
        coins = [{"id": "near"}, {"id": "bitcoin"}, {"id": "near"}, {"id": None}]
        self.assertEqual(reference_ids(coins), ("bitcoin", "near"))

    def test_price_lookup(self):
        self.assertAlmostEqual(price_for({"id": "near"}, {"near": 5.49}), 5.49)

    def test_price_without_an_id(self):
        self.assertIsNone(price_for({"id": None}, {"near": 5.49}))

    def test_price_missing_from_the_response(self):
        self.assertIsNone(price_for({"id": "near"}, {}))
