import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hyperliquid_data as hl

hl.configure(spacing=0.0, max_retries=0, backoff=0.0)


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._p


META = [
    {"universe": [{"name": "BTC", "maxLeverage": 40}, {"name": "HYPE", "maxLeverage": 10},
                  {"name": "kPEPE", "maxLeverage": 10},
                  {"name": "DEAD", "isDelisted": True}, {"name": "WIF"}]},
    [{"markPx": "80000", "prevDayPx": "79000", "dayNtlVlm": "2000000000",
      "openInterest": "10000", "funding": "0.0000125"},
     {"markPx": "40", "prevDayPx": "38", "dayNtlVlm": "300000000",
      "openInterest": "5000000", "funding": "0.00002"},
     {"markPx": "0.012", "prevDayPx": "0.011", "dayNtlVlm": "90000000",
      "openInterest": "900000000", "funding": "-0.00001"},
     {"markPx": "1", "prevDayPx": "1", "dayNtlVlm": "999999999", "openInterest": "1"},
     {"markPx": "2.5", "prevDayPx": "2.4", "dayNtlVlm": "40000000",
      "openInterest": "1000000", "funding": "0.00001"}],
]


class Base(unittest.TestCase):
    def setUp(self):
        hl._spent.clear()          # a leftover full budget would stall later tests
        hl.configure(spacing=0.0, max_retries=0, backoff=0.0)
        self.orig = hl.requests.post

    def tearDown(self):
        hl.requests.post = self.orig


class TestMarketContexts(Base):
    def test_parses_volume_price_oi_funding(self):
        hl.requests.post = lambda *a, **k: FakeResp(META)
        ctxs, err = hl.fetch_market_contexts()
        self.assertIsNone(err)
        btc = next(c for c in ctxs if c.name == "BTC")
        self.assertEqual(btc.day_volume_usd, 2e9)
        self.assertEqual(btc.mark_price, 80000.0)
        self.assertEqual(btc.open_interest_usd, 10000 * 80000)
        self.assertAlmostEqual(btc.change_24h_pct, 1000 / 79000 * 100)

    def test_skips_delisted(self):
        hl.requests.post = lambda *a, **k: FakeResp(META)
        names = {c.name for c in hl.fetch_market_contexts()[0]}
        self.assertNotIn("DEAD", names)

    def test_network_failure_reported(self):
        hl.requests.post = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
        ctxs, err = hl.fetch_market_contexts()
        self.assertEqual(ctxs, [])
        self.assertIn("ConnectionError", err)

    def test_malformed_response_reported(self):
        hl.requests.post = lambda *a, **k: FakeResp({"oops": 1})
        self.assertEqual(hl.fetch_market_contexts()[0], [])

    def test_rate_limit_reported(self):
        hl.requests.post = lambda *a, **k: FakeResp({}, status=429)
        ctxs, err = hl.fetch_market_contexts()
        self.assertIn("429", err)


class TestKCoins(unittest.TestCase):
    def test_k_prefix_is_a_bundle_of_1000(self):
        self.assertEqual(hl.base_symbol("kPEPE"), "PEPE")
        self.assertEqual(hl.bundle_size("kPEPE"), 1000)

    def test_normal_coin_unchanged(self):
        self.assertEqual(hl.base_symbol("BTC"), "BTC")
        self.assertEqual(hl.bundle_size("BTC"), 1)

    def test_label_warns_about_bundles(self):
        c = hl.MarketContext("kPEPE", 1.0, 0.01, 1.0, 0.0, None, None)
        self.assertIn("per 1,000", c.label)

    def test_tickers_are_prefixed_so_prices_never_mix(self):
        c = hl.MarketContext("kPEPE", 1.0, 0.01, 1.0, 0.0, None, None)
        self.assertEqual(c.ticker, "HL:kPEPE")
        self.assertTrue(hl.is_hl_ticker(c.ticker))
        self.assertEqual(hl.hl_name(c.ticker), "kPEPE")
        self.assertFalse(hl.is_hl_ticker("PEPE-USD"))


class TestRanking(Base):
    def _ctxs(self):
        hl.requests.post = lambda *a, **k: FakeResp(META)
        return hl.fetch_market_contexts()[0]

    def test_excludes_top_market_cap_coins(self):
        ranked = hl.rank_by_volume(self._ctxs(), exclude_bases={"BTC"}, limit=10)
        self.assertNotIn("BTC", [c.name for c in ranked])

    def test_excludes_k_coins_by_underlying_token(self):
        ranked = hl.rank_by_volume(self._ctxs(), exclude_bases={"PEPE"}, limit=10)
        self.assertNotIn("kPEPE", [c.name for c in ranked])

    def test_sorted_by_volume(self):
        ranked = hl.rank_by_volume(self._ctxs(), exclude_bases=set(), limit=10)
        vols = [c.day_volume_usd for c in ranked]
        self.assertEqual(vols, sorted(vols, reverse=True))

    def test_minimum_volume(self):
        ranked = hl.rank_by_volume(self._ctxs(), exclude_bases=set(),
                                   min_volume_usd=50_000_000)
        self.assertTrue(all(c.day_volume_usd >= 50_000_000 for c in ranked))
        self.assertNotIn("WIF", [c.name for c in ranked])

    def test_limit(self):
        self.assertEqual(len(hl.rank_by_volume(self._ctxs(), set(), limit=2)), 2)


class TestCandles(Base):
    def test_parses_candles(self):
        rows = [{"t": 1758000000000 + i * 14400000, "T": 0, "s": "BTC", "i": "4h",
                 "o": "100", "h": "110", "l": "95", "c": "105", "v": "12", "n": 3}
                for i in range(5)]
        seen = {}

        def post(url, json=None, **k):
            seen.update(json)
            return FakeResp(rows)

        hl.requests.post = post
        df, err = hl.fetch_candles("HL:BTC", "4h", 5)
        self.assertIsNone(err)
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(df["High"].iloc[0], 110.0)
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertEqual(seen["req"]["coin"], "BTC")     # prefix stripped for the API

    def test_empty_candles_reported(self):
        hl.requests.post = lambda *a, **k: FakeResp([])
        df, err = hl.fetch_candles("BTC", "4h", 5)
        self.assertIsNone(df)
        self.assertIsNotNone(err)

    def test_unsupported_interval(self):
        df, err = hl.fetch_candles("BTC", "7m", 5)
        self.assertIsNone(df)


if __name__ == "__main__":
    unittest.main()


# The end-to-end scan tests live in the trading app's repo: they exercise
# trend_retrace, storage and tracking, which the Strategy Lab doesn't include.
class TestWeightBudget(unittest.TestCase):
    """The limit is weight per minute, not a gap between calls."""

    def setUp(self):
        self.orig = hl.requests.post
        hl._spent.clear()
        hl.configure(spacing=0.0, max_retries=0, backoff=0.0)

    def tearDown(self):
        hl.requests.post = self.orig
        hl._spent.clear()
        hl.configure(spacing=0.0, max_retries=0, backoff=0.0)

    def test_candle_weight_grows_with_the_number_of_candles(self):
        self.assertEqual(hl.candle_weight(60), 21)
        self.assertEqual(hl.candle_weight(1000), 37)
        self.assertGreater(hl.candle_weight(5000), hl.candle_weight(1000))

    def test_twenty_coins_fit_inside_one_minute(self):
        allowance = hl.WEIGHT_LIMIT_PER_MIN * hl.SAFETY_MARGIN
        self.assertLess(20 * hl.candle_weight(1000), allowance)

    def test_budget_records_what_was_spent(self):
        rows = [{"t": 1758000000000 + i * 300000, "o": "1", "h": "2", "l": "0.5",
                 "c": "1.5", "v": "1"} for i in range(100)]
        hl.requests.post = lambda *a, **k: FakeResp(rows)
        hl.fetch_candles("BTC", "5m", 100)
        self.assertEqual(hl.budget_used(), hl.candle_weight(100))

    def test_requests_wait_when_the_budget_is_full(self):
        import time
        hl.configure(spacing=0.0, max_retries=0, backoff=0.0, weight_limit=50)
        hl._spent.append((time.time(), 40))      # budget nearly gone
        slept = {"for": 0.0}
        real_sleep = hl.time.sleep
        hl.time.sleep = lambda s: slept.__setitem__("for", slept["for"] + s)
        try:
            hl._spend(hl.candle_weight(60))
        finally:
            hl.time.sleep = real_sleep
        self.assertGreater(slept["for"], 0, "should have waited for the window to clear")

    def test_waiting_is_bounded_so_a_scan_cannot_hang(self):
        import time
        hl.configure(spacing=0.0, max_retries=0, backoff=0.0, weight_limit=1)
        hl._spent.append((time.time(), 999))     # budget hopelessly over
        slept = {"for": 0.0}
        real_sleep = hl.time.sleep
        hl.time.sleep = lambda s: slept.__setitem__("for", slept["for"] + s)
        try:
            started = time.time()
            hl._spend(50, max_wait=0.01)
            self.assertLess(time.time() - started, 5.0)
        finally:
            hl.time.sleep = real_sleep


class TestParallelFetching(Base):
    def test_fetches_every_market(self):
        rows = [{"t": 1758000000000 + i * 300000, "o": "1", "h": "2", "l": "0.5",
                 "c": "1.5", "v": "1"} for i in range(80)]
        hl.requests.post = lambda *a, **k: FakeResp(rows)
        out = hl.fetch_candles_many(["BTC", "ETH", "SOL"], "5m", 80)
        self.assertEqual(set(out), {"BTC", "ETH", "SOL"})
        self.assertTrue(all(v is not None for v in out.values()))

    def test_one_bad_symbol_does_not_sink_the_scan(self):
        rows = [{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "1"}]

        def post(url, json=None, **k):
            if json["req"]["coin"] == "BAD":
                return FakeResp([])
            return FakeResp(rows)

        hl.requests.post = post
        out = hl.fetch_candles_many(["BTC", "BAD", "ETH"], "5m", 60)
        self.assertIsNone(out["BAD"])
        self.assertIsNotNone(out["BTC"])

    def test_progress_is_reported_for_each(self):
        rows = [{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "1"}]
        hl.requests.post = lambda *a, **k: FakeResp(rows)
        seen = []
        hl.fetch_candles_many(["BTC", "ETH"], "5m", 60,
                              progress=lambda i, n, name: seen.append((i, n)))
        self.assertEqual(sorted(seen), [(1, 2), (2, 2)])

    def test_empty_list(self):
        self.assertEqual(hl.fetch_candles_many([], "5m", 60), {})


class TestProgressCannotBreakAFetch(Base):
    """Progress callbacks run on worker threads, where some UI toolkits refuse
    to be touched. A reporting error must not lose the fetched data."""

    def test_a_raising_progress_callback_is_survivable(self):
        rows = [{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "1"}]
        hl.requests.post = lambda *a, **k: FakeResp(rows)

        def explode(i, n, name):
            raise RuntimeError("no session context")

        out = hl.fetch_candles_many(["BTC", "ETH"], "5m", 60, progress=explode)
        self.assertEqual(set(out), {"BTC", "ETH"})
        self.assertTrue(all(v is not None for v in out.values()))

    def test_progress_counts_are_sequential(self):
        rows = [{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "1"}]
        hl.requests.post = lambda *a, **k: FakeResp(rows)
        seen = []
        hl.fetch_candles_many(["A", "B", "C"], "5m", 60,
                              progress=lambda i, n, nm: seen.append(i))
        self.assertEqual(sorted(seen), [1, 2, 3])


class TestProgressRunsOnTheCallingThread(Base):
    """Streamlit widgets belong to the script's thread. Calling one from a
    worker raises NoSessionContext, which crashed the scan."""

    def test_callback_thread_is_the_caller(self):
        import threading
        rows = [{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "1"}]
        hl.requests.post = lambda *a, **k: FakeResp(rows)
        caller = threading.current_thread()
        seen = []
        hl.fetch_candles_many(["A", "B", "C", "D"], "5m", 60,
                              progress=lambda i, n, nm: seen.append(threading.current_thread()))
        self.assertTrue(seen)
        for t in seen:
            self.assertIs(t, caller, "progress must not run on a worker thread")

    def test_every_market_still_returned(self):
        rows = [{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "1"}]
        hl.requests.post = lambda *a, **k: FakeResp(rows)
        out = hl.fetch_candles_many(["A", "B", "C"], "5m", 60)
        self.assertEqual(set(out), {"A", "B", "C"})

    def test_progress_counts_up_to_the_total(self):
        rows = [{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "1"}]
        hl.requests.post = lambda *a, **k: FakeResp(rows)
        seen = []
        hl.fetch_candles_many(["A", "B", "C"], "5m", 60,
                              progress=lambda i, n, nm: seen.append((i, n)))
        self.assertEqual(seen[-1], (3, 3))

    def test_a_raising_worker_does_not_lose_the_others(self):
        def post(url, json=None, **k):
            if json["req"]["coin"] == "BOOM":
                raise RuntimeError("worker blew up")
            return FakeResp([{"t": 1758000000000, "o": "1", "h": "2", "l": "0.5",
                              "c": "1.5", "v": "1"}])
        hl.requests.post = post
        out = hl.fetch_candles_many(["A", "BOOM", "B"], "5m", 60)
        self.assertEqual(set(out), {"A", "BOOM", "B"})
        self.assertIsNone(out["BOOM"])
