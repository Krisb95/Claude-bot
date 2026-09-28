import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import exchanges
from exchanges import (to_binance_symbol, to_kraken_pair, base_asset,
                        fetch_binance_klines, fetch_kraken_ohlc,
                        fetch_binance_price, fetch_kraken_price,
                        fetch_spot, build_frames)

exchanges.configure(max_retries=0, backoff=0.0, spacing=0.0)


class FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def binance_rows(n=50):
    base = 1758000000000
    return [[base + i * 14400000, "100.0", "110.0", "95.0", "105.0", "12.5",
             base + (i + 1) * 14400000, "0", 10, "0", "0", "0"] for i in range(n)]


def kraken_rows(n=50):
    base = 1758000000
    return [[base + i * 14400, "100.0", "110.0", "95.0", "105.0", "104.0", "12.5", 10]
            for i in range(n)]


class TestSymbolMapping(unittest.TestCase):
    def setUp(self):
        exchanges.reset_blocked_hosts()

    def test_binance_uses_usdt_pairs(self):
        self.assertEqual(to_binance_symbol("BTC-USD"), "BTCUSDT")
        self.assertEqual(to_binance_symbol("SOL-USD"), "SOLUSDT")

    def test_kraken_uses_xbt_for_bitcoin(self):
        self.assertEqual(to_kraken_pair("BTC-USD"), "XBTUSD")

    def test_kraken_uses_xdg_for_doge(self):
        self.assertEqual(to_kraken_pair("DOGE-USD"), "XDGUSD")

    def test_other_assets_pass_through(self):
        self.assertEqual(to_kraken_pair("SOL-USD"), "SOLUSD")

    def test_base_asset_strips_suffix(self):
        self.assertEqual(base_asset("HYPE-USD"), "HYPE")


class TestBinance(unittest.TestCase):
    def setUp(self):
        exchanges.reset_blocked_hosts()
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def test_klines_parsed_to_ohlcv(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(binance_rows())
        df, err = fetch_binance_klines("BTC-USD", "4h")
        self.assertIsNone(err)
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(df["High"].iloc[0], 110.0)
        self.assertEqual(df["Volume"].iloc[0], 12.5)

    def test_true_daily_ohlc_unlike_coingecko(self):
        """The whole point of moving off CoinGecko: daily High != Low."""
        exchanges.requests.get = lambda *a, **k: FakeResp(binance_rows())
        df, err = fetch_binance_klines("BTC-USD", "1d")
        self.assertTrue((df["High"] > df["Low"]).all())

    def test_geo_block_451_is_reported_clearly(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({}, status_code=451)
        df, err = fetch_binance_klines("BTC-USD", "4h")
        self.assertIsNone(df)
        self.assertIn("geo-blocked", err)

    def test_unlisted_symbol_reports_empty(self):
        exchanges.requests.get = lambda *a, **k: FakeResp([])
        df, err = fetch_binance_klines("NOTREAL-USD", "4h")
        self.assertIsNone(df)
        self.assertIn("no candles", err)

    def test_unsupported_interval_rejected(self):
        df, err = fetch_binance_klines("BTC-USD", "7m")
        self.assertIsNone(df)
        self.assertIn("Unsupported", err)

    def test_price_endpoint(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({"price": "81544.12"})
        price, err = fetch_binance_price("BTC-USD")
        self.assertAlmostEqual(price, 81544.12)


class TestKraken(unittest.TestCase):
    def setUp(self):
        exchanges.reset_blocked_hosts()
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def test_ohlc_parsed(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(
            {"error": [], "result": {"XXBTZUSD": kraken_rows(), "last": 1}})
        df, err = fetch_kraken_ohlc("BTC-USD", "4h")
        self.assertIsNone(err)
        self.assertEqual(df["High"].iloc[0], 110.0)

    def test_json_level_error_is_surfaced(self):
        """Kraken returns HTTP 200 with errors in the body."""
        exchanges.requests.get = lambda *a, **k: FakeResp(
            {"error": ["EQuery:Unknown asset pair"], "result": {}})
        df, err = fetch_kraken_ohlc("NOTREAL-USD", "4h")
        self.assertIsNone(df)
        self.assertIn("Unknown asset pair", err)

    def test_price_endpoint(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(
            {"error": [], "result": {"XXBTZUSD": {"c": ["81544.12", "0.01"]}}})
        price, err = fetch_kraken_price("BTC-USD")
        self.assertAlmostEqual(price, 81544.12)


class TestFallbackBehaviour(unittest.TestCase):
    def setUp(self):
        exchanges.reset_blocked_hosts()
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def test_spot_falls_back_to_kraken_when_binance_blocked(self):
        def _get(url, params=None, **k):
            if "binance" in url:
                return FakeResp({}, status_code=451)
            return FakeResp({"error": [], "result": {"XXBTZUSD": {"c": ["100.0", "1"]}}})
        exchanges.requests.get = _get
        price, source, err = fetch_spot("BTC-USD")
        self.assertEqual(source, "Kraken")
        self.assertAlmostEqual(price, 100.0)

    def test_spot_prefers_binance_when_available(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({"price": "55.5"})
        price, source, err = fetch_spot("BTC-USD")
        self.assertEqual(source, "Binance")

    def test_build_frames_uses_binance_when_available(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(binance_rows())
        result = build_frames("BTC-USD")
        self.assertEqual(result.source, "Binance")
        for tf in ("1d", "4h", "1h", "5m"):
            self.assertFalse(result.frames[tf].empty)

    def test_build_frames_falls_back_to_kraken(self):
        def _get(url, params=None, **k):
            if "binance" in url:
                return FakeResp({}, status_code=451)
            return FakeResp({"error": [], "result": {"XXBTZUSD": kraken_rows(), "last": 1}})
        exchanges.requests.get = _get
        result = build_frames("BTC-USD")
        self.assertEqual(result.source, "Kraken")
        self.assertFalse(result.frames["4h"].empty)
        self.assertTrue(any("Binance unavailable" in p for p in result.problems))

    def test_build_frames_reports_total_failure(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({}, status_code=451)
        result = build_frames("NOTREAL-USD")
        self.assertIsNone(result.source)
        self.assertTrue(all(f.empty for f in result.frames.values()))

    def test_source_is_not_mixed_across_timeframes(self):
        """Once a venue is chosen on 4H, the rest come from the same venue —
        mixing two exchanges' prices in one analysis would be misleading."""
        seen = {"binance": 0, "kraken": 0}

        def _get(url, params=None, **k):
            if "binance" in url:
                seen["binance"] += 1
                return FakeResp(binance_rows())
            seen["kraken"] += 1
            return FakeResp({"error": [], "result": {"X": kraken_rows(), "last": 1}})

        exchanges.requests.get = _get
        build_frames("BTC-USD")
        self.assertEqual(seen["kraken"], 0)


if __name__ == "__main__":
    unittest.main()


class TestBinanceHistoryPaging(unittest.TestCase):
    """Binance caps a request at 1,000 candles; history beyond that is paged."""

    def setUp(self):
        exchanges.reset_blocked_hosts()
        self.original = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.original

    def _pager(self, available):
        """Serve `available` total candles, newest first, honouring endTime."""
        base = 1_700_000_000_000
        step = 3_600_000
        all_rows = [[base + i * step, "1", "2", "0.5", "1.5", "10",
                     base + (i + 1) * step, "0", 1, "0", "0", "0"]
                    for i in range(available)]
        calls = {"n": 0}

        def _get(url, params=None, **k):
            calls["n"] += 1
            end = params.get("endTime")
            pool = [r for r in all_rows if end is None or r[0] <= end]
            return FakeResp(pool[-params["limit"]:])
        return _get, calls

    def test_pages_beyond_one_thousand(self):
        get, calls = self._pager(2500)
        exchanges.requests.get = get
        df, err = exchanges.fetch_binance_history("BTC-USD", "1h", 2500)
        self.assertIsNone(err)
        self.assertEqual(len(df), 2500)
        self.assertEqual(calls["n"], 3)

    def test_result_is_sorted_and_deduplicated(self):
        get, _ = self._pager(2200)
        exchanges.requests.get = get
        df, _ = exchanges.fetch_binance_history("BTC-USD", "1h", 2200)
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertFalse(df.index.duplicated().any())

    def test_stops_at_start_of_listing(self):
        get, calls = self._pager(600)          # coin only has 600 candles
        exchanges.requests.get = get
        df, _ = exchanges.fetch_binance_history("NEW-USD", "1h", 4000)
        self.assertEqual(len(df), 600)
        self.assertEqual(calls["n"], 1)

    def test_partial_failure_keeps_pages_already_fetched(self):
        get, _ = self._pager(3000)
        state = {"n": 0}

        def flaky(url, params=None, **k):
            state["n"] += 1
            if state["n"] == 2:
                return FakeResp({}, status_code=451)
            return get(url, params=params)

        exchanges.requests.get = flaky
        df, err = exchanges.fetch_binance_history("BTC-USD", "1h", 3000)
        self.assertEqual(len(df), 1000)       # first page kept, not discarded

    def test_unsupported_interval(self):
        df, err = exchanges.fetch_binance_history("BTC-USD", "7m", 100)
        self.assertIsNone(df)


def bybit_ok(payload):
    return FakeResp({"retCode": 0, "retMsg": "OK", "result": payload})


class TestBybit(unittest.TestCase):
    def setUp(self):
        exchanges.reset_blocked_hosts()
        self.orig = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.orig

    def test_symbol_mapping(self):
        self.assertEqual(exchanges.to_bybit_symbol("BTC-USD"), "BTCUSDT")

    def test_tickers_include_change_in_percent_and_dollars(self):
        exchanges.requests.get = lambda *a, **k: bybit_ok({"list": [
            {"symbol": "BTCUSDT", "lastPrice": "82000", "prevPrice24h": "80000",
             "price24hPcnt": "0.025", "turnover24h": "1000000",
             "openInterestValue": "500000", "fundingRate": "0.0001",
             "highPrice24h": "83000", "lowPrice24h": "79000"}]})
        rows, err = exchanges.fetch_bybit_tickers()
        self.assertIsNone(err)
        self.assertAlmostEqual(rows["BTC"]["change_pct"], 2.5)
        self.assertAlmostEqual(rows["BTC"]["change_abs"], 2000.0)

    def test_non_usdt_pairs_ignored(self):
        exchanges.requests.get = lambda *a, **k: bybit_ok({"list": [
            {"symbol": "BTCUSDC", "lastPrice": "82000", "prevPrice24h": "80000"}]})
        rows, err = exchanges.fetch_bybit_tickers()
        self.assertEqual(rows, {})

    def test_klines_returned_oldest_first(self):
        base = 1758000000000
        rows = [[str(base + i * 14400000), "1", "2", "0.5", "1.5", "10", "15"]
                for i in range(5)][::-1]          # Bybit sends newest first
        exchanges.requests.get = lambda *a, **k: bybit_ok({"list": rows})
        df, err = exchanges.fetch_bybit_klines("BTC-USD", "4h")
        self.assertIsNone(err)
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertEqual(len(df), 5)

    def test_geo_block_explained_plainly(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({}, status_code=403)
        rows, err = exchanges.fetch_bybit_tickers()
        self.assertEqual(rows, {})
        self.assertIn("blocks requests from this server's location", err)

    def test_falls_back_to_the_alternate_domain(self):
        seen = []

        def _get(url, params=None, **k):
            seen.append(url)
            if "api.bybit.com" in url:
                return FakeResp({}, status_code=403)
            return bybit_ok({"list": [{"symbol": "BTCUSDT", "lastPrice": "82000",
                                       "prevPrice24h": "80000", "price24hPcnt": "0.025"}]})

        exchanges.requests.get = _get
        rows, err = exchanges.fetch_bybit_tickers()
        self.assertIsNone(err)
        self.assertIn("BTC", rows)
        self.assertTrue(any("bytick" in u for u in seen))

    def test_api_level_error_reported(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(
            {"retCode": 10001, "retMsg": "params error", "result": {}})
        rows, err = exchanges.fetch_bybit_tickers()
        self.assertIn("params error", err)

    def test_price_lookup(self):
        exchanges.requests.get = lambda *a, **k: bybit_ok(
            {"list": [{"symbol": "BTCUSDT", "lastPrice": "82000"}]})
        price, err = exchanges.fetch_bybit_price("BTC-USD")
        self.assertAlmostEqual(price, 82000.0)


class TestBlockedHostsAreRememberedNotRetried(unittest.TestCase):
    """A geo-block is permanent. Retrying it for every coin and timeframe is
    what turned a 20-coin scan into a 15-minute wait."""

    def setUp(self):
        exchanges.reset_blocked_hosts()
        self.orig = exchanges.requests.get
        exchanges.reset_blocked_hosts()

    def tearDown(self):
        exchanges.requests.get = self.orig
        exchanges.reset_blocked_hosts()

    def test_geo_block_is_not_retried(self):
        calls = {"n": 0}

        def _get(url, params=None, **k):
            calls["n"] += 1
            return FakeResp({}, status_code=451)

        exchanges.requests.get = _get
        exchanges.fetch_binance_klines("BTC-USD", "4h")
        self.assertEqual(calls["n"], 1, "a 451 must not be retried")

    def test_later_calls_skip_the_host_entirely(self):
        calls = {"n": 0}

        def _get(url, params=None, **k):
            calls["n"] += 1
            return FakeResp({}, status_code=451)

        exchanges.requests.get = _get
        for _ in range(10):
            exchanges.fetch_binance_klines("BTC-USD", "4h")
        self.assertEqual(calls["n"], 1, "the host should be tried once, then remembered")

    def test_403_is_treated_the_same_way(self):
        calls = {"n": 0}

        def _get(url, params=None, **k):
            calls["n"] += 1
            return FakeResp({}, status_code=403)

        exchanges.requests.get = _get
        exchanges.fetch_bybit_tickers()
        self.assertLessEqual(calls["n"], 2, "one attempt per Bybit host, no retries")

    def test_blocked_hosts_are_reported(self):
        exchanges.requests.get = lambda *a, **k: FakeResp({}, status_code=451)
        exchanges.fetch_binance_klines("BTC-USD", "4h")
        self.assertTrue(any("binance" in h for h in exchanges.blocked_hosts()))

    def test_an_ordinary_failure_is_still_retried(self):
        calls = {"n": 0}

        def _get(url, params=None, **k):
            calls["n"] += 1
            raise ConnectionError("blip")

        exchanges.requests.get = _get
        exchanges.configure(max_retries=2, backoff=0.0, spacing=0.0)
        exchanges.fetch_binance_klines("BTC-USD", "4h")
        exchanges.configure()
        self.assertEqual(calls["n"], 3, "transient errors should still retry")

    def test_a_working_host_is_never_blocked(self):
        exchanges.requests.get = lambda *a, **k: FakeResp(binance_rows())
        exchanges.fetch_binance_klines("BTC-USD", "4h")
        self.assertEqual(exchanges.blocked_hosts(), {})


class TestCoinbase(unittest.TestCase):
    def setUp(self):
        exchanges.reset_blocked_hosts()
        self.orig = exchanges.requests.get

    def tearDown(self):
        exchanges.requests.get = self.orig

    def test_product_mapping(self):
        self.assertEqual(exchanges.to_coinbase_product("BTC-USD"), "BTC-USD")
        self.assertEqual(exchanges.to_coinbase_product("HL:SOL"), "SOL-USD")

    def test_candles_returned_oldest_first(self):
        base = 1758000000
        rows = [[base + i * 300, 99.0, 101.0, 100.0, 100.5, 12.0] for i in range(5)][::-1]
        exchanges.requests.get = lambda *a, **k: FakeResp(rows)
        df, err = exchanges.fetch_coinbase_candles("BTC-USD", "5m")
        self.assertIsNone(err)
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertAlmostEqual(float(df["Close"].iloc[-1]), 100.5)

    def test_four_hour_not_offered(self):
        df, err = exchanges.fetch_coinbase_candles("BTC-USD", "4h")
        self.assertIsNone(df)
        self.assertIn("no 4h", err)

    def test_unlisted_coin(self):
        exchanges.requests.get = lambda *a, **k: FakeResp([])
        df, err = exchanges.fetch_coinbase_candles("ZZZ-USD", "5m")
        self.assertIsNone(df)
        self.assertIn("Not listed", err)


class TestBaseAssetStripsVenuePrefix(unittest.TestCase):
    """A Hyperliquid ticker must map to a plain symbol on other venues."""

    def setUp(self):
        exchanges.reset_blocked_hosts()

    def test_plain_ticker(self):
        self.assertEqual(exchanges.base_asset("BTC-USD"), "BTC")

    def test_hyperliquid_prefix_removed(self):
        self.assertEqual(exchanges.base_asset("HL:SOL"), "SOL")
        self.assertEqual(exchanges.base_asset("HL:ZEC-USD"), "ZEC")

    def test_every_exchange_mapper_agrees(self):
        self.assertEqual(exchanges.to_binance_symbol("HL:SOL"), "SOLUSDT")
        self.assertEqual(exchanges.to_bybit_symbol("HL:SOL"), "SOLUSDT")
        self.assertEqual(exchanges.to_coinbase_product("HL:SOL"), "SOL-USD")
