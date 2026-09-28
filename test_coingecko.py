import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
import pandas as pd
import coingecko
from coingecko import fetch_ohlc, fetch_spot_price, build_frames, GRANULARITY_BY_DAYS

# Disable real sleeping so retry/backoff logic does not slow the suite.
coingecko.configure_retries(max_retries=0, backoff=0.0, min_spacing=0.0)


class FakeResp:
    def __init__(self, payload, raise_exc=None, status_code=200):
        self._payload = payload
        self._raise = raise_exc
        self.status_code = status_code

    def raise_for_status(self):
        if self._raise:
            raise self._raise

    def json(self):
        return self._payload


def patch_get(payload=None, exc=None):
    def _get(*a, **k):
        if exc:
            raise exc
        return FakeResp(payload)
    return _get


class TestFetchOhlc(unittest.TestCase):
    def setUp(self):
        self.original = coingecko.requests.get

    def tearDown(self):
        coingecko.requests.get = self.original

    def test_parses_candles_into_ohlc_frame(self):
        rows = [[1758000000000, 10.0, 12.0, 9.0, 11.0],
                [1758014400000, 11.0, 13.0, 10.5, 12.5]]
        coingecko.requests.get = patch_get(rows)
        df, label, err = fetch_ohlc("hyperliquid", days=30)
        self.assertIsNone(err)
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df["Close"].iloc[-1], 12.5)
        self.assertEqual(str(df.index.tz), "UTC")

    def test_thirty_days_gives_native_4h(self):
        self.assertEqual(GRANULARITY_BY_DAYS[30][0], "4h")

    def test_one_day_gives_30m(self):
        self.assertEqual(GRANULARITY_BY_DAYS[1][0], "30m")

    def test_empty_list_is_an_error_not_an_empty_frame(self):
        coingecko.requests.get = patch_get([])
        df, label, err = fetch_ohlc("nothing", days=30)
        self.assertIsNone(df)
        self.assertIsNotNone(err)

    def test_network_error_is_reported(self):
        coingecko.requests.get = patch_get(exc=ConnectionError("offline"))
        df, label, err = fetch_ohlc("bitcoin", days=30)
        self.assertIsNone(df)
        self.assertIn("ConnectionError", err)

    def test_unsupported_days_snaps_to_nearest_supported(self):
        rows = [[1758000000000, 1.0, 1.0, 1.0, 1.0]]
        coingecko.requests.get = patch_get(rows)
        df, label, err = fetch_ohlc("bitcoin", days=13)  # not an exact key
        self.assertIsNone(err)
        self.assertIn(label, [v[0] for v in GRANULARITY_BY_DAYS.values()])

    def test_null_closes_are_dropped(self):
        rows = [[1758000000000, 1.0, 1.0, 1.0, 1.0],
                [1758014400000, None, None, None, None]]
        coingecko.requests.get = patch_get(rows)
        df, label, err = fetch_ohlc("bitcoin", days=30)
        self.assertEqual(len(df), 1)


class TestSpotPrice(unittest.TestCase):
    def setUp(self):
        self.original = coingecko.requests.get

    def tearDown(self):
        coingecko.requests.get = self.original

    def test_returns_price_and_timestamp(self):
        ts = int(datetime.now(timezone.utc).timestamp())
        coingecko.requests.get = patch_get({"hyperliquid": {"usd": 42.5, "last_updated_at": ts}})
        price, updated, err = fetch_spot_price("hyperliquid")
        self.assertAlmostEqual(price, 42.5)
        self.assertIsNotNone(updated)
        self.assertIsNone(err)

    def test_missing_coin_is_an_error(self):
        coingecko.requests.get = patch_get({})
        price, updated, err = fetch_spot_price("nope")
        self.assertIsNone(price)
        self.assertIn("No price", err)

    def test_non_positive_price_rejected(self):
        coingecko.requests.get = patch_get({"x": {"usd": 0}})
        price, updated, err = fetch_spot_price("x")
        self.assertIsNone(price)

    def test_missing_timestamp_still_returns_price(self):
        coingecko.requests.get = patch_get({"x": {"usd": 5.0}})
        price, updated, err = fetch_spot_price("x")
        self.assertEqual(price, 5.0)
        self.assertIsNone(updated)


class TestBuildFrames(unittest.TestCase):
    def setUp(self):
        self.original = coingecko.requests.get

    def tearDown(self):
        coingecko.requests.get = self.original

    def test_builds_all_expected_frames(self):
        rows = [[1758000000000 + i * 14400000, 10.0 + i, 12.0 + i, 9.0 + i, 11.0 + i]
                for i in range(30)]
        coingecko.requests.get = patch_get(rows)
        frames, problems = build_frames("hyperliquid")
        for key in ("1d", "4h", "5m", "1h"):
            self.assertIn(key, frames)
            self.assertFalse(frames[key].empty)

    def test_substitutions_are_reported_not_hidden(self):
        rows = [[1758000000000 + i * 14400000, 10.0, 12.0, 9.0, 11.0] for i in range(10)]
        coingecko.requests.get = patch_get(rows)
        frames, problems = build_frames("hyperliquid")
        joined = " ".join(problems).lower()
        self.assertIn("4-day candles", joined)   # daily is approximated
        self.assertIn("30-minute", joined)        # 1h/5m are approximated

    def test_total_failure_reports_problems_and_empty_frames(self):
        coingecko.requests.get = patch_get(exc=ConnectionError("offline"))
        frames, problems = build_frames("hyperliquid")
        self.assertTrue(all(f.empty for f in frames.values()))
        self.assertTrue(len(problems) >= 3)


if __name__ == "__main__":
    unittest.main()


class TestBuildFramesV2(unittest.TestCase):
    """The improved frame builder: native 4H OHLC plus 5-minute points
    resampled into finer candles than /ohlc alone can provide."""

    def setUp(self):
        self.original = coingecko.requests.get

    def tearDown(self):
        coingecko.requests.get = self.original

    def _router(self):
        """Serve /ohlc and /market_chart with appropriate shapes."""
        base = 1758000000000

        def _get(url, params=None, timeout=None, **k):
            if "/ohlc" in url:
                rows = [[base + i * 14400000, 10.0 + i, 12.0 + i, 9.0 + i, 11.0 + i]
                        for i in range(40)]
                return FakeResp(rows)
            days = (params or {}).get("days", 1)
            step = 300000 if days == 1 else 86400000
            n = 288 if days == 1 else 365
            prices = [[base + i * step, 100.0 + (i % 17)] for i in range(n)]
            return FakeResp({"prices": prices})
        return _get

    def test_builds_every_frame(self):
        coingecko.requests.get = self._router()
        frames, problems = coingecko.build_frames_v2("hyperliquid")
        for key in ("1d", "4h", "1h", "5m"):
            self.assertFalse(frames[key].empty, f"{key} frame is empty")

    def test_four_hour_frame_is_native_not_resampled(self):
        coingecko.requests.get = self._router()
        frames, problems = coingecko.build_frames_v2("hyperliquid")
        # Native /ohlc candles have genuinely differing High and Low.
        f4 = frames["4h"]
        self.assertTrue((f4["High"] > f4["Low"]).all())

    def test_hourly_resample_produces_real_ohlc_from_5m_points(self):
        coingecko.requests.get = self._router()
        frames, problems = coingecko.build_frames_v2("hyperliquid")
        h = frames["1h"]
        # 12 five-minute points per hour means High and Low should differ.
        self.assertTrue((h["High"] >= h["Low"]).all())
        self.assertTrue((h["High"] > h["Low"]).any())

    def test_daily_flat_candle_limitation_is_disclosed(self):
        coingecko.requests.get = self._router()
        frames, problems = coingecko.build_frames_v2("hyperliquid")
        joined = " ".join(problems).lower()
        self.assertIn("equal the close", joined)

    def test_series_to_ohlc_aggregates_correctly(self):
        idx = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        series = pd.Series([1, 5, 3, 9, 2, 7, 4, 8, 6, 2, 10, 3], index=idx)
        out = coingecko.series_to_ohlc(series, "1h")
        self.assertEqual(len(out), 1)
        self.assertEqual(out["Open"].iloc[0], 1)
        self.assertEqual(out["High"].iloc[0], 10)
        self.assertEqual(out["Low"].iloc[0], 1)
        self.assertEqual(out["Close"].iloc[0], 3)

    def test_empty_series_returns_empty_frame(self):
        self.assertTrue(coingecko.series_to_ohlc(pd.Series(dtype=float), "1h").empty)

    def test_price_series_network_failure_reported(self):
        coingecko.requests.get = patch_get(exc=ConnectionError("offline"))
        series, err = coingecko.fetch_price_series("bitcoin", days=1)
        self.assertIsNone(series)
        self.assertIn("ConnectionError", err)

    def test_price_series_missing_key_reported(self):
        coingecko.requests.get = patch_get({"not_prices": []})
        series, err = coingecko.fetch_price_series("bitcoin", days=1)
        self.assertIsNone(series)
        self.assertIsNotNone(err)


class TestRateLimitHandling(unittest.TestCase):
    """429 is CoinGecko's free-tier rate limit. It must be retried and then
    reported clearly, never silently turned into 'no data'."""

    def setUp(self):
        self.original = coingecko.requests.get
        coingecko.configure_retries(max_retries=2, backoff=0.0, min_spacing=0.0)

    def tearDown(self):
        coingecko.requests.get = self.original
        coingecko.configure_retries(max_retries=0, backoff=0.0, min_spacing=0.0)

    def test_429_is_retried_then_reported(self):
        calls = {"n": 0}

        def _get(*a, **k):
            calls["n"] += 1
            return FakeResp({}, status_code=429)

        coingecko.requests.get = _get
        df, label, err = fetch_ohlc("bitcoin", days=30)
        self.assertIsNone(df)
        self.assertIn("429", err)
        self.assertEqual(calls["n"], 3)  # 1 initial + 2 retries

    def test_429_then_success_recovers(self):
        calls = {"n": 0}
        rows = [[1758000000000, 1.0, 2.0, 0.5, 1.5]]

        def _get(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResp({}, status_code=429)
            return FakeResp(rows)

        coingecko.requests.get = _get
        df, label, err = fetch_ohlc("bitcoin", days=30)
        self.assertIsNone(err)
        self.assertEqual(len(df), 1)

    def test_rate_limit_message_is_human_readable(self):
        coingecko.requests.get = lambda *a, **k: FakeResp({}, status_code=429)
        price, updated, err = fetch_spot_price("bitcoin")
        self.assertIsNone(price)
        self.assertIn("rate limit", err.lower())

    def test_partial_failure_still_returns_good_frames(self):
        """A rate-limited daily call must not discard working 4H candles."""
        base = 1758000000000

        def _get(url, params=None, **k):
            if "/ohlc" in url:
                rows = [[base + i * 14400000, 10.0 + i, 12.0 + i, 9.0 + i, 11.0 + i]
                        for i in range(40)]
                return FakeResp(rows)
            return FakeResp({}, status_code=429)   # market_chart is limited

        coingecko.requests.get = _get
        frames, problems = coingecko.build_frames_v2("hyperliquid")
        self.assertFalse(frames["4h"].empty)       # good data preserved
        self.assertTrue(frames["1d"].empty)        # failed frame is empty
        self.assertTrue(any("429" in p or "rate limit" in p.lower() for p in problems))


class TestPeriodChanges(unittest.TestCase):
    def setUp(self):
        self.original = coingecko.requests.get

    def tearDown(self):
        coingecko.requests.get = self.original

    def test_returns_symbol_to_change(self):
        coingecko.requests.get = patch_get([
            {"symbol": "btc", "price_change_percentage_90d_in_currency": 12.5},
            {"symbol": "eth", "price_change_percentage_90d_in_currency": 30.0},
        ])
        changes, err = coingecko.fetch_period_changes()
        self.assertIsNone(err)
        self.assertEqual(changes, {"BTC": 12.5, "ETH": 30.0})

    def test_stablecoins_excluded(self):
        coingecko.requests.get = patch_get([
            {"symbol": "btc", "price_change_percentage_90d_in_currency": 5.0},
            {"symbol": "usdt", "price_change_percentage_90d_in_currency": 0.01},
        ])
        self.assertNotIn("USDT", coingecko.fetch_period_changes()[0])

    def test_missing_change_skipped(self):
        coingecko.requests.get = patch_get([
            {"symbol": "btc", "price_change_percentage_90d_in_currency": 5.0},
            {"symbol": "new", "price_change_percentage_90d_in_currency": None},
        ])
        self.assertEqual(list(coingecko.fetch_period_changes()[0]), ["BTC"])

    def test_failure_reported(self):
        coingecko.requests.get = patch_get(exc=ConnectionError("down"))
        changes, err = coingecko.fetch_period_changes()
        self.assertEqual(changes, {})
        self.assertIn("ConnectionError", err)


class TestCoinSearch(unittest.TestCase):
    def setUp(self):
        self.original = coingecko.requests.get

    def tearDown(self):
        coingecko.requests.get = self.original

    def test_returns_only_exact_ticker_matches(self):
        coingecko.requests.get = patch_get({"coins": [
            {"id": "derive", "symbol": "drv", "name": "Derive", "market_cap_rank": 300},
            {"id": "curve-dao-token", "symbol": "crv", "name": "Curve", "market_cap_rank": 90},
        ]})
        found, err = coingecko.search_coins("DRV")
        self.assertIsNone(err)
        self.assertEqual([c["id"] for c in found], ["derive"])

    def test_several_projects_sharing_a_ticker_are_all_returned(self):
        coingecko.requests.get = patch_get({"coins": [
            {"id": "derive", "symbol": "drv", "name": "Derive", "market_cap_rank": 300},
            {"id": "other-drv", "symbol": "drv", "name": "Other DRV", "market_cap_rank": None},
        ]})
        found, _ = coingecko.search_coins("drv")
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["id"], "derive")     # ranked coin first

    def test_no_match_reported(self):
        coingecko.requests.get = patch_get({"coins": []})
        found, err = coingecko.search_coins("ZZZZ")
        self.assertEqual(found, [])
        self.assertIn("ZZZZ", err)

    def test_empty_symbol(self):
        self.assertEqual(coingecko.search_coins("")[0], [])

    def test_network_failure(self):
        coingecko.requests.get = patch_get(exc=ConnectionError("down"))
        found, err = coingecko.search_coins("DRV")
        self.assertEqual(found, [])
        self.assertIn("ConnectionError", err)


class TestBulkPrices(unittest.TestCase):
    def setUp(self):
        self.original = coingecko.requests.get

    def tearDown(self):
        coingecko.requests.get = self.original

    def test_prices_for_many_coins_in_one_call(self):
        calls = []

        def _get(url, params=None, **k):
            calls.append(params)
            return patch_get({"bitcoin": {"usd": 82000.0},
                              "near": {"usd": 5.49}})(url, params)

        coingecko.requests.get = _get
        prices, err = coingecko.fetch_prices_bulk(["bitcoin", "near"])
        self.assertIsNone(err)
        self.assertAlmostEqual(prices["near"], 5.49)
        self.assertEqual(len(calls), 1)

    def test_duplicates_and_blanks_ignored(self):
        coingecko.requests.get = patch_get({"bitcoin": {"usd": 1.0}})
        prices, _ = coingecko.fetch_prices_bulk(["bitcoin", "bitcoin", None, ""])
        self.assertEqual(list(prices), ["bitcoin"])

    def test_empty_input(self):
        self.assertEqual(coingecko.fetch_prices_bulk([]), ({}, None))

    def test_network_failure(self):
        coingecko.requests.get = patch_get(exc=ConnectionError("down"))
        prices, err = coingecko.fetch_prices_bulk(["bitcoin"])
        self.assertEqual(prices, {})
        self.assertIsNotNone(err)
