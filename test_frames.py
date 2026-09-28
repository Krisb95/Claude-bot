import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from frames import check, median_gap, first_valid


def frame(n=100, freq="5min", end=None, price=100.0):
    end = end or pd.Timestamp.now(tz="UTC").floor("min")
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1,
                          "Close": price}, index=idx)


class TestSpacing(unittest.TestCase):
    def test_five_minute_frame_passes(self):
        self.assertTrue(check(frame(freq="5min"), "5m").ok)

    def test_four_hour_candles_labelled_5m_are_rejected(self):
        """The ZEC bug: a frame of coarse candles read as 5m data."""
        result = check(frame(freq="4h"), "5m")
        self.assertFalse(result.ok)
        self.assertIn("wrong timeframe", result.reason)

    def test_daily_candles_labelled_5m_are_rejected(self):
        self.assertFalse(check(frame(freq="1D"), "5m").ok)

    def test_five_minute_candles_labelled_daily_are_rejected(self):
        self.assertFalse(check(frame(freq="5min"), "1d").ok)

    def test_daily_frame_passes_as_daily(self):
        self.assertTrue(check(frame(freq="1D"), "1d").ok)

    def test_a_missing_candle_is_tolerated(self):
        df = frame(n=100, freq="5min")
        df = df.drop(df.index[40:42])           # a small gap, as real feeds have
        self.assertTrue(check(df, "5m").ok)

    def test_median_gap(self):
        self.assertAlmostEqual(median_gap(frame(freq="5min")), 300.0)


class TestFreshness(unittest.TestCase):
    def test_recent_frame_passes(self):
        self.assertTrue(check(frame(), "5m").ok)

    def test_stale_five_minute_data_is_rejected(self):
        old = frame(end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6))
        result = check(old, "5m")
        self.assertFalse(result.ok)
        self.assertIn("stale", result.reason)

    def test_months_old_data_is_rejected(self):
        old = frame(end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60))
        self.assertFalse(check(old, "5m").ok)

    def test_daily_data_a_day_old_is_fine(self):
        day_old = frame(freq="1D", end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1))
        self.assertTrue(check(day_old, "1d").ok)


class TestOtherChecks(unittest.TestCase):
    def test_empty_frame(self):
        self.assertFalse(check(pd.DataFrame(), "5m").ok)
        self.assertFalse(check(None, "5m").ok)

    def test_too_few_candles(self):
        result = check(frame(n=5), "5m")
        self.assertFalse(result.ok)
        self.assertIn("too few", result.reason.lower())

    def test_detail_is_readable(self):
        self.assertIn("bars", check(frame(), "5m").detail)


class TestFirstValid(unittest.TestCase):
    def test_picks_the_first_good_source(self):
        df, name, note = first_valid([("Bybit", None),
                                       ("Binance", frame(freq="4h")),
                                       ("Hyperliquid", frame(freq="5min"))], "5m")
        self.assertEqual(name, "Hyperliquid")
        self.assertIn("Binance", note)

    def test_explains_when_everything_fails(self):
        df, name, note = first_valid([("Bybit", None),
                                       ("Binance", frame(freq="1D"))], "5m")
        self.assertIsNone(df)
        self.assertIsNone(name)
        self.assertIn("Bybit", note)
        self.assertIn("Binance", note)

    def test_no_candidates(self):
        self.assertEqual(first_valid([], "5m")[0], None)
