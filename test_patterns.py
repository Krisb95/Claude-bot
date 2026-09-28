import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import patterns
from patterns import detect, bias_of, confirms, contradicts, summarise, BULLISH, BEARISH


def bars(rows):
    """rows: (open, high, low, close)"""
    idx = pd.date_range("2026-06-01", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def names(df):
    return {p.name for p in detect(df)}


class TestSingleCandle(unittest.TestCase):
    def test_doji(self):
        self.assertIn("Doji", names(bars([(100, 101, 99, 100), (100, 102, 98, 100.05)])))

    def test_hammer(self):
        found = names(bars([(100, 101, 99, 100), (100, 100.3, 97, 100.1)]))
        self.assertIn("Hammer / bullish pin bar", found)

    def test_shooting_star(self):
        found = names(bars([(100, 101, 99, 100), (100, 103, 99.9, 100.1)]))
        self.assertIn("Shooting star / bearish pin bar", found)

    def test_strong_candle(self):
        found = names(bars([(100, 101, 99, 100), (100, 104.2, 99.9, 104)]))
        self.assertIn("Strong bullish candle", found)

    def test_ordinary_candle_matches_nothing(self):
        self.assertEqual(names(bars([(100, 101, 99, 100.4), (100.4, 101.2, 99.8, 100.7)])),
                         set())


class TestTwoCandle(unittest.TestCase):
    def test_bullish_engulfing(self):
        found = names(bars([(102, 102.5, 100, 100.5), (100.2, 103.5, 100, 103)]))
        self.assertIn("Bullish engulfing", found)

    def test_bearish_engulfing(self):
        found = names(bars([(100, 102.5, 99.8, 102), (102.3, 102.5, 99, 99.5)]))
        self.assertIn("Bearish engulfing", found)

    def test_partial_overlap_is_not_engulfing(self):
        found = names(bars([(102, 102.5, 100, 100.5), (101, 102, 100.8, 101.8)]))
        self.assertNotIn("Bullish engulfing", found)

    def test_inside_bar(self):
        found = names(bars([(100, 105, 95, 102), (101, 103, 98, 102)]))
        self.assertIn("Inside bar", found)


class TestThreeCandle(unittest.TestCase):
    def test_morning_star(self):
        found = names(bars([(105, 105.2, 100, 100.5), (100.3, 100.8, 99.8, 100.4),
                            (100.5, 104.5, 100.4, 104)]))
        self.assertIn("Morning star", found)

    def test_evening_star(self):
        found = names(bars([(100, 105.2, 99.8, 105), (105.2, 105.6, 104.8, 105.1),
                            (105, 105.1, 100.5, 100.8)]))
        self.assertIn("Evening star", found)


class TestInterpretation(unittest.TestCase):
    def test_bias_and_confirmation(self):
        df = bars([(102, 102.5, 100, 100.5), (100.2, 103.5, 100, 103)])
        found = detect(df)
        self.assertEqual(bias_of(found), BULLISH)
        self.assertTrue(confirms(found, "Long"))
        self.assertTrue(contradicts(found, "Short"))

    def test_no_patterns_is_neutral(self):
        self.assertEqual(bias_of([]), "neutral")
        self.assertFalse(confirms([], "Long"))
        self.assertIsNone(summarise([]))

    def test_every_pattern_explains_itself_without_predicting(self):
        for df in (bars([(102, 102.5, 100, 100.5), (100.2, 103.5, 100, 103)]),
                   bars([(100, 101, 99, 100), (100, 100.3, 97, 100.1)])):
            for p in detect(df):
                self.assertTrue(len(p.description) > 30)
                self.assertTrue(len(p.convention) > 15)
                for word in ("will rise", "will fall", "guarantees", "predicts"):
                    self.assertNotIn(word, p.convention.lower())

    def test_needs_at_least_two_candles(self):
        self.assertEqual(detect(bars([(100, 101, 99, 100)])), [])
        self.assertEqual(detect(None), [])


class TestSwingPatterns(unittest.TestCase):
    def test_double_bottom_found(self):
        seq = ([100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 100, 99, 98, 97, 96.2,
                97, 98, 99, 100, 101, 102])
        rows = [(c, c + 0.2, c - 0.2, c) for c in seq]
        found = {p.name for p in patterns.swing_patterns(bars(rows))}
        self.assertIn("Double bottom", found)

    def test_nothing_found_in_a_trend(self):
        rows = [(100 + i, 100.2 + i, 99.8 + i, 100 + i) for i in range(25)]
        self.assertEqual(patterns.swing_patterns(bars(rows)), [])

    def test_short_frame(self):
        self.assertEqual(patterns.swing_patterns(bars([(1, 1, 1, 1)])), [])


class TestPinBarDefinition(unittest.TestCase):
    """A pin bar has a small body by definition, so the opposite wick must be
    judged against the candle's range, not its body."""

    def test_classic_hammer_is_not_reported_as_only_a_doji(self):
        found = names(bars([(100, 101, 99, 100), (100, 100.3, 97, 100.1)]))
        self.assertIn("Hammer / bullish pin bar", found)
        self.assertNotIn("Doji", found)

    def test_a_true_doji_is_still_a_doji(self):
        found = names(bars([(100, 101, 99, 100), (100, 101.5, 98.5, 100.02)]))
        self.assertIn("Doji", found)

    def test_long_wicks_on_both_sides_is_neither_pin_bar(self):
        found = names(bars([(100, 101, 99, 100), (100, 103, 97, 100.1)]))
        self.assertNotIn("Hammer / bullish pin bar", found)
        self.assertNotIn("Shooting star / bearish pin bar", found)
