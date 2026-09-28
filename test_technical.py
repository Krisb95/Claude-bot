import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from technical import find_swing_points, label_structure, find_equal_levels, fib_levels


def make_df(highs, lows, closes=None):
    closes = closes or [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        "Open": closes, "High": highs, "Low": lows, "Close": closes,
    })


class TestSwingPoints(unittest.TestCase):
    def test_detects_obvious_peak(self):
        highs = [1, 2, 3, 10, 3, 2, 1, 1, 1]
        lows =  [0, 1, 2, 9, 2, 1, 0, 0, 0]
        df = make_df(highs, lows)
        points = find_swing_points(df, left=2, right=2)
        peak_indices = [p.index for p in points if p.kind == "high" and p.price == 10]
        self.assertEqual(peak_indices, [3])

    def test_detects_obvious_trough(self):
        highs = [10, 9, 8, 1, 8, 9, 10, 10, 10]
        lows =  [9, 8, 7, 0, 7, 8, 9, 9, 9]
        df = make_df(highs, lows)
        points = find_swing_points(df, left=2, right=2)
        trough_indices = [p.index for p in points if p.kind == "low" and p.price == 0]
        self.assertEqual(trough_indices, [3])

    def test_no_lookahead_recent_bars_unconfirmed(self):
        # A swing at the very last bar can't be confirmed (no future bars to check).
        highs = [1, 2, 3, 4, 5, 100]
        lows = [0, 1, 2, 3, 4, 99]
        df = make_df(highs, lows)
        points = find_swing_points(df, left=2, right=2)
        # any swing point found near the tail end should be marked unconfirmed
        for p in points:
            if p.index >= len(df) - 2:
                self.assertFalse(p.confirmed)

    def test_too_short_dataframe_returns_empty(self):
        df = make_df([1, 2], [0, 1])
        self.assertEqual(find_swing_points(df, left=3, right=3), [])


class TestStructureLabeling(unittest.TestCase):
    def test_higher_high_higher_low_sequence(self):
        highs = [5, 1, 8, 1, 12, 1, 15]
        lows = [1, 0, 3, 2, 6, 4, 9]
        df = make_df(highs, lows)
        points = find_swing_points(df, left=1, right=1)
        labeled = label_structure(points)
        high_labels = [l.label for l in labeled if l.kind == "high" and l.label != ""]
        # Every subsequent high here is higher than the last -> all "HH"
        self.assertTrue(all(l == "HH" for l in high_labels))

    def test_lower_high_lower_low_sequence(self):
        highs = [15, 1, 12, 1, 8, 1, 5]
        lows = [9, 4, 6, 2, 3, 0, 1]
        df = make_df(highs, lows)
        points = find_swing_points(df, left=1, right=1)
        labeled = label_structure(points)
        high_labels = [l.label for l in labeled if l.kind == "high" and l.label != ""]
        self.assertTrue(all(l == "LH" for l in high_labels))


class TestEqualLevels(unittest.TestCase):
    def test_clusters_near_equal_highs(self):
        highs = [1, 10.00, 1, 10.02, 1, 5, 1, 10.01, 1]
        lows = [0, 9, 0, 9, 0, 4, 0, 9, 0]
        df = make_df(highs, lows)
        points = find_swing_points(df, left=1, right=1)
        clusters = find_equal_levels(points, tolerance_pct=0.005)
        high_clusters = [c for c in clusters if c["kind"] == "high" and c["touches"] >= 2]
        self.assertTrue(len(high_clusters) >= 1)


class TestFibLevels(unittest.TestCase):
    def test_long_retracement_between_high_and_low(self):
        levels = fib_levels(swing_low=100, swing_high=200, direction="Long")
        for key in ("retr_0.382", "retr_0.5", "retr_0.618", "retr_0.786"):
            self.assertGreater(levels[key], 100)
            self.assertLess(levels[key], 200)
        # deeper retracement ratio should be a lower price for a Long
        self.assertLess(levels["retr_0.786"], levels["retr_0.382"])

    def test_extension_beyond_high_for_long(self):
        levels = fib_levels(swing_low=100, swing_high=200, direction="Long")
        self.assertGreater(levels["ext_1.272"], 200)
        self.assertGreater(levels["ext_1.618"], levels["ext_1.272"])

    def test_short_mirrors_long(self):
        levels = fib_levels(swing_low=100, swing_high=200, direction="Short")
        for key in ("retr_0.382", "retr_0.5", "retr_0.618", "retr_0.786"):
            self.assertGreater(levels[key], 100)
            self.assertLess(levels[key], 200)
        self.assertLess(levels["ext_1.272"], 100)

    def test_invalid_span_raises(self):
        with self.assertRaises(ValueError):
            fib_levels(swing_low=200, swing_high=100, direction="Long")


if __name__ == "__main__":
    unittest.main()
