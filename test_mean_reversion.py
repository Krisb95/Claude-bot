import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from mean_reversion import (analyze, run_backtest, MeanReversionParams, atr,
                             levels_with_touches, READY, NOT_EXTENDED, NO_LEVEL,
                             NO_REJECTION)


def bars(closes, freq="1D"):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq=freq, tz="UTC")
    o = [closes[0]] + list(closes[:-1])
    nudge = [1e-6 * i for i in range(len(closes))]
    return pd.DataFrame({
        "Open": o,
        "High": [max(a, b) * (1 + 0.01 + d) for a, b, d in zip(o, closes, nudge)],
        "Low": [min(a, b) * (1 - 0.01 - d) for a, b, d in zip(o, closes, nudge)],
        "Close": closes}, index=idx)


def ranging(cycles=8, low=100.0, high=110.0, n=12):
    closes = [low]
    for _ in range(cycles):
        closes += list(np.linspace(closes[-1], high, n))[1:]
        closes += list(np.linspace(high, low, n))[1:]
    return closes


class TestExtension(unittest.TestCase):
    def test_price_at_the_average_is_not_a_setup(self):
        p = analyze("X", bars([100.0] * 60 + [100.2]))
        self.assertIn(p.stage, (NOT_EXTENDED, NO_LEVEL))
        self.assertIsNone(p.entry)

    def test_stretched_below_looks_for_a_long(self):
        closes = ranging()
        closes += [95.0]                       # a sharp drop below the range
        p = analyze("X", bars(closes))
        self.assertEqual(p.direction, "Long")

    def test_stretched_above_looks_for_a_short(self):
        closes = ranging()
        closes += [125.0]
        p = analyze("X", bars(closes))
        self.assertEqual(p.direction, "Short")


class TestPlanShape(unittest.TestCase):
    def _plan(self):
        closes = ranging()
        closes += [99.5]
        return analyze("X", bars(closes),
                        params=MeanReversionParams(require_rejection=False))

    def test_stop_is_wide_and_target_is_small(self):
        p = self._plan()
        if p.entry:
            risk = abs(p.entry - p.stop)
            reward = abs(p.target - p.entry)
            self.assertLess(reward, risk, "the target must be smaller than the risk")
            self.assertAlmostEqual(reward / risk, 0.4, places=6)

    def test_levels_ordered_for_a_long(self):
        p = self._plan()
        if p.entry and p.direction == "Long":
            self.assertLess(p.stop, p.entry)
            self.assertGreater(p.target, p.entry)

    def test_it_says_how_many_wins_one_loss_undoes(self):
        p = self._plan()
        if p.entry:
            self.assertIn("undoes", " ".join(p.reasons))

    def test_rejection_candle_required_by_default(self):
        closes = ranging()
        closes += [99.5]
        strict = analyze("X", bars(closes))
        loose = analyze("X", bars(closes), params=MeanReversionParams(require_rejection=False))
        if strict.stage == NO_REJECTION:
            self.assertEqual(loose.stage, READY)


class TestLevels(unittest.TestCase):
    def test_repeated_turns_become_levels(self):
        found = levels_with_touches(bars(ranging()), MeanReversionParams())
        self.assertTrue(found)
        self.assertTrue(all(c["touches"] >= 2 for c in found))

    def test_atr_needs_enough_bars(self):
        self.assertIsNone(atr(bars([100, 101])))


class TestBacktest(unittest.TestCase):
    def _random(self, seed=3, n=600):
        rng = np.random.default_rng(seed)
        return bars(list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))))

    def test_produces_trades(self):
        trades = run_backtest(self._random(), "X",
                              params=MeanReversionParams(require_rejection=False))
        self.assertTrue(trades)

    def test_high_win_rate_is_the_baseline_not_an_edge(self):
        """On random data a 0.4R target should win about 1/(1+0.4) = 71% of the
        time and still lose money. This is the trap the module warns about."""
        trades = run_backtest(self._random(), "X", fee_r=0.0,
                              params=MeanReversionParams(require_rejection=False))
        resolved = [t for t in trades if t["r_result"] is not None]
        if len(resolved) >= 20:
            wins = sum(1 for t in resolved if t["r_result"] > 0) / len(resolved)
            self.assertGreater(wins, 0.5, "small targets should win often")

    def test_no_trade_loses_more_than_one_r(self):
        for t in run_backtest(self._random(), "X",
                              params=MeanReversionParams(require_rejection=False)):
            if t["r_result"] is not None:
                self.assertGreaterEqual(t["r_result"], -1.0 - 1e-9)

    def test_no_lookahead(self):
        df = self._random()
        base = run_backtest(df, "X", params=MeanReversionParams(require_rejection=False))
        cutoff = df.index[400]
        tampered = df.copy()
        tampered.loc[tampered.index > cutoff,
                     ["Open", "High", "Low", "Close"]] *= 3.0
        after = run_backtest(tampered, "X",
                             params=MeanReversionParams(require_rejection=False))

        def early(ts):
            return [(t["time"], round(t["entry"], 8), round(t["stop"], 8))
                    for t in ts if t["time"] < cutoff]
        self.assertTrue(early(base))
        self.assertEqual(early(base), early(after))

    def test_short_history(self):
        self.assertEqual(run_backtest(bars([100] * 20), "X"), [])
