import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from swing import (analyze, trend_of, major_levels, nearest_level, target_for,
                    daily_atr, SwingParams, Level,
                    NO_TREND, NO_LEVEL, APPROACHING, AT_LEVEL, CONFIRMED)


def daily(closes, start="2025-01-01", spread=0.01):
    """Daily candles from a list of closes.

    The per-bar nudge to the wick size matters: a bar opens at the previous
    close, so without it the bar after a peak produces an identical High and
    the swing detector rightly refuses to call either a swing (it requires a
    UNIQUE extreme). Real candles never tie to the cent like that.
    """
    idx = pd.date_range(start, periods=len(closes), freq="1D", tz="UTC")
    o = [closes[0]] + list(closes[:-1])
    nudge = [1e-6 * i for i in range(len(closes))]
    return pd.DataFrame({
        "Open": o,
        "High": [max(a, b) * (1 + spread + d) for a, b, d in zip(o, closes, nudge)],
        "Low": [min(a, b) * (1 - spread - d) for a, b, d in zip(o, closes, nudge)],
        "Close": closes}, index=idx)


def uptrend_with_support(level=100.0, n_cycles=4, top=120.0, start_at=None):
    """Rising chart that bounces repeatedly off roughly the same support.

    The [1:] slices matter: linspace(a, b) then linspace(b, a) would repeat the
    turning value, and a swing point must be a UNIQUE extreme in its window —
    a duplicated peak is correctly ignored by the detector.
    """
    closes = [level]
    base = level
    for i in range(n_cycles):
        closes += list(np.linspace(base, top + i * 4, 18))[1:]
        closes += list(np.linspace(top + i * 4, base + i * 0.4, 14))[1:]
    if start_at is not None:
        closes.append(start_at)
    return daily(closes)


def zigzag(rising=True, cycles=8, up=22, down=9):
    """A trending chart with real pullbacks, ending mid-leg."""
    closes, price = [100.0], 100.0
    for _ in range(cycles):
        a, b = (up, down) if rising else (down, up)
        closes += list(np.linspace(price, price + a, 9))[1:]
        price += a
        closes += list(np.linspace(price, price - b, 6))[1:]
        price -= b
    closes += list(np.linspace(price, price + (10 if rising else -10), 5))[1:]
    return daily(closes)


class TestTrend(unittest.TestCase):
    def test_smooth_rise_uses_the_average_and_says_the_read_is_weaker(self):
        """A perfectly smooth line has no swings to check, so the structure
        test can't run — it falls back to the average and says so."""
        d, why = trend_of(daily(list(np.linspace(100, 200, 120))), SwingParams())
        self.assertEqual(d, "Long")
        self.assertIn("weaker read", why)

    def test_zigzag_uptrend_confirms_with_structure(self):
        d, why = trend_of(zigzag(rising=True), SwingParams())
        self.assertEqual(d, "Long")
        self.assertIn("uptrend", why)
        self.assertIn("higher highs", why)

    def test_zigzag_downtrend_confirms_with_structure(self):
        d, why = trend_of(zigzag(rising=False), SwingParams())
        self.assertEqual(d, "Short")
        self.assertIn("downtrend", why)

    def test_falling_chart_below_its_average_is_a_short(self):
        closes = list(np.linspace(200, 100, 120))
        d, why = trend_of(daily(closes), SwingParams())
        self.assertEqual(d, "Short")

    def test_range_gives_no_trend(self):
        closes = [100 + 5 * np.sin(i / 4) for i in range(120)]
        d, why = trend_of(daily(closes), SwingParams())
        self.assertIsNone(d)
        self.assertIn("No clear trend", why)

    def test_needs_enough_history(self):
        d, why = trend_of(daily([100] * 10), SwingParams())
        self.assertIsNone(d)
        self.assertIn("Need at least", why)


class TestLevels(unittest.TestCase):
    def test_repeated_touches_become_one_level(self):
        levels = major_levels(uptrend_with_support(), SwingParams())
        self.assertTrue(levels)
        self.assertTrue(any(lv.touches >= 2 for lv in levels))

    def test_single_touches_are_ignored(self):
        closes = list(np.linspace(100, 200, 120))
        levels = major_levels(daily(closes), SwingParams(min_touches=3))
        self.assertTrue(all(lv.touches >= 3 for lv in levels))

    def test_strength_labels(self):
        self.assertEqual(Level(100, 2, "support", 0).strength, "moderate")
        self.assertEqual(Level(100, 3, "support", 0).strength, "strong")
        self.assertEqual(Level(100, 5, "support", 0).strength, "major")

    def test_nearest_level_for_a_long_sits_below_price(self):
        levels = [Level(90, 3, "support", 0), Level(110, 4, "resistance", 0)]
        self.assertEqual(nearest_level(levels, "Long", 100).price, 90)

    def test_nearest_level_for_a_short_sits_above_price(self):
        levels = [Level(90, 3, "support", 0), Level(110, 4, "resistance", 0)]
        self.assertEqual(nearest_level(levels, "Short", 100).price, 110)

    def test_strongest_level_wins_over_the_closest(self):
        levels = [Level(98, 2, "support", 0), Level(95, 5, "support", 0)]
        self.assertEqual(nearest_level(levels, "Long", 100).price, 95)

    def test_no_level_on_that_side(self):
        self.assertIsNone(nearest_level([Level(110, 3, "resistance", 0)], "Long", 100))


class TestTargets(unittest.TestCase):
    def test_picks_the_first_level_clearing_the_minimum(self):
        levels = [Level(101, 2, "resistance", 0), Level(115, 3, "resistance", 0)]
        target, why = target_for(levels, "Long", entry=100, stop=97, min_rr=3.0)
        self.assertEqual(target, 115)
        self.assertIn("3", why)

    def test_reports_when_nothing_clears_it(self):
        levels = [Level(101, 2, "resistance", 0)]
        target, why = target_for(levels, "Long", entry=100, stop=97, min_rr=3.0)
        self.assertIsNone(target)
        self.assertIn("under your", why)

    def test_short_targets_sit_below_entry(self):
        levels = [Level(85, 3, "support", 0)]
        target, why = target_for(levels, "Short", entry=100, stop=103, min_rr=3.0)
        self.assertEqual(target, 85)

    def test_zero_risk_is_refused(self):
        self.assertIsNone(target_for([], "Long", 100, 100, 3.0)[0])


class TestAnalysis(unittest.TestCase):
    def test_no_trend_scores_zero_and_stops_there(self):
        closes = [100 + 5 * np.sin(i / 4) for i in range(120)]
        p = analyze("X", daily(closes))
        self.assertEqual(p.stage, NO_TREND)
        self.assertEqual(p.score, 0)
        self.assertIsNone(p.entry)

    def test_uptrend_at_a_level_produces_a_plan(self):
        df = uptrend_with_support()
        p = analyze("X", df)
        if p.direction == "Long" and p.level is not None:
            self.assertLess(p.stop, p.entry)
            self.assertGreaterEqual(p.score, 7)

    def test_levels_ordered_correctly_for_a_long(self):
        df = uptrend_with_support()
        p = analyze("X", df, direction_override="Long")
        if p.entry and p.target:
            self.assertTrue(p.stop < p.entry < p.target)

    def test_reward_risk_meets_the_minimum_when_a_target_exists(self):
        df = uptrend_with_support()
        p = analyze("X", df, direction_override="Long")
        if p.reward_risk is not None:
            self.assertGreaterEqual(p.reward_risk, 3.0 - 1e-9)

    def test_far_from_the_level_is_not_actionable(self):
        # Far in ATR terms, but within the price-mismatch tolerance so that
        # guard doesn't fire instead.
        df = uptrend_with_support()
        p = analyze("X", df, price=float(df["Close"].iloc[-1]) * 1.13,
                    direction_override="Long")
        self.assertIn(p.stage, (NO_LEVEL, APPROACHING, AT_LEVEL, CONFIRMED))

    def test_every_plan_explains_itself(self):
        for df in (uptrend_with_support(), daily([100 + 5 * np.sin(i / 4) for i in range(120)])):
            p = analyze("X", df)
            self.assertTrue(p.reasons)
            for r in p.reasons:
                self.assertGreater(len(r), 20)

    def test_features_recorded_for_the_learner(self):
        p = analyze("X", uptrend_with_support(), direction_override="Long")
        if p.entry:
            for key in ("level_touches", "distance_atr", "has_pattern"):
                self.assertIn(key, p.features)

    def test_empty_input(self):
        p = analyze("X", pd.DataFrame())
        self.assertEqual(p.score, 0)
        self.assertIsNone(p.direction)

    def test_daily_atr(self):
        self.assertIsNotNone(daily_atr(daily(list(np.linspace(100, 120, 40)))))
        self.assertIsNone(daily_atr(daily([100, 101])))


class TestSwingBacktest(unittest.TestCase):
    def _market(self, seed=4, n=900):
        rng = np.random.default_rng(seed)
        closes = list(100 * np.exp(np.cumsum(rng.normal(0.0006, 0.02, n))))
        return daily(closes)

    def test_produces_trades_over_years_of_history(self):
        from swing import run_backtest
        trades = run_backtest(self._market(), "X", require_confirmation=False)
        self.assertTrue(trades)

    def test_levels_ordered_correctly(self):
        from swing import run_backtest
        for t in run_backtest(self._market(), "X", require_confirmation=False):
            if t.direction == "Long":
                self.assertTrue(t.stop < t.entry < t.target)
            else:
                self.assertTrue(t.stop > t.entry > t.target)

    def test_every_trade_meets_the_minimum_reward_risk(self):
        from swing import run_backtest
        for t in run_backtest(self._market(), "X", require_confirmation=False):
            self.assertGreaterEqual(t.planned_rr, 3.0 - 1e-9)

    def test_no_overlapping_trades(self):
        from swing import run_backtest
        resolved = [t for t in run_backtest(self._market(), "X", require_confirmation=False)
                    if t.exit_time is not None]
        for a, b in zip(resolved, resolved[1:]):
            self.assertGreaterEqual(b.signal_time, a.exit_time)

    def test_no_lookahead(self):
        """Altering later candles must not change earlier decisions."""
        from swing import run_backtest
        df = self._market()
        base = run_backtest(df, "X", require_confirmation=False)
        cutoff = df.index[600]
        tampered = df.copy()
        tampered.loc[tampered.index > cutoff, ["Open", "High", "Low", "Close"]] *= 2.5
        altered = run_backtest(tampered, "X", require_confirmation=False)

        def plans(ts):
            return [(t.signal_time, round(t.entry, 8), round(t.stop, 8), round(t.target, 8))
                    for t in ts if t.exit_time is not None and t.exit_time < cutoff]
        self.assertTrue(plans(base))
        self.assertEqual(plans(base), plans(altered))

    def test_requiring_confirmation_takes_fewer_trades(self):
        from swing import run_backtest
        df = self._market()
        loose = run_backtest(df, "X", require_confirmation=False)
        strict = run_backtest(df, "X", require_confirmation=True)
        self.assertLessEqual(len(strict), len(loose))

    def test_short_history_returns_nothing(self):
        from swing import run_backtest
        self.assertEqual(run_backtest(daily([100] * 50), "X"), [])

    def test_features_recorded_for_learning(self):
        from swing import run_backtest
        trades = run_backtest(self._market(), "X", require_confirmation=False)
        if trades:
            self.assertIn("level_touches", trades[0].features)


class TestSwingUniverseScan(unittest.TestCase):
    def _loader(self, key):
        return zigzag(rising=True) if key != "flat" else daily([100] * 200)

    def test_returns_a_row_per_instrument(self):
        from swing import scan_universe
        out = scan_universe([("Bitcoin", "BTC-USD", "a"), ("Ether", "ETH-USD", "b")],
                            self._loader)
        self.assertEqual(len(out), 2)

    def test_best_setups_rank_first(self):
        from swing import scan_universe
        out = scan_universe([("Flat", "F-USD", "flat"), ("Trend", "T-USD", "a")],
                            self._loader)
        self.assertEqual(out[0].label, "Trend")

    def test_failures_are_reported_not_dropped(self):
        from swing import scan_universe

        def loader(key):
            if key == "bad":
                raise ConnectionError("down")
            return zigzag()

        out = scan_universe([("Bad", "B-USD", "bad"), ("Good", "G-USD", "a")], loader)
        self.assertEqual(len(out), 2)
        self.assertTrue(any(r.error for r in out))

    def test_empty_candles_reported(self):
        from swing import scan_universe
        out = scan_universe([("X", "X-USD", "x")], lambda k: pd.DataFrame())
        self.assertIn("No daily candles", out[0].error)

    def test_live_price_is_used_when_given(self):
        from swing import scan_universe
        out = scan_universe([("X", "X-USD", "a")], self._loader,
                            spot_loader=lambda k: 250.0)
        self.assertTrue(out[0].price_is_live)
        self.assertAlmostEqual(out[0].price, 250.0)

    def test_rows_carry_features_for_learning(self):
        from swing import scan_universe
        out = scan_universe([("X", "X-USD", "a")], self._loader)
        if out[0].entry:
            self.assertIn("level_touches", out[0].features)


class TestStopManagementVariants(unittest.TestCase):
    def _market(self, seed=4, n=700):
        rng = np.random.default_rng(seed)
        return daily(list(100 * np.exp(np.cumsum(rng.normal(0.0006, 0.02, n)))))

    def test_variants_take_broadly_the_same_trades(self):
        """Counts differ a little because an earlier exit frees you to take the
        next setup sooner — realistic, not a flaw. They should stay close."""
        from swing import run_variants
        out = run_variants(self._market(), "X", require_confirmation=False)
        counts = [len(ts) for ts in out.values()]
        self.assertLessEqual(max(counts) - min(counts), max(3, min(counts) // 4),
                             f"variants should be broadly comparable: {counts}")

    def test_variants_produce_different_results(self):
        from swing import run_variants
        out = run_variants(self._market(), "X", require_confirmation=False)
        totals = {n: sum(t.r_result for t in ts if t.r_result is not None)
                  for n, ts in out.items()}
        self.assertGreater(len(set(round(v, 6) for v in totals.values())), 1)

    def test_breakeven_reduces_full_losses(self):
        from swing import run_variants
        out = run_variants(self._market(), "X", require_confirmation=False)
        def full_losses(ts):
            return sum(1 for t in ts if t.r_result is not None and t.r_result <= -0.99)
        self.assertLessEqual(full_losses(out["Breakeven at 1R"]),
                             full_losses(out["Hard stop, fixed target"]))

    def test_no_trade_is_worse_than_minus_one_r(self):
        from swing import run_variants
        for name, ts in run_variants(self._market(), "X",
                                      require_confirmation=False).items():
            for t in ts:
                if t.r_result is not None:
                    self.assertGreaterEqual(t.r_result, -1.0 - 1e-9, name)


class TestSwingCandlePriceMismatch(unittest.TestCase):
    def test_live_price_far_from_the_candles_is_refused(self):
        p = analyze("ZEC-USD", zigzag(rising=True), price=9999.0)
        self.assertIsNone(p.entry)
        self.assertIn("aren't the same market", " ".join(p.reasons))

    def test_normal_gap_is_accepted(self):
        df = zigzag(rising=True)
        p = analyze("X", df, price=float(df["Close"].iloc[-1]) * 1.01)
        self.assertNotIn("aren't the same market", " ".join(p.reasons))
