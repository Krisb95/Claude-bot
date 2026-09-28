import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scanner import (detect_regime, structure_direction, detect_sweep_and_reclaim,
                      fib_confluence, is_extended, opposing_liquidity_ahead,
                      resample_to_4h, analyze_candidate, nearest_level)


def ohlc_from_closes(closes, spread=0.5, start="2026-01-01", freq="1D"):
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({
        "Open": closes,
        "High": [c + spread for c in closes],
        "Low": [c - spread for c in closes],
        "Close": closes,
        "Volume": [1000] * len(closes),
    }, index=idx)


class TestRegime(unittest.TestCase):
    def test_steady_uptrend_is_bullish(self):
        closes = list(np.linspace(100, 200, 120))
        df = ohlc_from_closes(closes)
        regime, reason = detect_regime(df)
        self.assertEqual(regime, "bullish")
        self.assertIn("above", reason)

    def test_steady_downtrend_is_bearish(self):
        closes = list(np.linspace(200, 100, 120))
        df = ohlc_from_closes(closes)
        regime, _ = detect_regime(df)
        self.assertEqual(regime, "bearish")

    def test_choppy_sideways_is_ranging(self):
        closes = [100 + (i % 5) for i in range(120)]
        df = ohlc_from_closes(closes)
        regime, _ = detect_regime(df)
        self.assertEqual(regime, "ranging")

    def test_insufficient_history_is_unknown_not_guessed(self):
        df = ohlc_from_closes(list(np.linspace(100, 110, 10)))
        regime, reason = detect_regime(df)
        self.assertEqual(regime, "unknown")
        self.assertIn("Not enough", reason)


def zigzag(n_legs=8, leg_len=7, start=100.0, rise=15.0, retrace=0.5, rising=True):
    """Zigzag with distinct turning points. Avoids repeating the peak/trough
    value across adjacent legs, which would fail the swing detector's
    uniqueness check."""
    closes = [start]
    level = start
    for _ in range(n_legs):
        if rising:
            high = level + rise
            closes += list(np.linspace(level, high, leg_len))[1:]
            level = high - rise * retrace
            closes += list(np.linspace(high, level, leg_len))[1:]
        else:
            low = level - rise
            closes += list(np.linspace(level, low, leg_len))[1:]
            level = low + rise * retrace
            closes += list(np.linspace(low, level, leg_len))[1:]
    return closes


class TestStructureDirection(unittest.TestCase):
    def test_rising_swings_favour_long(self):
        df = ohlc_from_closes(zigzag(rising=True))
        direction, reason = structure_direction(df)
        self.assertEqual(direction, "Long")
        self.assertIn("HH", reason)

    def test_falling_swings_favour_short(self):
        df = ohlc_from_closes(zigzag(rising=False, start=300.0))
        direction, reason = structure_direction(df)
        self.assertEqual(direction, "Short")

    def test_too_short_returns_none_not_a_guess(self):
        df = ohlc_from_closes([100, 101, 102])
        direction, reason = structure_direction(df)
        self.assertIsNone(direction)


class TestSweepReclaim(unittest.TestCase):
    def test_wick_only_pierce_without_reclaim_scores_false(self):
        # Build history with a clear swing low, then pierce it and STAY below.
        closes = [100, 95, 105, 90, 106, 92, 108, 94, 110] * 5
        df = ohlc_from_closes(closes)
        # Force the tail to break below everything and close there.
        tail = ohlc_from_closes([70] * 35, start="2027-01-01")
        df = pd.concat([df, tail])
        result, reason = detect_sweep_and_reclaim(df, "Long", lookback=30)
        self.assertIn(result, (False, None))
        if result is False:
            self.assertTrue("no reclaim" in reason.lower() or "no sweep" in reason.lower())

    def test_insufficient_data_returns_none(self):
        df = ohlc_from_closes([100, 101, 102])
        result, reason = detect_sweep_and_reclaim(df, "Long")
        self.assertIsNone(result)
        self.assertIn("Not enough", reason)


class TestFibConfluence(unittest.TestCase):
    def test_returns_none_when_no_anchors(self):
        df = ohlc_from_closes([100, 101])
        ok, reason, levels = fib_confluence(df, 100, "Long")
        self.assertIsNone(ok)

    def test_produces_levels_from_real_swings(self):
        closes = [100, 120, 105, 130, 110, 140, 115, 150, 120, 145] * 4
        df = ohlc_from_closes(closes)
        ok, reason, levels = fib_confluence(df, 130, "Long")
        if levels:
            self.assertTrue(any(k.startswith("retr_") for k in levels))


class TestExtension(unittest.TestCase):
    def test_far_from_sma_is_extended(self):
        closes = list(np.linspace(100, 105, 60)) + [180]
        df = ohlc_from_closes(closes)
        extended, reason = is_extended(df, 180)
        self.assertTrue(extended)
        self.assertIn("ATR", reason)

    def test_near_sma_is_not_extended(self):
        closes = list(np.linspace(100, 105, 60))
        df = ohlc_from_closes(closes)
        extended, reason = is_extended(df, float(df["Close"].iloc[-1]))
        self.assertFalse(extended)

    def test_insufficient_bars_returns_none(self):
        df = ohlc_from_closes([100, 101, 102])
        extended, _ = is_extended(df, 101)
        self.assertIsNone(extended)


class TestResample(unittest.TestCase):
    def test_1h_resamples_to_4h_with_correct_ohlc(self):
        idx = pd.date_range("2026-01-01 00:00", periods=8, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [1, 2, 3, 4, 5, 6, 7, 8],
            "High": [10, 11, 12, 13, 14, 15, 16, 17],
            "Low": [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8],
            "Close": [2, 3, 4, 5, 6, 7, 8, 9],
        }, index=idx)
        out = resample_to_4h(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["Open"].iloc[0], 1)      # first open of the 4 bars
        self.assertEqual(out["Close"].iloc[0], 5)     # last close of the 4 bars
        self.assertEqual(out["High"].iloc[0], 13)     # max high
        self.assertAlmostEqual(out["Low"].iloc[0], 0.2)  # min low

    def test_empty_input_returns_empty(self):
        self.assertTrue(resample_to_4h(pd.DataFrame()).empty)


class TestAnalyzeCandidate(unittest.TestCase):
    def _frames(self, closes_1d, closes_1h):
        return {
            "1d": ohlc_from_closes(closes_1d),
            "1h": ohlc_from_closes(closes_1h, start="2026-06-01", freq="1h"),
            "4h": resample_to_4h(ohlc_from_closes(closes_1h, start="2026-06-01", freq="1h")),
        }

    def test_uptrend_suggests_long(self):
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        self.assertEqual(result.regime_1d, "bullish")
        self.assertEqual(result.direction, "Long")

    def test_missing_data_is_reported_not_invented(self):
        frames = {"1d": pd.DataFrame(), "1h": pd.DataFrame(), "4h": pd.DataFrame()}
        result = analyze_candidate("TEST", frames)
        self.assertIsNone(result.current_price)
        self.assertTrue(len(result.data_problems) > 0)

    def test_unavailable_evidence_is_none_not_false(self):
        # Tiny dataset: most checks cannot be evaluated and must return None
        # (which scores zero) rather than a fabricated False/True.
        frames = self._frames([100, 101, 102], [100, 101, 102])
        result = analyze_candidate("TEST", frames)
        values = [item.value for item in result.score_evidence.values()]
        self.assertIn(None, values)

    def test_every_evidence_item_carries_a_reason(self):
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        for key, item in result.score_evidence.items():
            self.assertTrue(item.reason and len(item.reason) > 5,
                             f"{key} has no usable reason string")
        for key, item in result.sequence_evidence.items():
            self.assertTrue(item.reason and len(item.reason) > 5,
                             f"{key} has no usable reason string")

    def test_score_dict_is_consumable_by_scoring_module(self):
        from scoring import score_setup
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        scored = score_setup(result.score_dict())
        self.assertGreaterEqual(scored.normalized_score, 0.0)
        self.assertLessEqual(scored.normalized_score, 10.0)

    def test_rr_uses_derived_structural_levels(self):
        frames = self._frames(list(np.linspace(100, 200, 150)),
                               list(np.linspace(180, 200, 400)))
        result = analyze_candidate("TEST", frames)
        if result.stop is not None and result.target is not None:
            risk = abs(result.entry - result.stop)
            reward = abs(result.target - result.entry)
            self.assertAlmostEqual(result.reward_risk, reward / risk, places=4)


if __name__ == "__main__":
    unittest.main()


class TestRegimeFallbackWhenDailyMissing(unittest.TestCase):
    """A rate-limited daily request previously cascaded into: regime unknown
    -> no direction -> every check unevaluated -> 0.0/10. The 4H frame should
    carry the regime read instead, clearly labelled as weaker."""

    def _trending_4h(self):
        # A zigzag, not a straight line: linspace has no swing points at all,
        # so every structural check would be unevaluable for reasons that have
        # nothing to do with the regime fallback being tested here.
        return ohlc_from_closes(zigzag(n_legs=14, rising=True),
                                 start="2026-06-01", freq="4h")

    def test_missing_daily_falls_back_to_4h_regime(self):
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        self.assertNotEqual(result.regime_1d, "unknown")
        self.assertIsNotNone(result.direction)

    def test_fallback_is_disclosed_not_hidden(self):
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        joined = " ".join(result.data_problems).lower()
        self.assertIn("inferred from 4h", joined)

    def test_regime_reason_flags_the_weaker_read(self):
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        item = result.score_evidence.get("regime_alignment_1d_4h")
        self.assertIn("Daily data unavailable", item.reason)

    def test_score_is_not_zero_when_only_daily_is_missing(self):
        from scoring import score_setup
        frames = {"1d": pd.DataFrame(), "4h": self._trending_4h(),
                   "1h": self._trending_4h()}
        result = analyze_candidate("TEST", frames)
        scored = score_setup(result.score_dict())
        self.assertGreater(scored.normalized_score, 0.0)

    def test_no_fallback_when_4h_also_too_short(self):
        frames = {"1d": pd.DataFrame(), "4h": ohlc_from_closes([1, 2, 3]),
                   "1h": pd.DataFrame()}
        result = analyze_candidate("TEST", frames)
        self.assertEqual(result.regime_1d, "unknown")


class TestUniverseScan(unittest.TestCase):
    """Ranked shortlist across many instruments."""

    def _good_frames(self, rising=True):
        df = ohlc_from_closes(zigzag(n_legs=14, rising=rising),
                               start="2026-06-01", freq="4h")
        return {"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []

    def test_returns_one_result_per_instrument(self):
        from scanner import scan_universe
        insts = [("Bitcoin (BTC)", "BTC-USD", "bitcoin"),
                 ("Ethereum (ETH)", "ETH-USD", "ethereum")]
        out = scan_universe(insts, lambda k: self._good_frames())
        self.assertEqual(len(out), 2)

    def test_results_are_sorted_best_first(self):
        from scanner import scan_universe
        insts = [(f"C{i}", f"C{i}-USD", f"c{i}") for i in range(4)]

        def loader(key):
            # Alternate trending vs flat so scores genuinely differ.
            if key in ("c0", "c2"):
                return self._good_frames()
            return {"4h": ohlc_from_closes([100] * 80, start="2026-06-01", freq="4h"),
                     "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []

        out = scan_universe(insts, loader)
        scores = [r.score for r in out if r.error is None]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_failed_instruments_are_reported_not_dropped(self):
        from scanner import scan_universe
        insts = [("Good", "G-USD", "good"), ("Bad", "B-USD", "bad")]

        def loader(key):
            if key == "bad":
                raise ConnectionError("429")
            return self._good_frames()

        out = scan_universe(insts, loader)
        self.assertEqual(len(out), 2)
        errored = [r for r in out if r.error]
        self.assertEqual(len(errored), 1)
        self.assertIn("429", errored[0].error)

    def test_empty_frames_reported_as_no_data(self):
        from scanner import scan_universe
        insts = [("Empty", "E-USD", "empty")]
        out = scan_universe(insts, lambda k: ({"4h": pd.DataFrame()}, []))
        self.assertEqual(out[0].error, "No data returned.")

    def test_failures_sort_last(self):
        from scanner import scan_universe
        insts = [("Bad", "B-USD", "bad"), ("Good", "G-USD", "good")]

        def loader(key):
            if key == "bad":
                raise ValueError("boom")
            return self._good_frames()

        out = scan_universe(insts, loader)
        self.assertIsNone(out[0].error)
        self.assertIsNotNone(out[-1].error)

    def test_progress_callback_is_invoked(self):
        from scanner import scan_universe
        seen = []
        insts = [("A", "A-USD", "a"), ("B", "B-USD", "b")]
        scan_universe(insts, lambda k: self._good_frames(),
                       progress_callback=lambda i, t, l: seen.append((i, t)))
        self.assertEqual(seen[-1], (2, 2))

    def test_direction_override_is_passed_through(self):
        from scanner import scan_universe
        insts = [("A", "A-USD", "a")]
        out = scan_universe(insts, lambda k: self._good_frames(),
                             direction_override="Short")
        self.assertEqual(out[0].direction, "Short")

    def test_empty_instrument_list_is_safe(self):
        from scanner import scan_universe
        self.assertEqual(scan_universe([], lambda k: self._good_frames()), [])


def pullback_df():
    """Uptrend with clean pullbacks, so confluence zones exist below price."""
    seq = []
    for a, b in [(100, 118), (118, 104), (104, 126), (126, 108), (108, 134), (134, 112)]:
        seq += list(np.linspace(a, b, 10))[1:]
    closes = [100] + seq
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="4h", tz="UTC")
    spread = [c * 0.008 for c in closes]
    return pd.DataFrame({
        "Open": closes,
        "High": [c + s for c, s in zip(closes, spread)],
        "Low": [c - s for c, s in zip(closes, spread)],
        "Close": closes,
    }, index=idx)


class TestEntryZones(unittest.TestCase):
    """The strategy waits for price to reach a location. Entering at market
    takes whatever R:R exists at that second instead of the R:R the setup
    actually offers."""

    def test_long_zone_sits_below_current_price(self):
        from scanner import derive_entry_zone
        plan = derive_entry_zone(pullback_df(), 130.0, "Long")
        self.assertIsNotNone(plan)
        self.assertLess(plan.reference, 130.0)

    def test_short_zone_sits_above_current_price(self):
        from scanner import derive_entry_zone
        plan = derive_entry_zone(pullback_df(), 110.0, "Short")
        if plan is not None:
            self.assertGreater(plan.reference, 110.0)

    def test_price_inside_zone_is_at_zone_and_market_order(self):
        from scanner import derive_entry_zone, ENTRY_AT_ZONE
        df = pullback_df()
        plan = derive_entry_zone(df, 130.0, "Long")
        mid = plan.reference
        at = derive_entry_zone(df, mid, "Long")
        # Asking again from inside the zone should not return that same zone,
        # since it is no longer below price — but whatever it returns must be
        # self-consistent.
        if at is not None and at.status == ENTRY_AT_ZONE:
            self.assertEqual(at.order_type, "Market")

    def test_distant_zone_is_a_distant_limit_not_a_market_order(self):
        """A far zone is still the planned entry — a resting limit that may
        take time to fill — never a reason to enter at market."""
        from scanner import derive_entry_zone, ENTRY_FAR
        plan = derive_entry_zone(pullback_df(), 400.0, "Long")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.status, ENTRY_FAR)
        self.assertNotEqual(plan.order_type, "Market")
        self.assertIn("Limit", plan.order_type)

    def test_approaching_price_suggests_a_limit_order(self):
        from scanner import derive_entry_zone, ENTRY_APPROACHING
        plan = derive_entry_zone(pullback_df(), 116.0, "Long")
        self.assertEqual(plan.status, ENTRY_APPROACHING)
        self.assertEqual(plan.order_type, "Limit")

    def test_zone_has_width(self):
        from scanner import derive_entry_zone
        plan = derive_entry_zone(pullback_df(), 130.0, "Long")
        self.assertGreater(plan.zone_high, plan.zone_low)

    def test_confluence_and_sources_reported(self):
        from scanner import derive_entry_zone
        plan = derive_entry_zone(pullback_df(), 130.0, "Long")
        self.assertGreaterEqual(plan.confluence, 1)
        self.assertEqual(len(plan.sources), plan.confluence)

    def test_rationale_always_populated(self):
        from scanner import derive_entry_zone
        for px in (116.0, 130.0, 400.0):
            plan = derive_entry_zone(pullback_df(), px, "Long")
            if plan:
                self.assertTrue(plan.rationale and len(plan.rationale) > 20)

    def test_no_zone_on_flat_data(self):
        from scanner import derive_entry_zone
        flat = ohlc_from_closes([100] * 60, start="2026-01-01", freq="4h")
        self.assertIsNone(derive_entry_zone(flat, 100.0, "Long"))

    def test_invalid_direction_returns_none(self):
        from scanner import derive_entry_zone
        self.assertIsNone(derive_entry_zone(pullback_df(), 116.0, "Sideways"))

    def test_empty_frame_returns_none(self):
        from scanner import derive_entry_zone
        self.assertIsNone(derive_entry_zone(pd.DataFrame(), 100.0, "Long"))


class TestPlannedEntryInAnalysis(unittest.TestCase):
    def _frames(self):
        df = pullback_df()
        return {"1d": df, "4h": df, "1h": df}

    def test_analysis_uses_planned_entry_not_spot(self):
        result = analyze_candidate("TEST", self._frames(), direction_override="Long")
        if result.entry_plan and result.entry_is_planned:
            self.assertAlmostEqual(result.entry, result.entry_plan.reference)

    def test_entry_plan_attached_to_result(self):
        result = analyze_candidate("TEST", self._frames(), direction_override="Long")
        self.assertIsNotNone(result.entry_plan)

    def test_zone_bounds_exposed_as_key_levels(self):
        result = analyze_candidate("TEST", self._frames(), direction_override="Long")
        if result.entry_plan:
            self.assertIn("entry_zone_low", result.key_levels)
            self.assertIn("entry_zone_high", result.key_levels)

    def test_location_evidence_follows_zone_status(self):
        from scanner import ENTRY_AT_ZONE
        result = analyze_candidate("TEST", self._frames(), direction_override="Long")
        loc = result.sequence_evidence["location"]
        if result.entry_plan:
            self.assertEqual(loc.value, result.entry_plan.status == ENTRY_AT_ZONE)

    def test_missed_zone_counts_as_chasing(self):
        from scanner import ENTRY_MISSED
        result = analyze_candidate("TEST", self._frames(), direction_override="Long")
        if result.entry_plan and result.entry_plan.status == ENTRY_MISSED:
            self.assertTrue(result.score_evidence["chasing_extended_move"].value)



def extended_uptrend_df():
    """Uptrend currently extended well above its pullback levels (SOL-like)."""
    seq = [80]
    for a, b in [(80, 95), (95, 86), (86, 104), (104, 93), (93, 115), (115, 101), (101, 124)]:
        seq += list(np.linspace(a, b, 10))[1:]
    idx = pd.date_range("2026-01-01", periods=len(seq), freq="4h", tz="UTC")
    sp = [x * 0.008 for x in seq]
    return pd.DataFrame({"Open": seq, "High": [x + s for x, s in zip(seq, sp)],
                          "Low": [x - s for x, s in zip(seq, sp)], "Close": seq}, index=idx)


class TestBetterThanMarketEntries(unittest.TestCase):
    """Regression tests for the reported bug: the entry shown was the live
    price even when the plan said to wait for a zone."""

    def _analyse(self):
        df = extended_uptrend_df()
        return analyze_candidate("SOL-USD", {"1d": df, "4h": df, "1h": df, "5m": df},
                                  direction_override="Long")

    def test_entry_is_not_the_live_price(self):
        a = self._analyse()
        self.assertTrue(a.entry_is_planned)
        self.assertNotAlmostEqual(a.entry, a.current_price, places=2)

    def test_long_entry_is_below_live_price(self):
        a = self._analyse()
        self.assertLess(a.entry, a.current_price)

    def test_planned_entry_beats_market_reward_risk(self):
        from scanner import derive_levels
        a = self._analyse()
        _, _, market_rr, _ = derive_levels(extended_uptrend_df(), a.current_price,
                                            "Long", min_rr=2.0)
        self.assertGreater(a.reward_risk, market_rr)

    def test_chosen_zone_clears_minimum_rr_when_one_exists(self):
        a = self._analyse()
        self.assertGreaterEqual(a.reward_risk, 2.0)

    def test_deeper_zone_chosen_when_nearest_fails_rr(self):
        a = self._analyse()
        self.assertIn("deeper zone was chosen", a.level_reason)

    def test_alternatives_are_exposed(self):
        a = self._analyse()
        self.assertTrue(len(a.alternative_zones) >= 1)
        self.assertNotIn(a.entry_plan, a.alternative_zones)

    def test_stop_uses_a_swing_low_not_an_old_high(self):
        from technical import find_swing_points
        df = extended_uptrend_df()
        a = self._analyse()
        lows = [s.price for s in find_swing_points(df, 2, 2)
                if s.confirmed and s.kind == "low"]
        # stop = a confirmed swing low minus a small ATR buffer
        self.assertTrue(any(abs(a.stop - lo) / lo < 0.02 for lo in lows))


class TestRecentExtreme(unittest.TestCase):
    def test_long_extreme_is_the_live_high(self):
        from scanner import recent_extreme
        df = extended_uptrend_df()
        self.assertAlmostEqual(recent_extreme(df, "Long"), df["High"].iloc[-10:].max(), places=4)

    def test_fibs_anchor_on_current_leg(self):
        from scanner import _fib_levels_for
        fibs = _fib_levels_for(extended_uptrend_df(), "Long")
        self.assertTrue(fibs)
        # 0.382 of the ~101 -> ~125 leg sits near 115-116, not on the older leg
        self.assertTrue(113 < fibs["retr_0.382"] < 118)

    def test_empty_frame_safe(self):
        from scanner import recent_extreme
        self.assertIsNone(recent_extreme(pd.DataFrame(), "Long"))


class TestZoneRanking(unittest.TestCase):
    def test_confluence_can_outweigh_small_distance(self):
        from scanner import rank_zones, EntryPlan, ENTRY_APPROACHING
        near_weak = EntryPlan(99, 101, 100, ENTRY_APPROACHING, "Limit", 1.0, 1.0, 1, ["a"])
        far_strong = EntryPlan(94, 96, 95, ENTRY_APPROACHING, "Limit", 5.0, 2.5, 3,
                                ["a", "b", "c"])
        self.assertIs(rank_zones([near_weak, far_strong])[0], far_strong)

    def test_very_distant_zone_does_not_win_over_reachable_one(self):
        from scanner import rank_zones, EntryPlan, ENTRY_APPROACHING, ENTRY_FAR
        near = EntryPlan(99, 101, 100, ENTRY_APPROACHING, "Limit", 1.0, 1.0, 1, ["a"])
        remote = EntryPlan(40, 42, 41, ENTRY_FAR, "Limit (distant)", 60.0, 20.0, 4,
                            ["a", "b", "c", "d"])
        self.assertIs(rank_zones([near, remote], max_distance_atr=8.0)[0], near)

    def test_missed_zones_sort_last(self):
        from scanner import rank_zones, EntryPlan, ENTRY_APPROACHING, ENTRY_MISSED
        ok = EntryPlan(99, 101, 100, ENTRY_APPROACHING, "Limit", 1.0, 1.0, 1, ["a"])
        missed = EntryPlan(109, 111, 110, ENTRY_MISSED, "None", 0.0, 0.0, 5,
                            ["a", "b", "c", "d", "e"])
        self.assertIs(rank_zones([missed, ok])[-1], missed)


def zec_like_df():
    seq = [1100]
    for a, b in [(1100, 1300), (1300, 1210), (1210, 1420), (1420, 1330),
                 (1330, 1540), (1540, 1440), (1440, 1534)]:
        seq += list(np.linspace(a, b, 10))[1:]
    idx = pd.date_range("2026-01-01", periods=len(seq), freq="4h", tz="UTC")
    sp = [x * 0.006 for x in seq]
    return pd.DataFrame({"Open": seq, "High": [x + s for x, s in zip(seq, sp)],
                          "Low": [x - s for x, s in zip(seq, sp)], "Close": seq}, index=idx)


class TestTargetsBeyondLivePrice(unittest.TestCase):
    """Regression for the ZEC report: a long's target sat BELOW the live
    price, so the row read like a short. Levels between a planned entry and
    the live price were broken on the way up — they are support, not
    resistance, and cannot be profit targets."""

    def _analyse(self, live=1533.88):
        df = zec_like_df()
        return analyze_candidate("ZEC-USD", {"1d": df, "4h": df, "1h": df, "5m": df},
                                  live_price=live)

    def test_long_target_is_above_live_price(self):
        a = self._analyse()
        self.assertEqual(a.direction, "Long")
        self.assertGreater(a.target, a.current_price)

    def test_long_entry_below_live_stop_below_entry(self):
        a = self._analyse()
        self.assertLess(a.entry, a.current_price)
        self.assertLess(a.stop, a.entry)

    def test_live_price_override_is_used(self):
        a = self._analyse(live=1533.88)
        self.assertAlmostEqual(a.current_price, 1533.88)
        self.assertEqual(a.price_source, "live spot")

    def test_levels_are_ordered_like_a_long(self):
        a = self._analyse()
        self.assertTrue(a.stop < a.entry < a.current_price < a.target)

    def test_derive_levels_excludes_levels_below_live_for_long(self):
        from scanner import derive_levels
        df = zec_like_df()
        _, t_no_live, _, _ = derive_levels(df, 1450.0, "Long", min_rr=1.0)
        _, t_live, _, _ = derive_levels(df, 1450.0, "Long", min_rr=1.0,
                                        current_price=1533.88)
        if t_live is not None:
            self.assertGreater(t_live, 1533.88)
        if t_no_live is not None and t_live is not None:
            self.assertGreaterEqual(t_live, t_no_live)

    def test_short_target_is_below_live_price(self):
        from scanner import derive_levels
        seq = [2000]
        for a, b in [(2000, 1800), (1800, 1890), (1890, 1680), (1680, 1770),
                     (1770, 1560), (1560, 1660), (1660, 1566)]:
            seq += list(np.linspace(a, b, 10))[1:]
        idx = pd.date_range("2026-01-01", periods=len(seq), freq="4h", tz="UTC")
        sp = [x * 0.006 for x in seq]
        df = pd.DataFrame({"Open": seq, "High": [x + s for x, s in zip(seq, sp)],
                            "Low": [x - s for x, s in zip(seq, sp)], "Close": seq}, index=idx)
        _, t, _, _ = derive_levels(df, 1620.0, "Short", min_rr=1.0, current_price=1566.0)
        if t is not None:
            self.assertLess(t, 1566.0)


class TestUniverseCarriesEntryAndLivePrice(unittest.TestCase):
    def test_ranked_row_has_entry_price_and_live_flag(self):
        from scanner import scan_universe
        df = zec_like_df()
        out = scan_universe(
            [("Zcash (ZEC)", "ZEC-USD", "zcash")],
            lambda k: ({"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []),
            spot_loader=lambda k: 1533.88)
        row = out[0]
        self.assertIsNotNone(row.entry)
        self.assertTrue(row.price_is_live)
        self.assertAlmostEqual(row.price, 1533.88)

    def test_spot_failure_falls_back_to_candle_close(self):
        from scanner import scan_universe
        df = zec_like_df()
        out = scan_universe(
            [("Zcash (ZEC)", "ZEC-USD", "zcash")],
            lambda k: ({"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []),
            spot_loader=lambda k: None)
        self.assertFalse(out[0].price_is_live)

    def test_spot_loader_exception_does_not_break_scan(self):
        from scanner import scan_universe
        df = zec_like_df()

        def boom(k):
            raise ConnectionError("spot down")

        out = scan_universe(
            [("Zcash (ZEC)", "ZEC-USD", "zcash")],
            lambda k: ({"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []),
            spot_loader=boom)
        self.assertIsNone(out[0].error)
        self.assertFalse(out[0].price_is_live)
