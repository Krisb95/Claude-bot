import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trade_sim import (simulate_limit_trade, planned_reward_r,
                        WIN, LOSS, EXPIRED, PENDING, FILLED)


def bars(rows):
    """rows: list of (high, low, close)."""
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC")
    return pd.DataFrame({"Open": [r[2] for r in rows], "High": [r[0] for r in rows],
                          "Low": [r[1] for r in rows], "Close": [r[2] for r in rows]},
                         index=idx)


class TestFills(unittest.TestCase):
    def test_long_limit_fills_only_when_price_reaches_entry(self):
        b = bars([(105, 101, 103), (104, 99, 100), (112, 100, 111)])
        r = simulate_limit_trade(b, "Long", entry=100, stop=95, target=110, expiry_bars=10)
        self.assertEqual(r.bars_to_fill, 1)

    def test_unfilled_order_expires(self):
        b = bars([(110, 105, 108)] * 5)
        r = simulate_limit_trade(b, "Long", entry=100, stop=95, target=120, expiry_bars=5)
        self.assertEqual(r.status, EXPIRED)
        self.assertIsNone(r.r_result)

    def test_unfilled_inside_window_is_pending(self):
        b = bars([(110, 105, 108)] * 3)
        r = simulate_limit_trade(b, "Long", entry=100, stop=95, target=120, expiry_bars=10)
        self.assertEqual(r.status, PENDING)

    def test_no_bars_is_pending(self):
        r = simulate_limit_trade(pd.DataFrame(), "Long", 100, 95, 110, 10)
        self.assertEqual(r.status, PENDING)


class TestOutcomes(unittest.TestCase):
    def test_long_win_pays_planned_r(self):
        b = bars([(101, 99, 100), (111, 100, 110)])
        r = simulate_limit_trade(b, "Long", entry=100, stop=95, target=110, expiry_bars=10)
        self.assertEqual(r.status, WIN)
        self.assertAlmostEqual(r.r_result, 2.0)

    def test_long_loss_is_minus_one_r(self):
        b = bars([(101, 99, 100), (100, 94, 95)])
        r = simulate_limit_trade(b, "Long", entry=100, stop=95, target=110, expiry_bars=10)
        self.assertEqual(r.status, LOSS)
        self.assertAlmostEqual(r.r_result, -1.0)

    def test_short_win(self):
        b = bars([(101, 99, 100), (100, 89, 90)])
        r = simulate_limit_trade(b, "Short", entry=100, stop=105, target=90, expiry_bars=10)
        self.assertEqual(r.status, WIN)
        self.assertAlmostEqual(r.r_result, 2.0)

    def test_short_loss(self):
        b = bars([(101, 99, 100), (106, 100, 105)])
        r = simulate_limit_trade(b, "Short", entry=100, stop=105, target=90, expiry_bars=10)
        self.assertEqual(r.status, LOSS)

    def test_open_at_end_of_data_marked_to_market(self):
        b = bars([(101, 99, 100), (104, 101, 103)])
        r = simulate_limit_trade(b, "Long", entry=100, stop=95, target=120, expiry_bars=10)
        self.assertEqual(r.status, FILLED)
        self.assertAlmostEqual(r.last_r, 0.6)   # (103-100)/5

    def test_fees_reduce_r(self):
        b = bars([(101, 99, 100), (111, 100, 110)])
        r = simulate_limit_trade(b, "Long", 100, 95, 110, 10, fee_r=0.1)
        self.assertAlmostEqual(r.r_result, 1.9)


class TestConservativeAmbiguity(unittest.TestCase):
    """With OHLC only, intrabar order is unknown — the worse outcome is assumed."""

    def test_fill_bar_that_also_hits_stop_is_a_loss(self):
        b = bars([(101, 94, 96)])     # touches entry 100 AND stop 95 in one bar
        r = simulate_limit_trade(b, "Long", 100, 95, 110, 10)
        self.assertEqual(r.status, LOSS)

    def test_target_on_fill_bar_is_not_counted(self):
        # Bar spans entry and target: price may have hit target BEFORE dipping
        # to entry, meaning no fill — so a win must not be credited here.
        b = bars([(111, 99, 105)])
        r = simulate_limit_trade(b, "Long", 100, 95, 110, 10)
        self.assertNotEqual(r.status, WIN)

    def test_bar_hitting_both_stop_and_target_resolves_to_stop(self):
        b = bars([(101, 99, 100), (112, 94, 100)])
        r = simulate_limit_trade(b, "Long", 100, 95, 110, 10)
        self.assertEqual(r.status, LOSS)


class TestValidation(unittest.TestCase):
    def test_bad_direction_raises(self):
        with self.assertRaises(ValueError):
            simulate_limit_trade(bars([(1, 1, 1)]), "Sideways", 100, 95, 110, 10)

    def test_zero_risk_raises(self):
        with self.assertRaises(ValueError):
            simulate_limit_trade(bars([(101, 99, 100)]), "Long", 100, 100, 110, 10)

    def test_planned_reward_r(self):
        self.assertAlmostEqual(planned_reward_r("Long", 100, 95, 110), 2.0)
        self.assertIsNone(planned_reward_r("Long", 100, 100, 110))


if __name__ == "__main__":
    unittest.main()


class TestStopManagement(unittest.TestCase):
    """Breakeven and trailing stops, modelled so they can be compared."""

    def test_unchanged_when_no_management_is_asked_for(self):
        b = bars([(101, 99, 100), (100, 94, 95)])
        plain = simulate_limit_trade(b, "Long", 100, 95, 110, 10)
        self.assertEqual(plain.status, LOSS)
        self.assertAlmostEqual(plain.r_result, -1.0)

    def test_breakeven_turns_a_giveback_into_a_scratch(self):
        # fills, runs to +1R, then falls back through entry
        b = bars([(101, 99, 100), (106, 100, 105), (105, 94, 95)])
        managed = simulate_limit_trade(b, "Long", 100, 95, 120, 10, breakeven_at_r=1.0)
        self.assertAlmostEqual(managed.r_result, 0.0, places=6)
        plain = simulate_limit_trade(b, "Long", 100, 95, 120, 10)
        self.assertAlmostEqual(plain.r_result, -1.0)

    def test_breakeven_not_applied_before_the_threshold(self):
        b = bars([(101, 99, 100), (102, 100, 101), (101, 94, 95)])
        r = simulate_limit_trade(b, "Long", 100, 95, 120, 10, breakeven_at_r=1.0)
        self.assertAlmostEqual(r.r_result, -1.0)

    def test_trailing_locks_in_profit_instead_of_a_full_loss(self):
        b = bars([(101, 99, 100), (115, 100, 114), (114, 104, 105)])
        r = simulate_limit_trade(b, "Long", 100, 95, 200, 10,
                                 trail_atr_mult=2.0, atr_value=2.0)
        self.assertEqual(r.status, WIN)
        self.assertGreater(r.r_result, 0)

    def test_trailing_can_cut_a_winner_short(self):
        """The real cost: a trail can stop you out before the target."""
        b = bars([(101, 99, 100), (112, 100, 111), (111, 105, 106), (130, 106, 129)])
        trailed = simulate_limit_trade(b, "Long", 100, 95, 130, 10,
                                       trail_atr_mult=1.0, atr_value=2.0)
        plain = simulate_limit_trade(b, "Long", 100, 95, 130, 10)
        self.assertEqual(plain.status, WIN)
        self.assertAlmostEqual(plain.r_result, 6.0)      # full target
        self.assertLess(trailed.r_result, plain.r_result)

    def test_a_trailed_stop_never_widens(self):
        b = bars([(101, 99, 100), (110, 100, 109), (109, 101, 102), (104, 100.5, 103)])
        r = simulate_limit_trade(b, "Long", 100, 95, 200, 10,
                                 trail_atr_mult=2.0, atr_value=2.0)
        if r.exit_time is not None:
            self.assertGreater(r.r_result, -1.0, "a trailed stop cannot be worse than -1R")

    def test_short_side_mirrors(self):
        b = bars([(101, 99, 100), (100, 94, 95), (106, 95, 105)])
        r = simulate_limit_trade(b, "Short", 100, 105, 80, 10, breakeven_at_r=1.0)
        self.assertAlmostEqual(r.r_result, 0.0, places=6)
