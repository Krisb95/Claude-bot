import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from formatting import format_price, format_signed, decimals_for


class TestNoScientificNotation(unittest.TestCase):
    """The bug: PEPE at ~0.000003991 rendered as '3.991e-06'."""

    def test_tiny_price_is_plain_decimal(self):
        out = format_price(0.000003991)
        self.assertNotIn("e", out.lower())
        self.assertTrue(out.startswith("0.0000039"))

    def test_extremely_tiny_price_still_plain(self):
        out = format_price(0.00000000123)
        self.assertNotIn("e", out.lower())

    def test_tiny_values_remain_distinguishable(self):
        """A stop and target near PEPE's price must not collapse to the
        same string — that would make the table useless."""
        price = format_price(0.000003991)
        stop = format_price(0.00000393014)
        target = format_price(0.000004203)
        self.assertNotEqual(price, stop)
        self.assertNotEqual(price, target)
        self.assertNotEqual(stop, target)


class TestNormalMagnitudes(unittest.TestCase):
    def test_large_price_uses_thousands_separator(self):
        self.assertEqual(format_price(81544.156), "81,544.16")

    def test_mid_price(self):
        self.assertEqual(format_price(3.4382), "3.4382")

    def test_price_just_above_one(self):
        self.assertEqual(format_price(1.5), "1.5")

    def test_trailing_zeros_trimmed(self):
        self.assertEqual(format_price(2.5000), "2.5")

    def test_whole_number_has_no_dangling_point(self):
        out = format_price(100.0)
        self.assertFalse(out.endswith("."))
        self.assertEqual(out, "100")

    def test_sub_one_price(self):
        out = format_price(0.5)
        self.assertNotIn("e", out.lower())
        self.assertEqual(out, "0.5")


class TestEdgeCases(unittest.TestCase):
    def test_none_renders_as_dash(self):
        self.assertEqual(format_price(None), "—")

    def test_nan_renders_as_dash(self):
        self.assertEqual(format_price(float("nan")), "—")

    def test_infinity_renders_as_dash(self):
        self.assertEqual(format_price(float("inf")), "—")

    def test_zero(self):
        self.assertEqual(format_price(0), "0")

    def test_negative_value(self):
        out = format_price(-3.4382)
        self.assertTrue(out.startswith("-"))


class TestSignedFormatting(unittest.TestCase):
    def test_positive_gets_plus(self):
        self.assertTrue(format_signed(1.25).startswith("+"))

    def test_negative_gets_minus(self):
        self.assertTrue(format_signed(-1.25).startswith("-"))

    def test_none_is_dash(self):
        self.assertEqual(format_signed(None), "—")


class TestDecimalsFor(unittest.TestCase):
    def test_more_decimals_for_smaller_numbers(self):
        self.assertGreater(decimals_for(0.000001), decimals_for(0.1))

    def test_large_numbers_need_no_decimals(self):
        self.assertEqual(decimals_for(123456), 0)

    def test_capped_to_avoid_absurd_precision(self):
        self.assertLessEqual(decimals_for(1e-30), 12)

    def test_zero_is_safe(self):
        self.assertEqual(decimals_for(0), 2)


if __name__ == "__main__":
    unittest.main()


class TestRewardRiskFormatting(unittest.TestCase):
    def test_written_as_three_to_one(self):
        from formatting import format_rr
        self.assertEqual(format_rr(3.0), "3:1")

    def test_whole_numbers_lose_the_decimals(self):
        from formatting import format_rr
        self.assertEqual(format_rr(2.0), "2:1")
        self.assertEqual(format_rr(4.0), "4:1")

    def test_fractional_values_keep_two_places(self):
        from formatting import format_rr
        self.assertEqual(format_rr(2.47), "2.47:1")

    def test_missing_and_invalid(self):
        from formatting import format_rr
        self.assertEqual(format_rr(None), "—")
        self.assertEqual(format_rr(0), "—")
        self.assertEqual(format_rr(float("nan")), "—")
