"""
Human-readable price formatting.

Python's %g switches to scientific notation for small magnitudes, so a coin
like PEPE (~0.000003991) rendered as "3.991e-06" — technically correct and
useless at a glance when you are comparing it to a stop and a target.

These helpers keep plain decimal notation at every magnitude, showing enough
significant digits that a stop and a target remain visibly different from the
entry, and trimming pointless trailing zeros.
"""

import math
from typing import Optional

SIGNIFICANT_DIGITS = 5
MAX_DECIMALS = 12


def decimals_for(value: float, significant: int = SIGNIFICANT_DIGITS) -> int:
    """How many decimal places are needed to show `significant` digits."""
    v = abs(value)
    if v == 0 or not math.isfinite(v):
        return 2
    exponent = math.floor(math.log10(v))
    if exponent >= significant - 1:
        return 0          # large numbers need no decimals for this many digits
    return min(MAX_DECIMALS, significant - 1 - exponent)


def format_price(value: Optional[float], significant: int = SIGNIFICANT_DIGITS) -> str:
    """Format a price in plain decimal notation, never scientific.

        81544.156   -> "81,544.16"
        3.4382      -> "3.4382"
        0.000003991 -> "0.0000039910"
        None        -> "—"
    """
    if value is None or not math.isfinite(value):
        return "—"

    v = abs(value)
    if v >= 1000:
        return f"{value:,.2f}"
    if v >= 1:
        text = f"{value:,.4f}"
    else:
        text = f"{value:,.{decimals_for(value, significant)}f}"

    # Trim trailing zeros, but never leave a bare trailing decimal point.
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_signed(value: Optional[float], significant: int = SIGNIFICANT_DIGITS) -> str:
    """Same, with an explicit + or - sign."""
    if value is None or not math.isfinite(value):
        return "—"
    body = format_price(abs(value), significant)
    return f"{'-' if value < 0 else '+'}{body}"


def format_rr(value: Optional[float]) -> str:
    """Reward:risk written as "3:1" — reward first, risk as 1.

    Whole numbers lose the decimals ("3:1", not "3.00:1"); anything else keeps
    two ("2.47:1").
    """
    if value is None or not math.isfinite(value) or value <= 0:
        return "—"
    if abs(value - round(value)) < 0.005:
        return f"{round(value):g}:1"
    return f"{value:.2f}:1"
