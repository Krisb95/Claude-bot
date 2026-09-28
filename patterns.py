"""
Pattern recognition: candlestick shapes and simple swing patterns.

WHAT THESE ARE WORTH: candlestick patterns are precise, objective descriptions
of what buyers and sellers did in a few bars. They are NOT predictions. Studies
of them in isolation generally find little or no edge; what they may add is
confirmation at a level that already matters for another reason — which is
exactly how this strategy would use them, at a 5m support during a 4H trend.

So nothing here claims a pattern means price will rise. Each detection says
what happened and how it is conventionally read, and every detected pattern is
recorded with the setup so the learner can find out whether it makes any
difference to YOUR results. That measurement is the point; the detection is
just the input.

Definitions are deliberately strict. A loose definition finds a pattern in
almost every candle, which would make the whole exercise meaningless.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"

DOJI_BODY_MAX = 0.10        # body this small relative to range counts as a doji
PIN_WICK_MIN = 2.0          # rejection wick at least this many times the body
PIN_BODY_MAX = 0.35         # and the body no more than this share of the range
PIN_OPPOSITE_WICK_MAX = 0.25  # the wick on the other side, as a share of the range
STRONG_BODY_MIN = 0.55      # a "decisive" candle's body, as a share of its range


@dataclass
class Pattern:
    name: str
    bias: str                # bullish / bearish / neutral
    description: str         # what actually happened
    convention: str          # how it is usually read — not a prediction
    bars: int                # how many candles it spans


def _parts(row) -> Dict[str, float]:
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    rng = h - l
    return {"o": o, "h": h, "l": l, "c": c, "rng": rng,
            "body": abs(c - o),
            "body_ratio": (abs(c - o) / rng) if rng > 0 else 0.0,
            "upper": h - max(o, c), "lower": min(o, c) - l,
            "up": c > o, "down": c < o}


def detect(df: pd.DataFrame) -> List[Pattern]:
    """Patterns completed by the LAST candle of the frame.

    Pass closed candles only — a pattern that hasn't finished forming isn't a
    pattern yet, and will often look completely different by the close.
    """
    out: List[Pattern] = []
    if df is None or len(df) < 2:
        return out
    cur = _parts(df.iloc[-1])
    prev = _parts(df.iloc[-2])

    # A pin bar is measured against the candle's RANGE, not its body: hammers
    # have small bodies by definition, so comparing the upper wick to the body
    # would rule out the very candles being looked for.
    hammer = (cur["rng"] > 0 and cur["body"] > 0
              and cur["lower"] >= PIN_WICK_MIN * cur["body"]
              and cur["body_ratio"] <= PIN_BODY_MAX
              and cur["upper"] <= PIN_OPPOSITE_WICK_MAX * cur["rng"])
    star = (cur["rng"] > 0 and cur["body"] > 0
            and cur["upper"] >= PIN_WICK_MIN * cur["body"]
            and cur["body_ratio"] <= PIN_BODY_MAX
            and cur["lower"] <= PIN_OPPOSITE_WICK_MAX * cur["rng"])

    # --- single candle -------------------------------------------------
    # A pin bar also has a tiny body, so it would match "doji" too. The pin
    # bar is the more specific and more useful reading, so it wins.
    if cur["rng"] > 0 and cur["body_ratio"] <= DOJI_BODY_MAX and not (hammer or star):
        out.append(Pattern(
            "Doji", NEUTRAL,
            "The candle opened and closed at almost the same price after moving in both "
            "directions.",
            "Read as indecision — buyers and sellers finished level.", 1))

    if hammer:
        out.append(Pattern(
            "Hammer / bullish pin bar", BULLISH,
            "Price was pushed well below the open and then bought back before the close, "
            "leaving a long lower wick.",
            "Read as sellers being rejected at that low.", 1))

    if star:
        out.append(Pattern(
            "Shooting star / bearish pin bar", BEARISH,
            "Price was pushed well above the open and then sold back before the close, "
            "leaving a long upper wick.",
            "Read as buyers being rejected at that high.", 1))

    if cur["body_ratio"] >= STRONG_BODY_MIN:
        out.append(Pattern(
            "Strong " + ("bullish" if cur["up"] else "bearish") + " candle",
            BULLISH if cur["up"] else BEARISH,
            f"The body is {cur['body_ratio'] * 100:.0f}% of the candle's range — one side "
            f"controlled the whole period.",
            "Read as conviction rather than drift.", 1))

    # --- two candles ---------------------------------------------------
    if (prev["down"] and cur["up"]
            and cur["c"] >= prev["o"] and cur["o"] <= prev["c"]
            and cur["body"] > prev["body"]):
        out.append(Pattern(
            "Bullish engulfing", BULLISH,
            "This candle's body completely covers the previous down candle's body — the "
            "whole of the prior fall was bought back.",
            "Read as buyers taking control from sellers.", 2))

    if (prev["up"] and cur["down"]
            and cur["c"] <= prev["o"] and cur["o"] >= prev["c"]
            and cur["body"] > prev["body"]):
        out.append(Pattern(
            "Bearish engulfing", BEARISH,
            "This candle's body completely covers the previous up candle's body — the whole "
            "of the prior rise was sold off.",
            "Read as sellers taking control from buyers.", 2))

    if cur["h"] < prev["h"] and cur["l"] > prev["l"]:
        out.append(Pattern(
            "Inside bar", NEUTRAL,
            "The whole candle fits inside the previous one's range — the market went "
            "quiet.",
            "Read as a pause; often precedes a move, direction unknown.", 2))

    # --- three candles -------------------------------------------------
    if len(df) >= 3:
        first = _parts(df.iloc[-3])
        mid = _parts(df.iloc[-2])
        midpoint = (first["o"] + first["c"]) / 2
        if (first["down"] and first["body_ratio"] >= 0.4
                and mid["body_ratio"] <= 0.35
                and cur["up"] and cur["c"] > midpoint):
            out.append(Pattern(
                "Morning star", BULLISH,
                "A strong fall, then a pause, then a rise closing back above the midpoint "
                "of the fall.",
                "Read as a three-bar reversal from selling to buying.", 3))
        if (first["up"] and first["body_ratio"] >= 0.4
                and mid["body_ratio"] <= 0.35
                and cur["down"] and cur["c"] < midpoint):
            out.append(Pattern(
                "Evening star", BEARISH,
                "A strong rise, then a pause, then a fall closing back below the midpoint "
                "of the rise.",
                "Read as a three-bar reversal from buying to selling.", 3))
    return out


def bias_of(patterns: List[Pattern]) -> str:
    """Overall lean of the detected patterns."""
    bull = sum(1 for p in patterns if p.bias == BULLISH)
    bear = sum(1 for p in patterns if p.bias == BEARISH)
    if bull > bear:
        return BULLISH
    if bear > bull:
        return BEARISH
    return NEUTRAL


def confirms(patterns: List[Pattern], direction: str) -> bool:
    """Does anything here point the same way as the trade?"""
    want = BULLISH if direction == "Long" else BEARISH
    return any(p.bias == want for p in patterns)


def contradicts(patterns: List[Pattern], direction: str) -> bool:
    """Is anything here pointing the opposite way?"""
    against = BEARISH if direction == "Long" else BULLISH
    return any(p.bias == against for p in patterns)


def summarise(patterns: List[Pattern]) -> Optional[str]:
    return ", ".join(p.name for p in patterns) if patterns else None


# ---------------------------------------------------------------------
# Swing patterns
# ---------------------------------------------------------------------

def double_bottom(swings, tolerance: float = 0.01) -> Optional[Pattern]:
    """Two lows at roughly the same price with a higher low between them."""
    lows = [s for s in swings if s.kind == "low"]
    if len(lows) < 2:
        return None
    a, b = lows[-2], lows[-1]
    if a.price <= 0:
        return None
    if abs(b.price - a.price) / a.price <= tolerance:
        return Pattern(
            "Double bottom", BULLISH,
            f"Price found support twice at about {a.price:.6g} without breaking it.",
            "Read as a level buyers defended more than once.", 0)
    return None


def double_top(swings, tolerance: float = 0.01) -> Optional[Pattern]:
    highs = [s for s in swings if s.kind == "high"]
    if len(highs) < 2:
        return None
    a, b = highs[-2], highs[-1]
    if a.price <= 0:
        return None
    if abs(b.price - a.price) / a.price <= tolerance:
        return Pattern(
            "Double top", BEARISH,
            f"Price was rejected twice at about {a.price:.6g} without breaking through.",
            "Read as a level sellers defended more than once.", 0)
    return None


def swing_patterns(df: pd.DataFrame, left: int = 3, right: int = 3,
                   tolerance: float = 0.01) -> List[Pattern]:
    from technical import find_swing_points
    if df is None or len(df) < left + right + 3:
        return []
    swings = [s for s in find_swing_points(df, left, right) if s.confirmed]
    found = [p for p in (double_bottom(swings, tolerance), double_top(swings, tolerance))
             if p is not None]
    return found
