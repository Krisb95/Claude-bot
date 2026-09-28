"""
Check that candles are what they claim to be.

A data source can hand back a frame that is labelled "5m" but holds four-hour
or daily candles, or that ends days ago. Nothing downstream can tell: the
strategy reads the last few hundred rows, finds a "5m support" that is really
a low from a month back, and produces a plan with prices nowhere near the
market. That is exactly how a ZEC plan came to suggest buying at 481 while the
coin traded at 1,662.

So every frame is checked before use:

  * SPACING — the gap between candles must match the timeframe asked for. A
    frame of 4-hour candles labelled 5m fails here.
  * FRESHNESS — the newest candle must be recent. Old history fails here.
  * SIZE — enough candles to read structure from.

A frame that fails is not "slightly off", it is the wrong data, and the caller
moves to the next source rather than building a plan on it.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
           "4h": 14400, "1d": 86400, "1w": 604800}

# How far the median gap may drift from the nominal one before the frame is
# judged to be a different timeframe. Real feeds have gaps (halts, outages),
# so this is deliberately loose — it is meant to catch 4h-labelled-5m, not
# a missing candle.
SPACING_TOLERANCE = 0.5

# How stale the newest candle may be, as a multiple of the timeframe. Six 5m
# candles is half an hour; six daily candles is most of a week.
MAX_AGE_MULTIPLE = 6.0
MIN_AGE_SECONDS = 900          # never complain about less than 15 minutes


@dataclass
class FrameCheck:
    ok: bool
    reason: str = ""
    bars: int = 0
    median_gap_seconds: Optional[float] = None
    age_seconds: Optional[float] = None

    @property
    def detail(self) -> str:
        bits = [f"{self.bars} bars"]
        if self.median_gap_seconds:
            bits.append(f"{self.median_gap_seconds / 60:.0f}m apart")
        if self.age_seconds is not None:
            bits.append(f"newest {self.age_seconds / 3600:.1f}h old")
        return ", ".join(bits)


def median_gap(df: pd.DataFrame) -> Optional[float]:
    """Typical seconds between candles."""
    if df is None or len(df) < 3:
        return None
    diffs = pd.Series(df.index).diff().dropna()
    if diffs.empty:
        return None
    return float(diffs.median().total_seconds())


def check(df: pd.DataFrame, timeframe: str, min_bars: int = 30,
          now: Optional[pd.Timestamp] = None) -> FrameCheck:
    """Is this frame really `timeframe` candles, recent enough to trade from?"""
    expected = SECONDS.get(timeframe)
    if df is None or df.empty:
        return FrameCheck(False, "No candles returned.")
    bars = len(df)
    if bars < min_bars:
        return FrameCheck(False, f"Only {bars} candles — too few to read structure.", bars)
    if expected is None:
        return FrameCheck(True, "", bars)

    gap = median_gap(df)
    if gap:
        low, high = expected * (1 - SPACING_TOLERANCE), expected * (1 + SPACING_TOLERANCE)
        if not (low <= gap <= high):
            return FrameCheck(
                False,
                f"Candles are {gap / 60:.0f} minutes apart but {timeframe} was requested "
                f"({expected / 60:.0f} minutes). This is the wrong timeframe, so any level "
                f"read from it would be wrong.",
                bars, gap)

    now = now or pd.Timestamp.now(tz="UTC")
    last = df.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    age = (now - last).total_seconds()
    allowed = max(expected * MAX_AGE_MULTIPLE, MIN_AGE_SECONDS)
    if age > allowed:
        return FrameCheck(
            False,
            f"The newest candle is {age / 3600:.1f} hours old — stale for {timeframe} data. "
            f"Levels read from it would describe a market that has moved on.",
            bars, gap, age)
    return FrameCheck(True, "", bars, gap, age)


def first_valid(candidates, timeframe: str, min_bars: int = 30,
                now: Optional[pd.Timestamp] = None) -> Tuple[Optional[pd.DataFrame],
                                                              Optional[str], str]:
    """Walk (name, frame) pairs and return the first that passes.

    Returns (frame, source_name, note). The note lists why each rejected source
    was rejected, so a silent fallback can still be explained.
    """
    problems = []
    for name, df in candidates:
        result = check(df, timeframe, min_bars=min_bars, now=now)
        if result.ok:
            return df, name, "; ".join(problems)
        problems.append(f"{name}: {result.reason}")
    return None, None, "; ".join(problems)
