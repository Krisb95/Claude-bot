"""
Review an open trade: hold it, tighten the stop, or close it early.

WHAT IT CHECKS
  1. Thesis  — is the reason for entering still true? For the Trend Retrace
               strategy that's the 4H rule. If the 4H candles now trend the
               OTHER way, the thesis is broken.
  2. Progress — how far the trade has moved in R, the best it reached (MFE)
               and the worst (MAE) since entry.
  3. Time    — how long it has been open. A trade that has gone nowhere for
               far longer than trades normally take is "stale".
  4. 1H      — whether the last closed 1H candle is working against the trade.
  5. Structure — whether a newer confirmed swing allows a TIGHTER stop.

WHAT IT NEVER DOES
  It never suggests moving a stop further away. A tighter stop is only offered
  when it sits beyond a confirmed swing and still leaves room below price
  (above, for shorts); otherwise the advice is to hold or close.

A WARNING BUILT INTO THE ADVICE
  In a 3:1 strategy, many eventual winners spend hours going nowhere first.
  Closing every sideways trade trims small losses but can cut off the large
  wins that pay for everything. Whether closing stale trades HELPS this
  strategy is an empirical question — the backtest measures it — so a "close
  early" verdict for staleness alone is presented as a judgement call, while
  a broken thesis is presented as a clear signal.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd

from technical import find_swing_points
from formatting import format_price as fp

HOLD = "Hold"
TIGHTEN = "Tighten the stop"
CLOSE = "Consider closing early"
CLOSE_THESIS = "Close — the reason you entered no longer holds"

DEFAULT_STALE_HOURS = 12.0
STALE_MAX_PROGRESS_R = 0.5      # below this, a long-open trade counts as going nowhere


@dataclass
class ReviewResult:
    verdict: str
    reasons: List[str]
    current_r: Optional[float]
    mfe_r: Optional[float]           # best move in your favour since entry, in R
    mae_r: Optional[float]           # worst move against you, in R (negative)
    hours_open: float
    thesis: str                      # "intact" / "weakening" / "broken" / "unknown"
    stale: bool
    suggested_stop: Optional[float] = None
    close_now_r: Optional[float] = None
    stop_r: Optional[float] = None
    target_r: Optional[float] = None
    actions: List[str] = field(default_factory=list)


def _r(direction: str, entry: float, stop: float, price: float) -> Optional[float]:
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return ((price - entry) if direction == "Long" else (entry - price)) / risk


def excursions(direction: str, entry: float, stop: float, candles_since_entry: pd.DataFrame
               ) -> Tuple[Optional[float], Optional[float]]:
    """(MFE, MAE) in R from candles printed after entry."""
    if candles_since_entry is None or candles_since_entry.empty:
        return None, None
    hi = float(candles_since_entry["High"].max())
    lo = float(candles_since_entry["Low"].min())
    if direction == "Long":
        return _r(direction, entry, stop, hi), _r(direction, entry, stop, lo)
    return _r(direction, entry, stop, lo), _r(direction, entry, stop, hi)


def tighter_stop(direction: str, current_stop: float, price: float,
                 candles: pd.DataFrame, buffer: float) -> Optional[float]:
    """A tighter stop placed just beyond the most recent confirmed swing that
    sits between the current stop and price. None if no such swing exists.

    'Tighter' is strict: for a long the new stop must be ABOVE the current one
    and at least `buffer` below price; mirrored for a short. It can never widen.
    """
    if candles is None or len(candles) < 7:
        return None
    swings = [s for s in find_swing_points(candles, 3, 3) if s.confirmed]
    if direction == "Long":
        valid = [s for s in swings if s.kind == "low"
                 and current_stop < s.price - buffer < price - buffer]
        if not valid:
            return None
        return max(valid, key=lambda s: s.index).price - buffer
    highs = [s for s in swings if s.kind == "high"]
    valid = [s for s in highs if price + buffer < s.price + buffer < current_stop]
    if not valid:
        return None
    latest = max(valid, key=lambda s: s.index)
    return latest.price + buffer


def review(direction: str, entry: float, stop: float, target: float,
           entry_time: pd.Timestamp, price: float, now: pd.Timestamp,
           df_4h: pd.DataFrame, df_1h: pd.DataFrame, candles_since_entry: pd.DataFrame,
           structure_candles: pd.DataFrame, atr_value: Optional[float],
           trend_candles: int = 2, stale_hours: float = DEFAULT_STALE_HOURS) -> ReviewResult:
    """Review one open trade. All candle inputs must be CLOSED candles."""
    from trend_retrace import four_hour_trend, one_hour_confirms

    reasons: List[str] = []
    actions: List[str] = []
    hours = max(0.0, (now - entry_time).total_seconds() / 3600)
    cur = _r(direction, entry, stop, price)
    mfe, mae = excursions(direction, entry, stop, candles_since_entry)
    stop_r = _r(direction, entry, stop, stop)       # -1 unless stop already moved
    target_r = _r(direction, entry, stop, target)

    # 1. Thesis: the 4H rule.
    trend, trend_reason = four_hour_trend(df_4h, trend_candles)
    if trend is None and df_4h is not None and len(df_4h) >= trend_candles + 1:
        thesis = "weakening"
        reasons.append("The 4H candles are no longer making consistent higher highs and "
                       "lows (lower, for a short) — the trend has paused, not reversed.")
    elif trend is None:
        thesis = "unknown"
        reasons.append("Not enough 4H candles to re-check the trend.")
    elif trend == direction:
        thesis = "intact"
        reasons.append("The 4H trend still runs in your direction — the reason you entered "
                       "still holds.")
    else:
        thesis = "broken"
        reasons.append(f"The 4H candles now trend the other way ({trend.lower()}). The reason "
                       f"you entered has reversed.")

    # 2. 1H candle.
    h1_against = False
    if df_1h is not None and not df_1h.empty:
        with_trade = one_hour_confirms(df_1h, direction).passed
        against = one_hour_confirms(df_1h, "Short" if direction == "Long" else "Long").passed
        h1_against = against
        if with_trade:
            reasons.append("The last closed 1H candle moved in your favour.")
        elif against:
            reasons.append("The last closed 1H candle moved against you.")

    # 3. Progress and time.
    stale = (hours >= stale_hours and cur is not None and abs(cur) < STALE_MAX_PROGRESS_R
             and (mfe is None or mfe < 1.0))
    if cur is not None:
        reasons.append(f"Open {hours:.1f}h. Now at {cur:+.2f}R"
                       + (f"; best {mfe:+.2f}R, worst {mae:+.2f}R since entry."
                          if mfe is not None else "."))
    if stale:
        reasons.append(f"It has gone nowhere for {hours:.0f}h — longer than the "
                       f"{stale_hours:.0f}h this review treats as normal.")

    # 4. Possible tighter stop.
    new_stop = None
    if atr_value and atr_value > 0:
        new_stop = tighter_stop(direction, stop, price, structure_candles, 0.5 * atr_value)

    gave_back = mfe is not None and mfe >= 1.0 and cur is not None and cur < 0.5
    in_profit = cur is not None and cur > 0
    breakeven_ok = (entry < price if direction == "Long" else entry > price)

    # --- verdict ------------------------------------------------------
    if thesis == "broken":
        verdict = CLOSE_THESIS
        actions.append(f"Close at about {fp(price)} ({cur:+.2f}R) rather than waiting for "
                       f"the stop ({stop_r:+.2f}R)." if cur is not None else "Close the trade.")
    elif gave_back and breakeven_ok and ((direction == "Long" and entry > stop)
                                         or (direction == "Short" and entry < stop)):
        verdict = TIGHTEN
        new_stop = new_stop if new_stop is not None and (
            (direction == "Long" and new_stop > entry) or
            (direction == "Short" and new_stop < entry)) else entry
        actions.append(f"It reached {mfe:+.2f}R then gave most of it back. Move the stop to "
                       f"{fp(new_stop)} so a winner can't turn into a full loss.")
    elif stale and h1_against:
        verdict = CLOSE
        actions.append(f"Closing now costs {cur:+.2f}R instead of risking {stop_r:+.2f}R — "
                       f"but see the caution below before acting on staleness alone.")
    elif (stale or thesis == "weakening") and new_stop is not None:
        verdict = TIGHTEN
        actions.append(f"Tighten the stop to {fp(new_stop)}, just beyond the latest swing. "
                       f"That cuts the risk while giving the trade more time.")
    else:
        verdict = HOLD
        near_target = (in_profit and target_r and cur is not None
                       and cur >= target_r * 0.66)
        if near_target and new_stop is not None:
            # Optional trailing stop — offered, but the verdict stays Hold.
            actions.append(f"Close to target. You could trail the stop to {fp(new_stop)} to "
                           f"lock in part of the gain.")
        else:
            actions.append("Nothing in the review calls for action — let the plan run.")
            new_stop = None

    if verdict == CLOSE:
        actions.append("Caution: in a 3:1 strategy many winners go sideways first. Check the "
                       "backtest's 'close stale trades' comparison before making this a habit.")

    return ReviewResult(verdict=verdict, reasons=reasons, current_r=cur, mfe_r=mfe,
                        mae_r=mae, hours_open=hours, thesis=thesis, stale=stale,
                        suggested_stop=new_stop if verdict in (TIGHTEN, HOLD) else None,
                        close_now_r=cur, stop_r=stop_r, target_r=target_r, actions=actions)


# ---------------------------------------------------------------------
# Re-checking a setup you saved earlier
# ---------------------------------------------------------------------

STILL_VALID = "Still valid"
ENTRY_PASSED = "Entry already passed"
TREND_GONE = "Trend no longer supports it"
LEVELS_STALE = "Too old to trust"
INVALIDATED = "Invalidated — price is past the stop"


@dataclass
class ViabilityResult:
    verdict: str
    reasons: List[str]
    hours_old: float
    distance_to_entry_pct: Optional[float]
    price: float


def setup_still_viable(direction: str, entry: float, stop: float, created: pd.Timestamp,
                       price: float, now: pd.Timestamp, df_4h: pd.DataFrame,
                       trend_candles: int = 2,
                       max_age_hours: float = 48.0) -> ViabilityResult:
    """Is a setup saved earlier still worth waiting for?

    A plan made yesterday describes a chart that no longer exists. This
    re-checks the parts that can go stale: the 4H trend it was built on, and
    whether price has since blown past the entry or the stop.
    """
    from trend_retrace import four_hour_trend

    reasons: List[str] = []
    hours = max(0.0, (now - created).total_seconds() / 3600)
    gap = ((entry - price) / price * 100) if price else None

    beyond_stop = price <= stop if direction == "Long" else price >= stop
    trend, trend_reason = four_hour_trend(df_4h, trend_candles)

    if beyond_stop:
        reasons.append(f"Price ({price:g}) is already past the planned stop ({stop:g}). "
                       f"The level this setup was built on has broken.")
        verdict = INVALIDATED
    elif trend is not None and trend != direction:
        reasons.append(f"The 4H candles now trend {trend.lower()}, against this "
                       f"{direction.lower()} setup.")
        verdict = TREND_GONE
    elif hours > max_age_hours:
        reasons.append(f"Saved {hours:.0f}h ago. The 5m support it was built on is long "
                       f"gone — re-scan rather than trusting these levels.")
        verdict = LEVELS_STALE
    elif trend is None:
        reasons.append(f"The 4H trend has paused: {trend_reason}")
        verdict = TREND_GONE
    else:
        moved_away = (gap is not None
                      and ((direction == "Long" and price > entry and gap < -5)
                           or (direction == "Short" and price < entry and gap > 5)))
        if moved_away:
            reasons.append(f"Price has run {abs(gap):.1f}% away from the entry without "
                           f"filling it. Chasing it now is the extended entry your rules "
                           f"warn against.")
            verdict = ENTRY_PASSED
        else:
            reasons.append(f"The 4H trend still runs {direction.lower()}, and price is "
                           f"{abs(gap):.2f}% from the entry." if gap is not None
                           else "The 4H trend still supports this setup.")
            verdict = STILL_VALID
    reasons.append(f"Saved {hours:.1f}h ago.")
    return ViabilityResult(verdict, reasons, hours, gap, price)
