"""
Where should the stop be, given where you actually got in?

Your fill is rarely the planned price, and once a trade moves the original stop
often stops making sense. This works out a stop from the entry you really got,
and updates it as structure forms — always in one direction.

THE RULE THAT NEVER BENDS: a stop may only move closer to price, never further
away. Widening a stop converts a planned loss into an open-ended one, and it is
the single most common way a good plan turns into a bad one. Every suggestion
here is checked against the current stop and discarded if it would widen it.

THE LADDER, tightest first:
  1. Trail  — just beyond the most recent confirmed swing that sits between the
              current stop and price. Only offered once one exists.
  2. Breakeven — once the trade is up 1R, the entry itself, so a winner cannot
              become a full loss.
  3. Structure — beyond the last confirmed swing before entry: the original
              invalidation, used when nothing better has formed yet.

It is advice, not automation: nothing is sent to an exchange.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from technical import find_swing_points

TRAIL = "Trail behind new structure"
BREAKEVEN = "Move to breakeven"
STRUCTURE = "Structural stop"
HOLD = "Leave the stop where it is"

BREAKEVEN_AT_R = 1.0        # profit, in R, before breakeven is suggested


@dataclass
class StopSuggestion:
    basis: str
    price: Optional[float]
    reason: str
    is_tighter: bool
    current_r: Optional[float] = None
    locked_in: Optional[float] = None       # currency secured if it triggers, per unit
    risk_per_unit: Optional[float] = None
    alternatives: List[str] = None


def _beyond(direction: str, level: float, buffer: float) -> float:
    return level - buffer if direction == "Long" else level + buffer


def is_tighter(direction: str, new_stop: float, current_stop: Optional[float]) -> bool:
    """A stop is tighter only if it moves TOWARDS price."""
    if current_stop is None:
        return True
    return new_stop > current_stop if direction == "Long" else new_stop < current_stop


def safely_below(direction: str, stop: float, price: float, buffer: float) -> bool:
    """Leaves enough room that ordinary noise won't take it out immediately."""
    return (stop < price - buffer) if direction == "Long" else (stop > price + buffer)


def structural_stop(direction: str, entry: float, candles: pd.DataFrame, buffer: float,
                    left: int = 3, right: int = 3) -> Optional[float]:
    """Beyond the last confirmed swing on the entry side — the original
    invalidation for the trade."""
    if candles is None or len(candles) < left + right + 1:
        return None
    swings = [s for s in find_swing_points(candles, left, right) if s.confirmed]
    if direction == "Long":
        lows = [s.price for s in swings if s.kind == "low" and s.price < entry]
        return _beyond("Long", max(lows), buffer) if lows else None
    highs = [s.price for s in swings if s.kind == "high" and s.price > entry]
    return _beyond("Short", min(highs), buffer) if highs else None


def trailing_stop(direction: str, price: float, current_stop: Optional[float],
                  candles: pd.DataFrame, buffer: float,
                  left: int = 3, right: int = 3) -> Optional[float]:
    """Beyond the newest confirmed swing sitting between the stop and price."""
    if candles is None or len(candles) < left + right + 1:
        return None
    swings = [s for s in find_swing_points(candles, left, right) if s.confirmed]
    kind = "low" if direction == "Long" else "high"
    candidates = []
    for s in swings:
        if s.kind != kind:
            continue
        level = _beyond(direction, s.price, buffer)
        if not safely_below(direction, level, price, buffer):
            continue
        if current_stop is not None and not is_tighter(direction, level, current_stop):
            continue
        candidates.append((s.index, level))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])[1]


def suggest(direction: str, entry: float, price: float, candles: pd.DataFrame,
            atr: Optional[float], current_stop: Optional[float] = None,
            buffer_atr: float = 0.5, allow_breakeven: bool = True) -> StopSuggestion:
    """Where the stop should be now, given the entry you actually got."""
    if direction not in ("Long", "Short"):
        raise ValueError("direction must be Long or Short")
    if entry <= 0 or price <= 0:
        raise ValueError("entry and price must be positive")

    buffer = (atr or 0.0) * buffer_atr
    alternatives: List[str] = []

    initial = current_stop if current_stop is not None else structural_stop(
        direction, entry, candles, buffer)
    risk = abs(entry - initial) if initial is not None else None
    moved = (price - entry) if direction == "Long" else (entry - price)
    current_r = (moved / risk) if risk else None

    # 1. Trail behind new structure — only once a stop already exists. With no
    # stop yet there is nothing to trail: the first one comes from structure.
    trail = None
    if current_stop is not None:
        trail = trailing_stop(direction, price, current_stop, candles, buffer)
        if trail is not None:
            alternatives.append(f"{TRAIL}: {trail:,.6g}")

    # 2. Breakeven, once the trade has earned it.
    breakeven = None
    if (allow_breakeven and current_r is not None and current_r >= BREAKEVEN_AT_R
            and safely_below(direction, entry, price, buffer)
            and is_tighter(direction, entry, current_stop)):
        breakeven = entry
        alternatives.append(f"{BREAKEVEN}: {entry:,.6g}")

    # Pick the tightest that still leaves room.
    options = [(TRAIL, trail), (BREAKEVEN, breakeven)]
    valid = [(b, p) for b, p in options
             if p is not None and is_tighter(direction, p, current_stop)
             and safely_below(direction, p, price, buffer)]
    if valid:
        basis, chosen = max(valid, key=lambda bp: bp[1] if direction == "Long" else -bp[1])
    elif current_stop is None and initial is not None:
        basis, chosen = STRUCTURE, initial
    else:
        basis, chosen = HOLD, current_stop

    if chosen is None:
        return StopSuggestion(
            HOLD, None,
            "No confirmed swing to place a stop behind yet. Until one forms, the only "
            "options are a fixed distance from entry or staying out.",
            False, current_r, None, None, alternatives)

    locked = ((chosen - entry) if direction == "Long" else (entry - chosen))
    tighter = is_tighter(direction, chosen, current_stop)

    if basis == TRAIL:
        reason = (f"A new swing has formed since you entered. Moving the stop just beyond "
                  f"it to {chosen:,.6g} cuts the risk while leaving room for normal noise.")
    elif basis == BREAKEVEN:
        reason = (f"The trade is up {current_r:.2f}R. Moving the stop to your entry "
                  f"({chosen:,.6g}) means it can no longer become a full loss.")
    elif basis == STRUCTURE:
        reason = (f"Stop at {chosen:,.6g}, just beyond the last confirmed swing before "
                  f"your entry — the price that would say the trade was wrong.")
    else:
        reason = ("Nothing has formed that would let you tighten the stop without putting "
                  "it in the way of ordinary noise. Leave it where it is.")

    return StopSuggestion(basis, chosen, reason, tighter, current_r,
                          locked_in=locked, risk_per_unit=abs(entry - chosen),
                          alternatives=alternatives)


@dataclass
class TrailingPlan:
    activate_at: float          # price at which trailing switches on
    activate_at_r: float        # that price, in R from entry
    distance: float             # trail distance in price terms
    distance_pct: float         # the same distance as a % of the activation price
    locked_in_at_activation: float   # R secured if it triggers straight after switching on
    description: str

    @property
    def summary(self) -> str:
        return (f"activate at {self.activate_at:,.6g}, trail by {self.distance:,.6g} "
                f"({self.distance_pct:.2f}%)")


def trailing_plan(direction: str, entry: float, stop: float, atr: Optional[float],
                  activate_at_r: float = 1.0, trail_atr_mult: float = 2.0
                  ) -> Optional[TrailingPlan]:
    """The two numbers an exchange trailing stop needs: when to switch it on,
    and how far behind price to follow.

    Activation is set in R rather than a fixed percentage so it scales with the
    trade's own risk: "once this has made as much as it was risking" means the
    same thing on a tight stop and a wide one.

    The trail distance comes from ATR for the same reason — a fixed percentage
    is too tight on a volatile market and too loose on a quiet one.

    Returns None when there is nothing sensible to suggest (no ATR, or a trail
    so wide it would sit beyond the entry and lock in a loss).
    """
    risk = abs(entry - stop)
    if risk <= 0 or not atr or atr <= 0 or entry <= 0:
        return None

    activate = (entry + activate_at_r * risk if direction == "Long"
                else entry - activate_at_r * risk)
    distance = trail_atr_mult * atr

    # Where the stop would sit the moment trailing switches on.
    first_stop = activate - distance if direction == "Long" else activate + distance
    locked = ((first_stop - entry) if direction == "Long" else (entry - first_stop)) / risk
    if locked < 0:
        # The trail is wider than the gain at activation, so switching on there
        # would still leave the stop below entry. Say so rather than pretend.
        note = (f"At {activate_at_r:g}R the trail would still sit {abs(locked):.2f}R below "
                f"your entry, so it wouldn't protect anything yet. Either activate later "
                f"or trail more tightly.")
    else:
        note = (f"Once price reaches {activate:,.6g} ({activate_at_r:g}R), the stop starts "
                f"following {distance:,.6g} behind — locking in about {locked:.2f}R "
                f"immediately and more as price runs.")

    return TrailingPlan(activate_at=activate, activate_at_r=activate_at_r,
                        distance=distance,
                        distance_pct=distance / activate * 100 if activate else 0.0,
                        locked_in_at_activation=locked, description=note)
