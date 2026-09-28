"""
Laddered entries: several limit orders instead of one.

WHEN THIS IS OFFERED: only when the 5m chart shows more than one confirmed
support below price (resistance above, for a short), spaced far enough apart to
be distinct levels rather than noise. A single support means a single entry —
inventing extra levels would be making up structure that isn't there.

THE RULE THAT CANNOT BEND: the WHOLE ladder risks one trade's risk. Sizes are
solved so that, if every level fills and the stop is then hit, the loss is
exactly the amount you chose to risk — not that amount multiplied by the number
of levels. Filling only part of the ladder means risking LESS, never more.

THE STOP SITS BELOW THE LOWEST LEVEL, so it is further away than it would be
for a single entry. That is the real cost of laddering: a deeper adverse move
is needed to be proven wrong, and each unit carries more risk.

WHAT YOU GAIN AND LOSE
  Gain: a better average entry, and a much better chance of being filled at all.
  Lose: if price tags only the first level and runs, you're in with a fraction
        of the intended size — a smaller win than a single entry would have given.

NOT TESTED: the backtest models single entries. Whether laddering helps THIS
strategy is unmeasured, so it is offered as an option, never as the default.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

EQUAL = "Equal size at each level"
HEAVIER_LOWER = "Bigger size at better prices"
WEIGHTINGS = [EQUAL, HEAVIER_LOWER]

MIN_LEVELS = 2
MAX_LEVELS = 5


@dataclass
class GridLevel:
    index: int
    price: float
    weight: float            # share of the full position
    units: float
    notional: float
    distance_pct: float      # from the live price
    risk_if_stopped: float   # loss from THIS level alone, in currency
    cumulative_units: float
    cumulative_risk: float   # loss if filled to here and then stopped


@dataclass
class GridPlan:
    direction: str
    levels: List[GridLevel]
    stop: float
    target: float
    average_entry: float
    total_units: float
    total_risk: float              # currency risked if every level fills, then stop
    reward_at_target: float
    reward_risk: float
    single_entry_price: float      # what a one-shot entry would have used
    single_entry_rr: Optional[float]
    warnings: List[str] = field(default_factory=list)

    @property
    def partial_fill_note(self) -> str:
        first = self.levels[0]
        pct = first.weight * 100
        return (f"If only the first level fills, you're in with {pct:.0f}% of the position "
                f"and risking ${first.risk_if_stopped:,.2f} of the ${self.total_risk:,.2f} "
                f"budget — a smaller loss, but a smaller win too.")


def usable_levels(levels: Sequence[float], direction: str, price: float,
                  atr: float, min_gap_atr: float = 0.6,
                  max_distance_atr: float = 8.0, max_levels: int = 3) -> List[float]:
    """Pick distinct, reachable levels from the candidate supports.

    Levels closer together than min_gap_atr are the same shelf seen twice, and
    anything beyond max_distance_atr is unlikely to be reached before the setup
    goes stale.
    """
    if not levels or atr <= 0 or price <= 0:
        return []
    on_side = [l for l in levels
               if (l < price if direction == "Long" else l > price) and l > 0]
    on_side.sort(reverse=(direction == "Long"))     # nearest to price first
    chosen: List[float] = []
    for level in on_side:
        if abs(price - level) > atr * max_distance_atr:
            continue
        if any(abs(level - c) < atr * min_gap_atr for c in chosen):
            continue
        chosen.append(level)
        if len(chosen) >= max_levels:
            break
    return chosen


def weights_for(n: int, style: str) -> List[float]:
    """Share of the position at each level, summing to 1."""
    if n <= 0:
        return []
    if style == HEAVIER_LOWER:
        raw = list(range(1, n + 1))          # 1, 2, 3 ... most size at the best price
    else:
        raw = [1.0] * n
    total = float(sum(raw))
    return [r / total for r in raw]


def build_grid(direction: str, levels: Sequence[float], price: float, stop: float,
               risk_amount: float, target_r: float = 3.0,
               weighting: str = EQUAL) -> Optional[GridPlan]:
    """Build a ladder whose FULL fill risks exactly `risk_amount`.

    stop must sit beyond the furthest level. The target is placed so that a
    full fill gives `target_r` times the risk, measured from the average entry.
    """
    if direction not in ("Long", "Short"):
        raise ValueError("direction must be Long or Short")
    levels = [float(l) for l in levels]
    if len(levels) < MIN_LEVELS or risk_amount <= 0 or price <= 0:
        return None
    levels.sort(reverse=(direction == "Long"))
    furthest = levels[-1]
    beyond = stop < furthest if direction == "Long" else stop > furthest
    if not beyond:
        return None                     # the stop must protect every level

    w = weights_for(len(levels), weighting)
    # Risk per unit at each level, weighted: solving for total units so that a
    # full fill losing to the stop costs exactly risk_amount.
    per_unit = [abs(l - stop) for l in levels]
    weighted_risk = sum(wi * pu for wi, pu in zip(w, per_unit))
    if weighted_risk <= 0:
        return None
    total_units = risk_amount / weighted_risk
    average_entry = sum(wi * l for wi, l in zip(w, levels))

    # Target measured from the average entry, in units of the same risk.
    spread = abs(average_entry - stop)
    target = (average_entry + target_r * spread if direction == "Long"
              else average_entry - target_r * spread)

    rows: List[GridLevel] = []
    cum_units = cum_risk = 0.0
    for i, (level, wi, pu) in enumerate(zip(levels, w, per_unit), start=1):
        units = total_units * wi
        risk_here = units * pu
        cum_units += units
        cum_risk += risk_here
        rows.append(GridLevel(
            index=i, price=level, weight=wi, units=units, notional=units * level,
            distance_pct=(level - price) / price * 100,
            risk_if_stopped=risk_here, cumulative_units=cum_units,
            cumulative_risk=cum_risk))

    reward = total_units * abs(target - average_entry)
    single_rr = None
    single_risk = abs(levels[0] - stop)
    if single_risk > 0:
        single_rr = abs(target - levels[0]) / single_risk

    warnings: List[str] = []
    deepest = abs(price - furthest) / price * 100
    warnings.append(
        f"The stop sits below all {len(levels)} levels, so it's {abs(price - stop) / price * 100:.2f}% "
        f"from the current price — further than a single entry would need."
        if direction == "Long" else
        f"The stop sits above all {len(levels)} levels, {abs(stop - price) / price * 100:.2f}% "
        f"from the current price — further than a single entry would need.")
    if deepest > 5:
        warnings.append(f"The furthest level is {deepest:.1f}% away and may never be reached.")

    return GridPlan(direction=direction, levels=rows, stop=stop, target=target,
                    average_entry=average_entry, total_units=total_units,
                    total_risk=cum_risk, reward_at_target=reward,
                    reward_risk=target_r, single_entry_price=levels[0],
                    single_entry_rr=single_rr, warnings=warnings)
