"""
Dynamic stop/target management: initial hard stop, manual updates, trailing
plan/activation, and a log of every change. Enforces "never move a stop
farther away from entry" for profit-protecting updates — but that rule only
applies once a stop is being tightened in the trader's favor; it does not
block setting the ORIGINAL structural stop on a trade that is currently
losing (that's not "widening", that's the initial risk definition).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional


class StopManagementError(ValueError):
    pass


class ManagementLabel(str, Enum):
    """Rulebook's position-management labels, applied per open position at
    each review — deliberately separate from the entry-side Readiness enum
    in readiness.py, since managing an existing position is a different
    decision from deciding whether to enter a new one."""
    HOLD = "HOLD"          # thesis remains valid
    PROTECT = "PROTECT"    # adjust stop based on confirmed structure
    REDUCE = "REDUCE"      # consider lowering exposure
    EXIT = "EXIT"          # thesis invalidated or risk rule breached
    STAY_OUT = "STAY OUT"  # no qualifying new setup (candidates, not open positions)


def suggest_management_label(
    direction: Literal["Long", "Short"],
    invalidation_hit: bool,
    risk_rule_breached: bool,
    structure_supports_tightening: bool,
    r_multiple: float,
    reduce_threshold_r: Optional[float] = None,
) -> ManagementLabel:
    """A starting suggestion only — per the rulebook, this never auto-acts;
    it's a label for you to review, not an instruction that gets executed."""
    if invalidation_hit or risk_rule_breached:
        return ManagementLabel.EXIT
    if reduce_threshold_r is not None and r_multiple <= -abs(reduce_threshold_r):
        return ManagementLabel.REDUCE
    if structure_supports_tightening:
        return ManagementLabel.PROTECT
    return ManagementLabel.HOLD


@dataclass
class ManagementLogEntry:
    timestamp_utc: str
    field: Literal["stop", "target"]
    old_value: float
    new_value: float
    note: str


@dataclass
class StopManager:
    direction: Literal["Long", "Short"]
    entry: float
    current_stop: float
    current_target: float
    log: List[ManagementLogEntry] = field(default_factory=list)
    trailing_active: bool = False

    def _is_tightening(self, new_stop: float) -> bool:
        """True if new_stop moves the stop closer to (or past) current price
        in the trader's favor relative to the OLD stop — i.e. reduces risk."""
        if self.direction == "Long":
            return new_stop > self.current_stop
        return new_stop < self.current_stop

    def update_stop(self, new_stop: float, note: str = "", allow_initial_widen: bool = False) -> None:
        """Update the stop. Raises StopManagementError if this would move
        the stop AWAY from entry (increasing risk) after the trade has
        already had its initial stop set, unless allow_initial_widen=True
        (for explicitly re-basing a brand new trade's initial risk, not for
        an in-flight profit-protecting adjustment)."""
        if not allow_initial_widen and not self._is_tightening(new_stop):
            raise StopManagementError(
                f"Refusing to move stop from {self.current_stop} to {new_stop} for a "
                f"{self.direction} — that widens risk, not tightens it. The rule is "
                f"'never move a stop farther away from entry.' If this is genuinely a "
                f"fresh initial stop (not a live adjustment), pass allow_initial_widen=True."
            )
        old = self.current_stop
        self.current_stop = new_stop
        self.log.append(ManagementLogEntry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            field="stop", old_value=old, new_value=new_stop, note=note,
        ))

    def update_target(self, new_target: float, note: str = "") -> None:
        old = self.current_target
        self.current_target = new_target
        self.log.append(ManagementLogEntry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            field="target", old_value=old, new_value=new_target, note=note,
        ))

    def current_r_multiple(self, current_price: float) -> float:
        risk_distance = abs(self.entry - self.current_stop)
        if risk_distance == 0:
            return 0.0
        if self.direction == "Long":
            return (current_price - self.entry) / risk_distance
        return (self.entry - current_price) / risk_distance

    def suggest_transition_to_trailing(self, current_price: float,
                                        activation_r: float = 1.5) -> bool:
        """Per the spec: consider moving from the initial hard stop to a
        structure-based trailing stop only after ~1.5R-2R of favorable
        movement. This only SUGGESTS — it never auto-trails through noise."""
        return self.current_r_multiple(current_price) >= activation_r

    def validate_profit_protecting_stop(self, current_price: float,
                                         round_trip_cost_pct: float = 0.0) -> Optional[str]:
        """For a profit-protecting stop update, the new stop must sit between
        entry (adjusted for round-trip costs) and current price. Returns an
        error string, or None if valid. This check is only meaningful once
        trailing/protecting — it is NOT applied to a losing trade's original
        structural stop.

        round_trip_cost_pct: fees+slippage as a fraction of entry price
        (e.g. 0.001 = 0.1%). Per the rulebook, a stop isn't genuinely
        "profit-protecting" unless it clears entry by more than what
        round-trip costs would eat — otherwise it can lock in a net loss
        while still being labeled a "protecting" stop.
        """
        cost_buffer = self.entry * round_trip_cost_pct
        if self.direction == "Long":
            break_even_after_costs = self.entry + cost_buffer
            if not (break_even_after_costs <= self.current_stop <= current_price):
                return (f"Profit-protecting Long stop must be between break-even-after-costs "
                        f"({break_even_after_costs:.4g}, i.e. entry {self.entry} + costs) and "
                        f"current price ({current_price}); got {self.current_stop}.")
        else:
            break_even_after_costs = self.entry - cost_buffer
            if not (current_price <= self.current_stop <= break_even_after_costs):
                return (f"Profit-protecting Short stop must be between current price "
                        f"({current_price}) and break-even-after-costs "
                        f"({break_even_after_costs:.4g}, i.e. entry {self.entry} - costs); "
                        f"got {self.current_stop}.")
        return None
