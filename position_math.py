"""
Live position economics: unrealised P/L, outcomes at stop and target, and an
honest assessment of whether adding capital to an existing position is sound.

ON SCALING IN: the rulebook is explicit — never add to a losing leveraged
position unless that addition was planned in advance, and never exceed the
account risk budget. This module enforces both as hard checks rather than
suggestions, because "averaging down" is the single fastest way to convert a
small planned loss into an unplanned large one. A BLOCKED verdict is not a
formatting choice; it means the addition breaks a rule you set yourself.

Nothing here executes anything. Every number is an estimate based on the
prices you supply.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from formatting import format_price as _fp


@dataclass
class PositionSnapshot:
    """Everything derivable about an open position at the current price."""
    direction: str
    entry: float
    stop: float
    target: Optional[float]
    quantity: float
    current_price: float
    contract_multiplier: float = 1.0

    @property
    def notional(self) -> float:
        return self.entry * self.quantity * self.contract_multiplier

    @property
    def unrealised_pl(self) -> float:
        move = (self.current_price - self.entry if self.direction == "Long"
                else self.entry - self.current_price)
        return move * self.quantity * self.contract_multiplier

    @property
    def unrealised_pct(self) -> float:
        if self.notional == 0:
            return 0.0
        return self.unrealised_pl / self.notional * 100

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def loss_at_stop(self) -> float:
        """Negative number: what you lose if the stop is hit from here.

        If the stop has already moved past entry in your favour this is
        positive — a locked-in profit rather than a loss.
        """
        move = (self.stop - self.entry if self.direction == "Long"
                else self.entry - self.stop)
        return move * self.quantity * self.contract_multiplier

    @property
    def profit_at_target(self) -> Optional[float]:
        if self.target is None:
            return None
        move = (self.target - self.entry if self.direction == "Long"
                else self.entry - self.target)
        return move * self.quantity * self.contract_multiplier

    @property
    def reward_risk(self) -> Optional[float]:
        if self.target is None or self.risk_per_unit == 0:
            return None
        reward = abs(self.target - self.entry)
        return reward / self.risk_per_unit

    @property
    def r_multiple(self) -> float:
        """Current open profit measured in units of initial risk."""
        if self.risk_per_unit == 0:
            return 0.0
        move = (self.current_price - self.entry if self.direction == "Long"
                else self.entry - self.current_price)
        return move / self.risk_per_unit

    @property
    def is_in_profit(self) -> bool:
        return self.unrealised_pl > 0

    @property
    def stop_is_protecting_profit(self) -> bool:
        """True when the stop sits beyond entry in the profitable direction."""
        return (self.stop > self.entry if self.direction == "Long"
                else self.stop < self.entry)

    def realised_pl(self, exit_price: float) -> float:
        move = (exit_price - self.entry if self.direction == "Long"
                else self.entry - exit_price)
        return move * self.quantity * self.contract_multiplier


class ScaleVerdict(str, Enum):
    ALLOWED = "ALLOWED"
    CAUTION = "CAUTION"
    BLOCKED = "BLOCKED"


@dataclass
class ScaleInResult:
    verdict: ScaleVerdict
    new_average_entry: float
    new_quantity: float
    new_total_risk: float          # currency at risk to the stop after adding
    new_risk_pct_of_equity: float
    added_notional: float
    new_reward_risk: Optional[float]
    reasons: List[str] = field(default_factory=list)


def analyse_scale_in(position: PositionSnapshot, add_quantity: float,
                      add_price: float, account_equity: float,
                      max_risk_pct: float = 1.0,
                      preplanned: bool = False) -> ScaleInResult:
    """Assess adding `add_quantity` at `add_price` to an existing position.

    max_risk_pct is your per-trade risk budget as a percentage of equity.
    preplanned records whether this addition was part of the original plan —
    the rulebook permits adding to a loser only in that case.
    """
    reasons: List[str] = []

    if add_quantity <= 0 or add_price <= 0:
        return ScaleInResult(ScaleVerdict.BLOCKED, position.entry, position.quantity,
                              0.0, 0.0, 0.0, None,
                              ["Add quantity and price must both be positive."])
    if account_equity <= 0:
        return ScaleInResult(ScaleVerdict.BLOCKED, position.entry, position.quantity,
                              0.0, 0.0, 0.0, None,
                              ["Account equity must be positive."])

    new_quantity = position.quantity + add_quantity
    new_average_entry = (
        (position.entry * position.quantity + add_price * add_quantity) / new_quantity
    )
    added_notional = add_price * add_quantity * position.contract_multiplier

    # Risk is measured from the NEW average entry to the existing stop.
    risk_per_unit = abs(new_average_entry - position.stop)
    new_total_risk = risk_per_unit * new_quantity * position.contract_multiplier
    new_risk_pct = new_total_risk / account_equity * 100

    new_reward_risk = None
    if position.target is not None and risk_per_unit > 0:
        new_reward_risk = abs(position.target - new_average_entry) / risk_per_unit

    # --- hard checks -------------------------------------------------
    stop_still_valid = (position.stop < new_average_entry if position.direction == "Long"
                        else position.stop > new_average_entry)
    if not stop_still_valid:
        reasons.append(
            f"Adding at {_fp(add_price)} moves the average entry to {_fp(new_average_entry)}, "
            f"which puts it on the wrong side of the stop ({_fp(position.stop)}). "
            f"The position would already be beyond its own invalidation."
        )
        return ScaleInResult(ScaleVerdict.BLOCKED, new_average_entry, new_quantity,
                              new_total_risk, new_risk_pct, added_notional,
                              new_reward_risk, reasons)

    verdict = ScaleVerdict.ALLOWED

    if new_risk_pct > max_risk_pct:
        verdict = ScaleVerdict.BLOCKED
        reasons.append(
            f"Total risk after adding would be {new_risk_pct:.2f}% of equity, over your "
            f"{max_risk_pct:.2f}% budget. Reduce the size added, or tighten the stop first."
        )

    if not position.is_in_profit:
        if preplanned:
            if verdict is not ScaleVerdict.BLOCKED:
                verdict = ScaleVerdict.CAUTION
            reasons.append(
                "This position is underwater. You have marked the addition as planned in "
                "advance, which the rulebook permits — but the loss if the stop is hit is "
                "now larger, so confirm the original invalidation still holds."
            )
        else:
            verdict = ScaleVerdict.BLOCKED
            reasons.append(
                "This position is underwater and the addition was not planned in advance. "
                "The rulebook forbids averaging down on an unplanned basis — it increases "
                "the loss without any new evidence the thesis is working."
            )

    if position.is_in_profit and verdict is ScaleVerdict.ALLOWED:
        reasons.append(
            f"Position is up {position.r_multiple:.2f}R. Adding here raises the average "
            f"entry to {_fp(new_average_entry)}, so existing open profit partly cushions the "
            f"added risk — but the stop is now further from the new average entry."
        )

    if new_reward_risk is not None and new_reward_risk < 1.0:
        if verdict is ScaleVerdict.ALLOWED:
            verdict = ScaleVerdict.CAUTION
        reasons.append(
            f"After adding, reward:risk to the existing target falls to "
            f"{new_reward_risk:.2f}:1 — you would be risking more than the remaining upside."
        )

    if not reasons:
        reasons.append("Within risk budget and the position is in profit.")

    return ScaleInResult(verdict, new_average_entry, new_quantity, new_total_risk,
                          new_risk_pct, added_notional, new_reward_risk, reasons)
