"""
Trade readiness state machine. Deliberately independent from scoring.py —
a high setup score must NEVER automatically make a setup READY. Readiness
depends on data validity, location, confirmation, and explicit user action.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List


class Readiness(str, Enum):
    NOT_READY = "NOT READY"
    CONDITIONAL = "CONDITIONAL"
    READY = "READY"
    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"
    DATA_ERROR = "DATA ERROR"


@dataclass
class ReadinessInputs:
    data_is_valid: bool                     # False => price/data unreliable
    invalidation_hit: bool = False          # structural invalidation or cancellation condition occurred
    user_marked_active: bool = False        # user has manually marked the trade as entered
    near_actionable_location: bool = False  # price is at/near the setup's relevant level
    confirmation_triggered: bool = False    # the entry-timeframe confirmation/trigger has occurred
    invalidation_defined: bool = False      # a structural invalidation level has been defined
    risk_checks_pass: bool = False          # e.g. position sizes, doesn't exceed risk budget, etc.


def determine_readiness(inputs: ReadinessInputs) -> Readiness:
    """Precedence, evaluated in order: DATA ERROR > INVALIDATED > ACTIVE >
    READY > CONDITIONAL > NOT READY (default)."""
    if not inputs.data_is_valid:
        return Readiness.DATA_ERROR
    if inputs.invalidation_hit:
        return Readiness.INVALIDATED
    if inputs.user_marked_active:
        return Readiness.ACTIVE
    if (
        inputs.near_actionable_location
        and inputs.confirmation_triggered
        and inputs.invalidation_defined
        and inputs.risk_checks_pass
    ):
        return Readiness.READY
    if inputs.near_actionable_location:
        return Readiness.CONDITIONAL
    return Readiness.NOT_READY


READINESS_DESCRIPTIONS = {
    Readiness.NOT_READY: "Required evidence is missing, or price is not near an actionable location.",
    Readiness.CONDITIONAL: "Location is relevant, but confirmation/trigger has not occurred yet.",
    Readiness.READY: "All required confirmations and risk checks pass.",
    Readiness.ACTIVE: "You have manually marked this trade as entered.",
    Readiness.INVALIDATED: "Structural invalidation or cancellation condition has occurred.",
    Readiness.DATA_ERROR: "Price/data is unreliable — no trade decision should be made on it.",
}


# Rulebook display terminology: candidates are shown to the user as one of
# three plain labels, even though the internal state machine has more detail.
DISPLAY_LABEL = {
    Readiness.READY: "READY",
    Readiness.ACTIVE: "READY",       # already in the trade; not a new-entry decision
    Readiness.CONDITIONAL: "WAITING",
    Readiness.NOT_READY: "NO TRADE",
    Readiness.INVALIDATED: "NO TRADE",
    Readiness.DATA_ERROR: "NO TRADE",
}


def display_label(readiness: Readiness) -> str:
    return DISPLAY_LABEL[readiness]


# The rulebook's required entry sequence, in order. A setup is only
# considered for READY once every applicable stage has evidence — this is
# the checklist UI code should walk through and display stage-by-stage,
# not a formula that outputs a single number (that's scoring.py's job, and
# the two are deliberately kept separate).
ENTRY_SEQUENCE_STAGES = [
    ("location", "Price reaches a meaningful support/resistance/Fibonacci/liquidity confluence area"),
    ("liquidity_event", "A relevant high/low is swept or an event occurs at that location"),
    ("reclaim_or_rejection", "Price reclaims the level, or clearly rejects it (close-based, not wick-only)"),
    ("confirmation", "Lower-timeframe confirmation consistent with the higher-timeframe thesis"),
    ("structural_invalidation", "A clear structural invalidation level is defined"),
    ("acceptable_rr", "Reward-to-risk is at least 3:1"),
    ("execution_plan", "Entry, stop, and target are fully specified and ready to log"),
]


def missing_entry_sequence_stages(evidence: Dict[str, bool]) -> List[str]:
    """Returns human-readable descriptions of which required stages are NOT
    yet confirmed — this is what the rulebook means by 'explain what would
    confirm it' when a setup is WAIT / NO TRADE."""
    missing = []
    for key, description in ENTRY_SEQUENCE_STAGES:
        if not evidence.get(key, False):
            missing.append(description)
    return missing
