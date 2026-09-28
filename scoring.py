"""
Setup scoring: a measure of strategy confluence, NOT a probability of profit,
and NOT the same thing as trade readiness (see readiness.py).

Evidence for each component must be explicitly True, False, or None
(unavailable). None always scores 0 — we never assume points for evidence
we don't actually have.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List

POSITIVE_COMPONENTS = [
    ("regime_alignment_1d_4h", 2, "1D/4H regime alignment"),
    ("support_resistance", 1, "Meaningful support/resistance"),
    ("liquidity_sweep_reclaim", 2, "Confirmed liquidity sweep and reclaim/rejection"),
    ("fib_confluence", 1, "Fibonacci confluence"),
    ("structure_4h_supports", 1, "4H structure supports the trade"),
    ("entry_confirmation_1h", 1, "Entry-timeframe confirmation"),
    ("invalidation_defined", 1, "Clear structural invalidation"),
    ("rr_at_least_2", 1, "Meets the minimum reward-to-risk (3:1)"),
]

NEGATIVE_COMPONENTS = [
    ("opposing_liquidity_ahead", -1, "Opposing liquidity immediately ahead"),
    ("chasing_extended_move", -1, "Chasing an extended move"),
]

ALL_COMPONENTS = POSITIVE_COMPONENTS + NEGATIVE_COMPONENTS
_MAX_RAW_SCORE = sum(pts for _, pts, _ in POSITIVE_COMPONENTS)  # 10


@dataclass
class ScoreLineItem:
    key: str
    label: str
    points_possible: int
    evidence: Optional[bool]   # True / False / None (unavailable)
    points_awarded: int


@dataclass
class ScoreResult:
    raw_score: float
    normalized_score: float  # clamped 0-10
    label: str                # "A+", "B", or "C" — setup quality, not win probability
    breakdown: List[ScoreLineItem]


def _label_for(score: float) -> str:
    if score >= 8:
        return "A+"
    if score >= 6:
        return "B"
    return "C"


def weakest_components(result: "ScoreResult", top_n: int = 3) -> List[ScoreLineItem]:
    """The positive-weighted components that scored 0 — i.e. what's actually
    missing from the setup. Used to explain why a B grade is still cautious,
    per the rulebook's 'treat B setups cautiously and explain what is
    weaker' requirement."""
    missing = [li for li in result.breakdown if li.points_possible > 0 and li.points_awarded == 0]
    missing.sort(key=lambda li: li.points_possible, reverse=True)
    return missing[:top_n]


def trade_policy_for_grade(label: str) -> str:
    """Rulebook policy, stated plainly: C setups are not traded; B setups
    are traded cautiously with the gap explained; A+ still isn't a
    guarantee — it's a checklist-quality label, not a win probability."""
    if label == "C":
        return "NO TRADE — score is below the B threshold. Do not trade C setups."
    if label == "B":
        return "TRADE CAUTIOUSLY — acceptable but not high-confluence. Review what's missing below before sizing normally."
    return "Meets the checklist bar for consideration — this is a quality label, not a probability of profit or a guarantee."


def score_setup(evidence: Dict[str, Optional[bool]]) -> ScoreResult:
    """
    evidence: dict mapping component key -> True (present) / False (checked,
    absent) / None (not evaluated / unavailable). Any key not present in the
    dict is treated as None (unavailable), scoring 0, never assumed.

    Duplicate-counting guard: each key can only contribute once, since this
    function reads each key exactly one time from the dict.
    """
    breakdown: List[ScoreLineItem] = []
    raw_score = 0.0

    for key, points, label in ALL_COMPONENTS:
        ev = evidence.get(key, None)
        if ev is True:
            awarded = points
        else:
            # False or None (missing/unavailable) both award zero — the
            # spec requires unavailable evidence to earn zero, not be assumed.
            awarded = 0
        raw_score += awarded
        breakdown.append(ScoreLineItem(key=key, label=label, points_possible=points,
                                        evidence=ev, points_awarded=awarded))

    normalized = max(0.0, min(10.0, raw_score))
    return ScoreResult(
        raw_score=raw_score,
        normalized_score=normalized,
        label=_label_for(normalized),
        breakdown=breakdown,
    )
