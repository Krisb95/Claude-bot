"""
Expectancy: does a kind of setup make money over many trades?

"Profitable over the long run" has a precise meaning: positive EXPECTANCY —
the average result per trade, across many trades, is above zero. It does NOT
mean individual trades don't lose. At 3:1 a profitable system can still lose
most of its trades; it is profitable because the wins are three times the size.

This module turns backtest (or forward-tracked) results into evidence:

  * Segments  — groups of trades (e.g. all Longs, or all BTC Shorts) with their
                average result and a verdict on whether that is distinguishable
                from luck.
  * Evidence  — a label for a live setup, taken from the most specific segment
                that has enough trades to judge.
  * Streaks & drawdowns — what a given win rate actually feels like to trade,
                so a normal losing run is recognised as normal.

HONEST LIMITS: evidence comes from past data. A segment proven positive on a
few months of history can stop working. More trades and a longer history make
a verdict more trustworthy; forward tracking is the strongest test because the
plan is fixed before the result is known.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np

MIN_TRADES = 30

PROVEN_POSITIVE = "Proven positive"
PROVEN_NEGATIVE = "Proven negative"
UNPROVEN = "Unproven — could be luck"
TOO_FEW = "Too few trades"


@dataclass
class Segment:
    key: Tuple
    n: int
    wins: int
    win_rate: Optional[float]
    avg_r: Optional[float]
    total_r: float
    luck_range: Optional[float]
    verdict: str

    @property
    def label(self) -> str:
        return " · ".join(str(k) for k in self.key)


def _trade_r(t) -> Optional[float]:
    """Result of one trade in R, weighted if the trade carries a weight."""
    w = getattr(t, "weighted_r", None)
    if w is not None:
        return float(w)
    r = getattr(t, "r_result", None)
    return None if r is None else float(r)


def summarise(results: List[float]) -> Tuple[int, int, Optional[float], Optional[float],
                                              float, Optional[float], str]:
    """(n, wins, win_rate, avg_r, total_r, luck_range, verdict) for R results."""
    n = len(results)
    if n == 0:
        return 0, 0, None, None, 0.0, None, TOO_FEW
    wins = sum(1 for r in results if r > 0)
    total = float(sum(results))
    avg = total / n
    luck = 2.0 * float(np.std(results, ddof=1)) * float(np.sqrt(n)) if n >= 2 else None
    if n < MIN_TRADES or luck is None:
        verdict = TOO_FEW
    elif total > luck:
        verdict = PROVEN_POSITIVE
    elif total < -luck:
        verdict = PROVEN_NEGATIVE
    else:
        verdict = UNPROVEN
    return n, wins, wins / n, avg, total, luck, verdict


def segments(trades: Iterable, key: Callable) -> Dict[Tuple, Segment]:
    """Group resolved trades by key(trade) and summarise each group."""
    groups: Dict[Tuple, List[float]] = {}
    for t in trades:
        if getattr(t, "status", None) not in ("WIN", "LOSS"):
            continue
        r = _trade_r(t)
        if r is None:
            continue
        k = key(t)
        k = k if isinstance(k, tuple) else (k,)
        groups.setdefault(k, []).append(r)
    out = {}
    for k, rs in groups.items():
        n, wins, wr, avg, total, luck, verdict = summarise(rs)
        out[k] = Segment(k, n, wins, wr, avg, total, luck, verdict)
    return out


@dataclass
class EvidenceBook:
    """Evidence at three levels of detail, most specific first."""
    by_coin_direction: Dict[Tuple, Segment]
    by_direction: Dict[Tuple, Segment]
    overall: Optional[Segment]
    source: str = "backtest"

    @classmethod
    def from_trades(cls, trades: List, source: str = "backtest") -> "EvidenceBook":
        trades = list(trades)
        overall = segments(trades, lambda t: ("All",)).get(("All",))
        return cls(
            by_coin_direction=segments(trades, lambda t: (t.ticker, t.direction)),
            by_direction=segments(trades, lambda t: (t.direction,)),
            overall=overall, source=source)

    def evidence_for(self, ticker: str, direction: Optional[str]) -> Tuple[str, str]:
        """Label for one live setup, from the most specific segment that has
        enough trades for a verdict. Returns (verdict, explanation)."""
        if direction is None:
            return TOO_FEW, "No direction, so no matching evidence."
        candidates = [
            (self.by_coin_direction.get((ticker, direction)), f"{ticker} {direction}s"),
            (self.by_direction.get((direction,)), f"all {direction}s"),
            (self.overall, "all trades"),
        ]
        for seg, name in candidates:
            if seg is not None and seg.verdict != TOO_FEW:
                return seg.verdict, (f"{name}: {seg.n} trades, {seg.win_rate:.0%} won, "
                                     f"{seg.avg_r:+.2f}R per trade on average")
        return TOO_FEW, "Not enough backtested trades yet to judge this setup."

    def to_dict(self) -> Dict:
        def seg_list(d):
            return [{"key": list(s.key), "n": s.n, "wins": s.wins, "win_rate": s.win_rate,
                     "avg_r": s.avg_r, "total_r": s.total_r, "luck_range": s.luck_range,
                     "verdict": s.verdict} for s in d.values()]
        return {"source": self.source,
                "by_coin_direction": seg_list(self.by_coin_direction),
                "by_direction": seg_list(self.by_direction),
                "overall": seg_list({("All",): self.overall}) if self.overall else []}

    @classmethod
    def from_dict(cls, d: Dict) -> "EvidenceBook":
        def seg_map(items):
            out = {}
            for it in items:
                k = tuple(it["key"])
                out[k] = Segment(k, it["n"], it["wins"], it["win_rate"], it["avg_r"],
                                 it["total_r"], it["luck_range"], it["verdict"])
            return out
        overall = list(seg_map(d.get("overall", [])).values())
        return cls(seg_map(d.get("by_coin_direction", [])),
                   seg_map(d.get("by_direction", [])),
                   overall[0] if overall else None, d.get("source", "backtest"))


# ---------------------------------------------------------------------
# What a win rate feels like to trade
# ---------------------------------------------------------------------

def expected_r_per_trade(win_rate: float, reward_risk: float) -> float:
    """Average result per trade in R: win% x reward - loss% x 1."""
    return win_rate * reward_risk - (1.0 - win_rate)


def break_even_win_rate(reward_risk: float) -> float:
    return 1.0 / (1.0 + reward_risk)


@dataclass
class Expectations:
    win_rate: float
    reward_risk: float
    risk_pct: float
    n_trades: int
    r_per_trade: float
    pct_per_trade: float           # account % per trade on average
    expected_total_pct: float      # over n_trades
    streak_typical: int            # median longest losing streak
    streak_bad: int                # 95th percentile longest losing streak
    drawdown_typical_pct: float    # median worst peak-to-trough, % of account
    drawdown_bad_pct: float        # 95th percentile
    chance_of_loss_pct: float      # chance of being down after n_trades


def simulate_expectations(win_rate: float, reward_risk: float, risk_pct: float = 1.0,
                           n_trades: int = 100, sims: int = 4000,
                           seed: int = 7) -> Expectations:
    """Monte Carlo of many possible sequences of n_trades at this win rate.

    Shows the range of outcomes a trader should expect even when the edge is
    real — especially losing streaks and drawdowns, which are what cause
    people to abandon a working strategy or to over-risk trying to recover.
    """
    if not (0.0 <= win_rate <= 1.0):
        raise ValueError("win_rate must be between 0 and 1")
    if reward_risk <= 0 or risk_pct <= 0 or n_trades <= 0:
        raise ValueError("reward_risk, risk_pct and n_trades must be positive")

    rng = np.random.default_rng(seed)
    wins = rng.random((sims, n_trades)) < win_rate
    r = np.where(wins, reward_risk, -1.0)

    # Longest losing streak in each simulated sequence.
    current = np.zeros(sims, dtype=int)
    longest = np.zeros(sims, dtype=int)
    for i in range(n_trades):
        current = np.where(wins[:, i], 0, current + 1)
        longest = np.maximum(longest, current)

    equity = np.cumsum(r * risk_pct, axis=1)                 # % of account, simple
    peak = np.maximum.accumulate(np.concatenate([np.zeros((sims, 1)), equity], axis=1),
                                 axis=1)[:, 1:]
    drawdown = (peak - equity).max(axis=1)

    ev_r = expected_r_per_trade(win_rate, reward_risk)
    return Expectations(
        win_rate=win_rate, reward_risk=reward_risk, risk_pct=risk_pct, n_trades=n_trades,
        r_per_trade=ev_r, pct_per_trade=ev_r * risk_pct,
        expected_total_pct=ev_r * risk_pct * n_trades,
        streak_typical=int(np.median(longest)),
        streak_bad=int(np.percentile(longest, 95)),
        drawdown_typical_pct=float(np.median(drawdown)),
        drawdown_bad_pct=float(np.percentile(drawdown, 95)),
        chance_of_loss_pct=float((equity[:, -1] < 0).mean() * 100))


def required_rr(win_rate: float, edge_per_trade: float = 0.0) -> Optional[float]:
    """The reward:risk needed for a given win rate to produce a given edge.

    Rearranged from expectancy = p*R - (1-p):   R = (edge + 1 - p) / p

    This is the trade-off nobody escapes: a high win rate REQUIRES a small
    reward relative to risk, and a large reward REQUIRES accepting a low win
    rate. You can choose which side to sit on, not have both.
    """
    if not (0 < win_rate <= 1):
        return None
    return (edge_per_trade + 1.0 - win_rate) / win_rate


def win_rate_needed(reward_risk: float, edge_per_trade: float = 0.0) -> Optional[float]:
    """The win rate needed at this reward:risk to produce a given edge."""
    if reward_risk <= 0:
        return None
    return (1.0 + edge_per_trade) / (1.0 + reward_risk)
