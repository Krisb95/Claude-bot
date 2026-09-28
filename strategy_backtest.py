"""
Walk-forward backtest of the ACTUAL scanner strategy.

At each step the scanner is handed only the candles that had fully CLOSED at
that moment, produces its plan (entry zone, stop, target, score, grade) exactly
as it would live, and the plan is then replayed against the candles that came
afterwards using the shared limit-order simulator.

The question this answers: do higher grades actually produce better results?
If A+ setups don't beat C setups, the grading is not measuring anything useful,
no matter how sensible its rules sound.

LOOK-AHEAD PROTECTION — the easiest way to make a backtest lie is to let the
strategy peek at the future. Guarded against here by:
  * 4H: the slice ends at the bar whose close is the decision point.
  * 1D: a daily candle is only included once it has CLOSED. At 08:00 on a given
    day, that day's candle is still forming, so it is excluded — using its
    close would be using a price that did not exist yet.
  * Swing points are recomputed on each slice, so a swing is only "confirmed"
    if the bars confirming it were already in the past.

WHAT THIS STILL CANNOT TELL YOU:
  * A few months of one market regime is one sample. A strategy can look strong
    in a bull run and fail in a range. Small trade counts are reported as such.
  * The rules were written before this data was seen, which helps, but some
    thresholds were chosen with judgement, not measured.
  * Past performance does not guarantee future results. This measures whether
    the grades had an edge historically; it cannot promise they will again.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import pandas as pd

from scanner import analyze_candidate
from scoring import score_setup
from trade_sim import simulate_limit_trade, WIN, LOSS, EXPIRED, FILLED, PENDING

ONE_HOUR = pd.Timedelta(hours=1)
FOUR_HOURS = pd.Timedelta(hours=4)
ONE_DAY = pd.Timedelta(days=1)

GRADE_ORDER = ["A+", "B", "C"]
MIN_TRADES_FOR_CONFIDENCE = 30


@dataclass
class BacktestTrade:
    ticker: str
    signal_time: pd.Timestamp
    direction: str
    score: float
    grade: str
    entry: float
    stop: float
    target: float
    planned_rr: float
    status: str
    r_result: Optional[float]
    fill_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None


def closed_daily_slice(df_1d: pd.DataFrame, decision_time: pd.Timestamp,
                        max_bars: int = 200) -> pd.DataFrame:
    """Daily candles that had fully closed by decision_time.

    A daily candle opening at D closes at D + 1 day. It is only knowable once
    that close has happened, so candles still forming are excluded.
    """
    if df_1d is None or df_1d.empty:
        return pd.DataFrame()
    closed = df_1d[df_1d.index + ONE_DAY <= decision_time]
    return closed.tail(max_bars)


def closed_hourly_slice(df_1h: pd.DataFrame, decision_time: pd.Timestamp,
                         max_bars: int = 200) -> pd.DataFrame:
    """1H candles that had fully closed by decision_time (same rule as daily)."""
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    return df_1h[df_1h.index + ONE_HOUR <= decision_time].tail(max_bars)


def run_strategy_backtest(df_4h: pd.DataFrame, df_1d: Optional[pd.DataFrame],
                           ticker: str, min_rr: float = 2.0, window: int = 300,
                           step: int = 3, expiry_bars: int = 18, warmup: int = 120,
                           fee_r: float = 0.0,
                           progress: Optional[Callable[[int, int], None]] = None,
                           df_1h: Optional[pd.DataFrame] = None
                           ) -> List[BacktestTrade]:
    """Replay the scanner over history for one instrument.

    expiry_bars: how long a limit order rests before cancellation (18 x 4H = 3 days)
    step:        evaluate every Nth bar (1 = every bar; higher is faster)
    One trade at a time: while an order is pending or a position is open, new
    signals are ignored — otherwise overlapping duplicates inflate the counts.
    """
    trades: List[BacktestTrade] = []
    if df_4h is None or len(df_4h) < warmup + 2:
        return trades

    n = len(df_4h)
    busy_until = -1
    positions = list(range(warmup, n - 1, step))

    for k, t in enumerate(positions):
        if progress:
            progress(k, len(positions))
        if t <= busy_until:
            continue

        decision_time = df_4h.index[t] + FOUR_HOURS     # bar t has now closed
        slice_4h = df_4h.iloc[max(0, t - window + 1):t + 1]
        slice_1d = closed_daily_slice(df_1d, decision_time)
        slice_1h = closed_hourly_slice(df_1h, decision_time)

        try:
            analysis = analyze_candidate(
                ticker, {"4h": slice_4h, "1d": slice_1d, "1h": slice_1h},
                min_rr=min_rr)
        except Exception:
            continue

        if (analysis.direction is None or analysis.entry is None
                or analysis.stop is None or analysis.target is None
                or analysis.reward_risk is None or analysis.reward_risk < min_rr):
            continue

        scored = score_setup(analysis.score_dict())
        future = df_4h.iloc[t + 1:]
        try:
            sim = simulate_limit_trade(future, analysis.direction, analysis.entry,
                                        analysis.stop, analysis.target,
                                        expiry_bars=expiry_bars, fee_r=fee_r)
        except ValueError:
            continue

        trades.append(BacktestTrade(
            ticker=ticker, signal_time=decision_time, direction=analysis.direction,
            score=scored.normalized_score, grade=scored.label,
            entry=analysis.entry, stop=analysis.stop, target=analysis.target,
            planned_rr=analysis.reward_risk, status=sim.status, r_result=sim.r_result,
            fill_time=sim.fill_time, exit_time=sim.exit_time))

        # Block new signals until this one is resolved.
        if sim.exit_time is not None:
            busy_until = df_4h.index.get_loc(sim.exit_time)
        elif sim.status == EXPIRED:
            busy_until = t + expiry_bars
        else:
            busy_until = n   # still open/pending at end of data

    if progress:
        progress(len(positions), len(positions))
    return trades


@dataclass
class GradeStats:
    grade: str
    signals: int
    filled: int
    wins: int
    losses: int
    expired: int
    still_open: int
    win_rate: Optional[float]
    avg_r: Optional[float]           # expectancy per filled, resolved trade
    total_r: float
    fill_rate: Optional[float]
    enough_data: bool


def stats_by_grade(trades: List[BacktestTrade]) -> List[GradeStats]:
    out = []
    for grade in GRADE_ORDER + ["All"]:
        group = trades if grade == "All" else [t for t in trades if t.grade == grade]
        resolved = [t for t in group if t.status in (WIN, LOSS)]
        wins = sum(1 for t in resolved if t.status == WIN)
        losses = len(resolved) - wins
        expired = sum(1 for t in group if t.status == EXPIRED)
        still_open = sum(1 for t in group if t.status in (FILLED, PENDING))
        filled = len(resolved) + sum(1 for t in group if t.status == FILLED)
        total_r = sum(t.r_result for t in resolved if t.r_result is not None)
        out.append(GradeStats(
            grade=grade, signals=len(group), filled=filled, wins=wins,
            losses=losses, expired=expired, still_open=still_open,
            win_rate=(wins / len(resolved)) if resolved else None,
            avg_r=(total_r / len(resolved)) if resolved else None,
            total_r=total_r,
            fill_rate=(filled / len(group)) if group else None,
            enough_data=len(resolved) >= MIN_TRADES_FOR_CONFIDENCE))
    return out


def verdict(stats: List[GradeStats]) -> str:
    """Plain-English read of whether the grading separated good from bad."""
    by = {s.grade: s for s in stats}
    a, c = by.get("A+"), by.get("C")
    if not a or not c or a.avg_r is None or c.avg_r is None:
        return ("Not enough resolved trades in both A+ and C to compare grades. "
                "Run more coins or a longer history before drawing conclusions.")
    thin = not (a.enough_data and c.enough_data)
    caveat = (f" Caution: fewer than {MIN_TRADES_FOR_CONFIDENCE} resolved trades in at "
              f"least one grade, so this could easily be luck.") if thin else ""
    if a.avg_r > c.avg_r and a.avg_r > 0:
        return (f"A+ setups averaged {a.avg_r:+.2f}R per trade vs {c.avg_r:+.2f}R for C. "
                f"The grading separated better setups from worse ones in this sample."
                + caveat)
    if a.avg_r > c.avg_r:
        return (f"A+ beat C ({a.avg_r:+.2f}R vs {c.avg_r:+.2f}R) but was still negative. "
                f"Grading helped relative to C, yet A+ lost money here." + caveat)
    return (f"A+ did NOT beat C ({a.avg_r:+.2f}R vs {c.avg_r:+.2f}R). In this sample the "
            f"grades are not identifying better trades, and the scoring needs rethinking."
            + caveat)
