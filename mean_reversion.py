"""
Mean Reversion — the high-win-rate design, built so it can be measured.

THE IDEA: buy a stretched market at a level it has respected before, and take a
small profit as it snaps back. Small target, wide stop. That produces a high
win rate by construction — and a large loss when it fails.

THE ARITHMETIC NOBODY ESCAPES: at reward:risk R, a market with no edge gives a
win rate of 1/(1+R). At 0.33:1 that is 75%. So a 75% win rate here is the
BASELINE, not an achievement — the strategy only makes money above about 80%.
Seeing "75% wins" and concluding it works would be the mistake this module
exists to test for.

WHY IT MIGHT STILL WORK: extended price at a level that has held before is one
of the few places short-term reversion is plausible. Whether that beats the
baseline is an empirical question, which is why this ships with a backtest
rather than a recommendation.

WHAT IT COSTS: one skipped stop erases six wins. This is the style that ruins
accounts when someone decides to "give it room".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from technical import find_swing_points
from formatting import format_price as _fp

NO_LEVEL = "No level nearby"
NOT_EXTENDED = "Not stretched far enough"
NO_REJECTION = "Waiting for a rejection candle"
READY = "Ready — rejection at the level"


@dataclass
class MeanReversionParams:
    lookback: int = 300
    swing_left: int = 3
    swing_right: int = 3
    touch_tolerance: float = 0.015
    min_touches: int = 2
    sma_period: int = 20
    min_extension_atr: float = 1.5     # how stretched from the average
    near_level_atr: float = 0.75       # how close to the level price must be
    stop_atr: float = 1.5              # deliberately wide
    target_r: float = 0.4              # deliberately small
    max_hold_bars: int = 24            # if it hasn't reverted, the reason has gone
    require_rejection: bool = True


@dataclass
class MeanReversionPlan:
    ticker: str
    direction: Optional[str]
    stage: str
    score: float
    grade: str
    price: Optional[float]
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    reward_risk: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, object] = field(default_factory=dict)


def atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(period).mean().iloc[-1])
    return v if np.isfinite(v) and v > 0 else None


def levels_with_touches(df: pd.DataFrame, params: MeanReversionParams):
    """Prices the market has turned at more than once."""
    if df is None or len(df) < params.swing_left + params.swing_right + 5:
        return []
    window = df.tail(params.lookback)
    swings = [s for s in find_swing_points(window, params.swing_left, params.swing_right)
              if s.confirmed]
    clusters = []
    for s in swings:
        for c in clusters:
            if abs(s.price - c["price"]) / c["price"] <= params.touch_tolerance:
                c["prices"].append(s.price)
                c["price"] = float(np.mean(c["prices"]))
                c["touches"] += 1
                c["kinds"].append(s.kind)
                break
        else:
            clusters.append({"price": s.price, "prices": [s.price], "touches": 1,
                             "kinds": [s.kind]})
    return [c for c in clusters if c["touches"] >= params.min_touches]


def analyze(ticker: str, df: pd.DataFrame, price: Optional[float] = None,
            params: Optional[MeanReversionParams] = None) -> MeanReversionPlan:
    """Look for a stretched market at a level that has held before."""
    import patterns as pattern_lib

    params = params or MeanReversionParams()
    if price is None and df is not None and not df.empty:
        price = float(df["Close"].iloc[-1])
    if not price or df is None or len(df) < params.sma_period + 5:
        return MeanReversionPlan(ticker, None, NO_LEVEL, 0.0, "C", price,
                                  reasons=["Not enough candles."])

    a = atr(df)
    sma = float(df["Close"].tail(params.sma_period).mean())
    if not a or not sma:
        return MeanReversionPlan(ticker, None, NO_LEVEL, 0.0, "C", price,
                                  reasons=["Couldn't measure volatility."])

    extension = (price - sma) / a
    reasons = [f"Price {_fp(price)} is {extension:+.1f} ATR from its "
               f"{params.sma_period}-period average ({_fp(sma)})."]

    # Stretched below the average = look to buy; above = look to sell.
    direction = ("Long" if extension <= -params.min_extension_atr
                 else "Short" if extension >= params.min_extension_atr else None)
    if direction is None:
        return MeanReversionPlan(ticker, None, NOT_EXTENDED, 0.0, "C", price,
                                  reasons=reasons + [
                                      f"Needs at least {params.min_extension_atr:g} ATR of "
                                      f"stretch before there is anything to revert."])
    score = 4.0

    levels = levels_with_touches(df, params)
    side = [c for c in levels
            if (c["price"] <= price * 1.02 if direction == "Long"
                else c["price"] >= price * 0.98)]
    if not side:
        return MeanReversionPlan(ticker, direction, NO_LEVEL, score, "C", price,
                                  reasons=reasons + ["No level with two or more touches "
                                                      "here — nothing to lean on."])
    level = min(side, key=lambda c: abs(c["price"] - price))
    distance_atr = abs(price - level["price"]) / a
    reasons.append(f"Level at {_fp(level['price'])}, touched {level['touches']} times, "
                   f"{distance_atr:.1f} ATR away.")
    if distance_atr > params.near_level_atr:
        return MeanReversionPlan(ticker, direction, NO_LEVEL, score, "C", price,
                                  reasons=reasons + ["Too far from the level to act on."])
    score += 3.0

    found = pattern_lib.detect(df)
    want = "bullish" if direction == "Long" else "bearish"
    rejection = [p.name for p in found if p.bias == want]
    if rejection:
        score += 3.0
        reasons.append("Rejection candle: " + ", ".join(rejection) + ".")
    elif params.require_rejection:
        return MeanReversionPlan(ticker, direction, NO_REJECTION, score,
                                  "B" if score >= 6 else "C", price, reasons=reasons)

    entry = price
    stop = (level["price"] - a * params.stop_atr if direction == "Long"
            else level["price"] + a * params.stop_atr)
    risk = abs(entry - stop)
    target = (entry + risk * params.target_r if direction == "Long"
              else entry - risk * params.target_r)
    reasons.append(f"Stop {_fp(stop)} sits {params.stop_atr:g} ATR beyond the level — wide "
                   f"on purpose, so ordinary noise doesn't reach it. Target {_fp(target)} "
                   f"is only {params.target_r:g}R away, which is where the high win rate "
                   f"comes from — and why one loss undoes "
                   f"{1 / params.target_r:.0f} wins.")

    plan = MeanReversionPlan(ticker, direction, READY,
                              min(score, 10.0), "A+" if score >= 9 else "B", price,
                              entry=entry, stop=stop, target=target,
                              reward_risk=params.target_r, reasons=reasons)
    plan.features = {
        "direction": direction, "extension_atr": round(float(extension), 2),
        "level_touches": level["touches"], "distance_atr": round(distance_atr, 2),
        "risk_pct": round(risk / entry * 100, 3),
        "volatility_pct": round(a / price * 100, 3),
        "has_rejection": bool(rejection),
    }
    return plan


def run_backtest(df: pd.DataFrame, ticker: str,
                 params: Optional[MeanReversionParams] = None, warmup: int = 60,
                 fee_r: float = 0.0):
    """Replay the rules over history. Same no-look-ahead rule as the others."""
    from trade_sim import simulate_limit_trade

    params = params or MeanReversionParams()
    trades = []
    if df is None or len(df) < warmup + 10:
        return trades
    busy_until = -1
    for t in range(warmup, len(df) - 1):
        if t <= busy_until:
            continue
        past = df.iloc[:t + 1]
        try:
            plan = analyze(ticker, past, params=params)
        except Exception:
            continue
        if plan.stage != READY or not plan.entry:
            continue
        future = df.iloc[t + 1:]
        try:
            sim = simulate_limit_trade(future, plan.direction, plan.entry, plan.stop,
                                        plan.target, expiry_bars=params.max_hold_bars,
                                        fee_r=fee_r)
        except ValueError:
            continue
        trades.append({
            "ticker": ticker, "time": df.index[t], "direction": plan.direction,
            "grade": plan.grade, "entry": plan.entry, "stop": plan.stop,
            "target": plan.target, "planned_rr": plan.reward_risk,
            "status": sim.status, "r_result": sim.r_result,
            "features": plan.features,
        })
        if sim.exit_time is not None:
            busy_until = df.index.get_loc(sim.exit_time)
        else:
            busy_until = t + params.max_hold_bars
    return trades


def scan_universe(instruments, frame_loader, spot_loader=None,
                  params: Optional[MeanReversionParams] = None, progress=None):
    """Rank instruments on the mean-reversion rules.

    Returns scanner.RankedCandidate rows so the existing results table, saving
    and tracking all work unchanged.
    """
    from scanner import RankedCandidate

    params = params or MeanReversionParams()
    out = []
    total = len(instruments)
    for i, (label, ticker, key) in enumerate(instruments):
        if progress:
            progress(i, total, label)
        try:
            df = frame_loader(key)
            if df is None or df.empty:
                out.append(RankedCandidate(ticker, label, None, 0.0, "—", "unknown",
                                            None, None, None, None,
                                            error="No candles returned."))
                continue
            live = None
            if spot_loader is not None:
                try:
                    live = spot_loader(key)
                except Exception:
                    live = None
            plan = analyze(ticker, df, price=live, params=params)
            out.append(RankedCandidate(
                ticker=ticker, label=label, direction=plan.direction, score=plan.score,
                grade=plan.grade, regime=plan.stage, price=plan.price, stop=plan.stop,
                target=plan.target, reward_risk=plan.reward_risk,
                note="; ".join(plan.reasons[:3]), entry_status=plan.stage,
                entry=plan.entry, price_is_live=live is not None,
                features=plan.features or None))
        except Exception as e:
            out.append(RankedCandidate(ticker, label, None, 0.0, "—", "unknown",
                                        None, None, None, None,
                                        error=f"{type(e).__name__}: {e}"))
    if progress:
        progress(total, total, "done")

    stage_rank = {READY: 0, NO_REJECTION: 1, NO_LEVEL: 2, NOT_EXTENDED: 3}
    out.sort(key=lambda r: (r.error is not None, stage_rank.get(r.entry_status, 9),
                             -r.score))
    return out
