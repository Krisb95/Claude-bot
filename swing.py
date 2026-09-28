"""
Swing Levels — a higher-timeframe strategy with far fewer decisions.

WHAT THIS IS: the standard swing-trading principles taught widely (including by
traders like Gareth Soloway) — trade at major horizontal levels, in the
direction of the higher-timeframe trend, with a confirming candle, a stop
beyond the level, and a target at the next level.

WHAT THIS IS NOT: anyone's proprietary system. Named techniques taught in paid
courses are not reproduced here, because guessing at their rules and putting
someone's name on the result would be misleading. Every rule below is stated
explicitly so you can see exactly what it does.

WHY FEWER DECISIONS
  * Daily candles: one check a day, not every five minutes.
  * A level must have been touched at least twice — that alone rules out most
    charts on most days.
  * Trades last days to weeks, so there is nothing to monitor intraday.
  * Years of daily history exist, so it can actually be backtested properly,
    unlike 5-minute data which is capped at a couple of weeks here.

THE TRADE-OFF: wider stops in absolute terms, so position sizes are smaller for
the same risk, and fewer opportunities. You wait more and act less.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from technical import find_swing_points
from formatting import format_price as _fp

# Stages, in the order a setup develops.
NO_TREND = "No clear daily trend"
NO_LEVEL = "No major level nearby"
APPROACHING = "Approaching the level"
AT_LEVEL = "At the level — waiting for a confirming candle"
CONFIRMED = "Confirmed — ready"

POINTS_TREND = 3
POINTS_LEVEL = 4          # the level is the heart of this strategy
POINTS_PATTERN = 3


@dataclass
class SwingParams:
    lookback_days: int = 400        # how much history to read levels from
    swing_left: int = 3
    swing_right: int = 3
    touch_tolerance: float = 0.015  # prices within 1.5% count as the same level
    min_touches: int = 2            # a level must have held at least twice
    near_atr: float = 1.0           # "at the level" = within this many daily ATR
    approach_atr: float = 4.0       # beyond this, it isn't actionable yet
    stop_atr: float = 0.75          # stop this far beyond the level
    min_rr: float = 3.0
    trend_sma: int = 50             # the trend filter
    require_pattern: bool = True    # a confirming candle at the level


@dataclass
class Level:
    price: float
    touches: int
    kind: str                       # "support" / "resistance"
    last_touch_index: int

    @property
    def strength(self) -> str:
        if self.touches >= 4:
            return "major"
        if self.touches == 3:
            return "strong"
        return "moderate"


@dataclass
class SwingPlan:
    ticker: str
    direction: Optional[str]
    stage: str
    score: float
    grade: str
    price: Optional[float]
    level: Optional[Level] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    reward_risk: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    features: Dict[str, object] = field(default_factory=dict)


def daily_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(period).mean().iloc[-1])
    return v if np.isfinite(v) and v > 0 else None


def trend_of(df: pd.DataFrame, params: SwingParams) -> Tuple[Optional[str], str]:
    """Daily trend: price against its moving average, plus swing structure.

    Both must agree. A price above its average in a chart making lower highs is
    a bounce in a downtrend, not an uptrend — requiring agreement removes most
    of those.
    """
    n = params.trend_sma
    if df is None or len(df) < n + 5:
        return None, f"Need at least {n + 5} daily candles to judge the trend."
    sma = float(df["Close"].tail(n).mean())
    price = float(df["Close"].iloc[-1])
    swings = [s for s in find_swing_points(df, params.swing_left, params.swing_right)
              if s.confirmed]
    highs = [s.price for s in swings if s.kind == "high"][-2:]
    lows = [s.price for s in swings if s.kind == "low"][-2:]
    enough_structure = len(highs) == 2 and len(lows) == 2
    rising = enough_structure and highs[1] > highs[0] and lows[1] > lows[0]
    falling = enough_structure and highs[1] < highs[0] and lows[1] < lows[0]

    if not enough_structure:
        # Not enough confirmed swings to check structure — usually a very
        # smooth or very short chart. Fall back to the moving average alone
        # and say so, rather than rejecting a trend that may well be real.
        side = "above" if price > sma else "below"
        return (("Long" if price > sma else "Short"),
                f"Price is {side} its {n}-day average ({_fp(sma)}), but there aren't enough "
                f"confirmed swing points to check the structure — a weaker read than usual.")

    if price > sma and rising:
        return "Long", (f"Daily uptrend: price {_fp(price)} is above its {n}-day average "
                        f"{_fp(sma)}, and the chart is making higher highs and higher lows.")
    if price < sma and falling:
        return "Short", (f"Daily downtrend: price {_fp(price)} is below its {n}-day average "
                         f"{_fp(sma)}, and the chart is making lower highs and lower lows.")
    side = "above" if price > sma else "below"
    return None, (f"No clear trend: price is {side} its {n}-day average but the swing "
                  f"structure doesn't agree — usually a range or a turning point.")


def major_levels(df: pd.DataFrame, params: SwingParams) -> List[Level]:
    """Horizontal levels the market has respected more than once.

    Swing highs and lows within touch_tolerance of each other are treated as
    one level, and the count of touches is what makes it major. A level touched
    once is just a past price.
    """
    if df is None or len(df) < params.swing_left + params.swing_right + 5:
        return []
    window = df.tail(params.lookback_days)
    swings = [s for s in find_swing_points(window, params.swing_left, params.swing_right)
              if s.confirmed]
    clusters: List[Dict] = []
    for s in swings:
        placed = False
        for c in clusters:
            if abs(s.price - c["price"]) / c["price"] <= params.touch_tolerance:
                c["prices"].append(s.price)
                c["price"] = float(np.mean(c["prices"]))
                c["touches"] += 1
                c["last"] = max(c["last"], s.index)
                c["kinds"].append(s.kind)
                placed = True
                break
        if not placed:
            clusters.append({"price": s.price, "prices": [s.price], "touches": 1,
                             "last": s.index, "kinds": [s.kind]})
    out = []
    for c in clusters:
        if c["touches"] < params.min_touches:
            continue
        kind = "support" if c["kinds"].count("low") >= c["kinds"].count("high") else "resistance"
        out.append(Level(price=float(c["price"]), touches=int(c["touches"]),
                          kind=kind, last_touch_index=int(c["last"])))
    out.sort(key=lambda lv: (-lv.touches, -lv.last_touch_index))
    return out


def nearest_level(levels: List[Level], direction: str, price: float) -> Optional[Level]:
    """The strongest level on the side the trade would enter from."""
    if direction == "Long":
        below = [lv for lv in levels if lv.price < price]
        if not below:
            return None
        return max(below, key=lambda lv: (lv.touches, lv.price))
    above = [lv for lv in levels if lv.price > price]
    if not above:
        return None
    return max(above, key=lambda lv: (lv.touches, -lv.price))


def target_for(levels: List[Level], direction: str, entry: float, stop: float,
               min_rr: float) -> Tuple[Optional[float], str]:
    """The next major level beyond entry that clears the minimum reward:risk."""
    risk = abs(entry - stop)
    if risk <= 0:
        return None, "Stop and entry are the same price."
    beyond = sorted([lv.price for lv in levels if lv.price > entry]) if direction == "Long" \
        else sorted([lv.price for lv in levels if lv.price < entry], reverse=True)
    for lv in beyond:
        if abs(lv - entry) / risk >= min_rr:
            return lv, (f"Target {_fp(lv)} — the next level clearing "
                        f"{min_rr:g}:1, giving {abs(lv - entry) / risk:.2f}:1.")
    if beyond:
        best = beyond[-1]
        return None, (f"The furthest level beyond entry ({_fp(best)}) only gives "
                      f"{abs(best - entry) / risk:.2f}:1 — under your {min_rr:g}:1 minimum.")
    return None, "No level beyond the entry to aim at."


def _grade(score: float) -> str:
    return "A+" if score >= 8 else "B" if score >= 6 else "C"


def analyze(ticker: str, df_daily: pd.DataFrame, price: Optional[float] = None,
            params: Optional[SwingParams] = None,
            direction_override: Optional[str] = None) -> SwingPlan:
    """Apply the swing rules to closed daily candles."""
    import patterns as pattern_lib

    params = params or SwingParams()
    if price is None and df_daily is not None and not df_daily.empty:
        price = float(df_daily["Close"].iloc[-1])
    if not price or df_daily is None or df_daily.empty:
        return SwingPlan(ticker, None, NO_TREND, 0.0, "C", price,
                         reasons=["No usable price or candles."])

    if price and df_daily is not None and not df_daily.empty:
        last_close = float(df_daily["Close"].iloc[-1])
        gap = abs(price - last_close) / last_close * 100 if last_close > 0 else 0
        if gap > 15.0:
            # Daily candles move more than 5m ones, so the tolerance is wider —
            # but a live price this far from the last daily close still means
            # the candles and the quote are different markets.
            return SwingPlan(ticker, None, NO_TREND, 0.0, "C", price,
                             reasons=[f"The daily candles end at {_fp(last_close)} but the "
                                      f"live price is {_fp(price)} — {gap:.0f}% apart. "
                                      f"They aren't the same market, so no level here can "
                                      f"be trusted."])

    reasons: List[str] = []
    trend, trend_reason = trend_of(df_daily, params)
    direction = direction_override or trend
    reasons.append(trend_reason)
    score = POINTS_TREND if (trend and trend == direction) else 0

    if direction is None:
        return SwingPlan(ticker, None, NO_TREND, float(score), _grade(score), price,
                         reasons=reasons)

    levels = major_levels(df_daily, params)
    level = nearest_level(levels, direction, price)
    atr = daily_atr(df_daily)
    if level is None or atr is None:
        reasons.append("No level with two or more touches on the side this trade enters from.")
        return SwingPlan(ticker, direction, NO_LEVEL, float(score), _grade(score), price,
                         reasons=reasons)

    score += POINTS_LEVEL
    distance_atr = abs(price - level.price) / atr
    reasons.append(f"{level.strength.title()} {level.kind} at {_fp(level.price)}, touched "
                   f"{level.touches} times — {distance_atr:.1f} daily ATR away "
                   f"({abs(price - level.price) / price * 100:.1f}%).")

    found = pattern_lib.detect(df_daily)
    confirming = [p.name for p in found if p.bias == (
        "bullish" if direction == "Long" else "bearish")]
    if confirming:
        score += POINTS_PATTERN
        reasons.append("Confirming daily candle: " + ", ".join(confirming) + ".")
    elif params.require_pattern:
        reasons.append("No confirming daily candle yet — the level alone isn't the trigger.")

    entry = level.price
    stop = (entry - atr * params.stop_atr if direction == "Long"
            else entry + atr * params.stop_atr)
    target, target_reason = target_for(levels, direction, entry, stop, params.min_rr)
    reasons.append(target_reason)
    rr = (abs(target - entry) / abs(entry - stop)) if target else None

    if distance_atr <= params.near_atr:
        stage = CONFIRMED if confirming else AT_LEVEL
    elif distance_atr <= params.approach_atr:
        stage = APPROACHING
    else:
        stage = NO_LEVEL
        reasons.append(f"The level is {distance_atr:.1f} ATR away — too far to act on yet.")

    plan = SwingPlan(ticker, direction, stage, float(score), _grade(score), price,
                     level=level, entry=entry, stop=stop, target=target, reward_risk=rr,
                     reasons=reasons, patterns=[p.name for p in found])
    plan.features = {
        "direction": direction, "level_touches": level.touches,
        "distance_atr": round(distance_atr, 2),
        "risk_pct": round(abs(entry - stop) / entry * 100, 3) if entry else None,
        "has_pattern": bool(confirming),
        "volatility_pct": round(atr / price * 100, 3),
    }
    return plan


# ---------------------------------------------------------------------
# Backtest
#
# Daily candles go back years, so unlike the 5-minute strategy this can be
# tested over real history rather than a fortnight. Same no-look-ahead rule:
# at each step only candles that had already closed are visible.
# ---------------------------------------------------------------------

@dataclass
class SwingTrade:
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
    r_result: Optional[float] = None
    fill_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    features: Optional[Dict] = None
    exit_reason: str = ""


def run_backtest(df_daily: pd.DataFrame, ticker: str,
                 params: Optional[SwingParams] = None, warmup: int = 120,
                 expiry_days: int = 10, fee_r: float = 0.0,
                 require_confirmation: bool = True,
                 breakeven_at_r: Optional[float] = None,
                 trail_atr_mult: Optional[float] = None) -> List[SwingTrade]:
    """Replay the swing rules over daily history, one trade at a time."""
    from trade_sim import simulate_limit_trade, EXPIRED

    params = params or SwingParams()
    trades: List[SwingTrade] = []
    if df_daily is None or len(df_daily) < warmup + 10:
        return trades

    busy_until = -1
    for t in range(warmup, len(df_daily) - 1):
        if t <= busy_until:
            continue
        past = df_daily.iloc[:t + 1]           # only closed candles
        try:
            plan = analyze(ticker, past, params=params)
        except Exception:
            continue
        ready = plan.stage == CONFIRMED or (not require_confirmation
                                            and plan.stage in (AT_LEVEL, APPROACHING))
        if not (ready and plan.entry and plan.stop and plan.target and plan.reward_risk):
            continue

        future = df_daily.iloc[t + 1:]
        try:
            sim = simulate_limit_trade(future, plan.direction, plan.entry, plan.stop,
                                        plan.target, expiry_bars=expiry_days, fee_r=fee_r,
                                        breakeven_at_r=breakeven_at_r,
                                        trail_atr_mult=trail_atr_mult,
                                        atr_value=daily_atr(past))
        except ValueError:
            continue

        trades.append(SwingTrade(
            ticker=ticker, signal_time=df_daily.index[t], direction=plan.direction,
            score=plan.score, grade=plan.grade, entry=plan.entry, stop=plan.stop,
            target=plan.target, planned_rr=plan.reward_risk, status=sim.status,
            r_result=sim.r_result, fill_time=sim.fill_time, exit_time=sim.exit_time,
            features=plan.features,
            exit_reason=("stop" if sim.status == "LOSS" else
                         "target" if sim.status == "WIN" else "")))

        if sim.exit_time is not None:
            busy_until = df_daily.index.get_loc(sim.exit_time)
        elif sim.status == EXPIRED:
            busy_until = t + expiry_days
        else:
            busy_until = len(df_daily)
    return trades


def scan_universe(instruments, frame_loader, spot_loader=None,
                  params: Optional[SwingParams] = None, progress=None):
    """Rank many instruments on the swing rules.

    instruments:  list of (label, ticker, fetch_key)
    frame_loader: fetch_key -> daily OHLC DataFrame (closed candles only)
    Returns scanner.RankedCandidate rows so the existing results table,
    filters, saving and tracking all work unchanged.
    """
    from scanner import RankedCandidate

    params = params or SwingParams()
    out = []
    total = len(instruments)
    for i, (label, ticker, key) in enumerate(instruments):
        if progress:
            progress(i, total, label)
        try:
            daily = frame_loader(key)
            if daily is None or daily.empty:
                out.append(RankedCandidate(ticker, label, None, 0.0, "—", "unknown",
                                            None, None, None, None,
                                            error="No daily candles returned."))
                continue
            live = None
            if spot_loader is not None:
                try:
                    live = spot_loader(key)
                except Exception:
                    live = None
            plan = analyze(ticker, daily, price=live, params=params)
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

    stage_rank = {CONFIRMED: 0, AT_LEVEL: 1, APPROACHING: 2, NO_LEVEL: 3, NO_TREND: 4}

    def key_fn(r):
        dist = abs(r.entry - r.price) / r.price if (r.entry and r.price) else 9e9
        return (r.error is not None, -r.score, stage_rank.get(r.entry_status, 9), dist)

    out.sort(key=key_fn)
    return out


# Stop-management options, so the choice can be measured rather than argued.
STOP_VARIANTS = {
    "Hard stop, fixed target": {},
    "Breakeven at 1R": {"breakeven_at_r": 1.0},
    "Trail 2x ATR": {"trail_atr_mult": 2.0},
    "Trail 3x ATR": {"trail_atr_mult": 3.0},
    "Breakeven at 1R + trail 3x ATR": {"breakeven_at_r": 1.0, "trail_atr_mult": 3.0},
}


def run_variants(df_daily: pd.DataFrame, ticker: str,
                 params: Optional[SwingParams] = None, warmup: int = 120,
                 expiry_days: int = 10, fee_r: float = 0.0,
                 require_confirmation: bool = True,
                 variants: Optional[Dict[str, Dict]] = None
                 ) -> Dict[str, List[SwingTrade]]:
    """The same strategy, with the stop managed several different ways.

    Trade counts differ slightly between variants, and that is realistic rather
    than a flaw: a trailing stop exits earlier, which frees you to take the next
    setup sooner (only one trade runs at a time). So the fair comparison is the
    TOTAL result over the same history, not a trade-by-trade one.
    """
    variants = variants or STOP_VARIANTS
    return {name: run_backtest(df_daily, ticker, params=params, warmup=warmup,
                                expiry_days=expiry_days, fee_r=fee_r,
                                require_confirmation=require_confirmation, **kw)
            for name, kw in variants.items()}
