"""
Trend Retrace — the trader's own strategy, implemented as written.

LONG (short is the exact reverse):
  1. 4H trend:  the last 2 closed 4H candles each made a higher high AND a
                higher low than the candle before.
  2. 1H confirm: the last closed 1H candle is bullish (closed above its open).
  3. 5m entry:  wait for price to retrace to a previous support on the 5m
                chart, and enter there with a limit order.

RE-ENTRY after being stopped out:
  a. wait for a closed 4H candle in the trade's direction (bullish for a long)
  b. on the 5m, wait for a retrace to support and re-enter 50%
  c. wait for another closed 1H candle in the trade's direction, then add
     the remaining 50%

NOT SPECIFIED BY THE TRADER — defaults chosen here, all adjustable:
  * Stop loss:   below the 5m support used for entry, by a multiple of 5m ATR
                 (alternative: below the low of the last closed 4H candle)
  * Take profit: a fixed multiple of the risk — minimum and default 3R
  * Order expiry: an unfilled limit is cancelled after a day, or sooner if
                 the 4H trend condition stops holding

SCORING: the strategy is a checklist, not a confluence score. Points simply
reflect which of the three rules are met — 4H trend 4, 1H confirmation 3,
5m support entry 3 — so a complete setup is 10/10 and anything short of all
three rules is not a full setup. This is a checklist count, NOT a probability
of profit.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from technical import find_swing_points
from formatting import format_price as _fp

POINTS_4H = 4
POINTS_1H = 3
POINTS_5M = 3

STOP_BELOW_SUPPORT = "support"
STOP_BELOW_4H_CANDLE = "4h_candle"

# Stages, in the order a setup develops.
NO_TREND = "No 4H trend"
WAIT_1H = "Waiting for 1H confirmation"
NO_SUPPORT = "No 5m support found"
DATA_MISMATCH = "Candles don't match the live price"
WAIT_RETRACE = "Waiting for retrace"
AT_ENTRY = "At entry now"
LOW_QUALITY = "Rejected by quality filters"


@dataclass
class StrategyParams:
    trend_candles: int = 2          # consecutive 4H HH+HL candles required
    stop_mode: str = STOP_BELOW_SUPPORT
    # Distance below support, in 5m ATR. 1x proved too tight: trades resolved
    # in ~12 minutes, before any 1H candle could close, so the rule "add the
    # second 50% after a 1H confirmation" could never fire. 3x lets trades run
    # for hours, which is what that re-entry rule implies.
    stop_atr_mult: float = 3.0
    target_r: float = 3.0           # take profit as a multiple of risk (minimum 3)
    support_lookback: int = 288     # 5m bars searched for support (288 = 24h)
    at_entry_atr: float = 0.3       # within this many 5m ATR = "at entry"
    # A live price this far from the last candle means the two are not the same
    # market — stale candles, a wrong symbol, or a bundled "k" coin mixed with a
    # per-coin price. A ZEC plan once showed an entry 71% below the live price
    # for exactly this reason, and a plan built on the wrong prices is worse
    # than no plan.
    max_price_gap_pct: float = 10.0
    swing_left: int = 3
    swing_right: int = 3
    # --- optional quality filters -------------------------------------
    # These go BEYOND the trader's three rules. They exist because two 4H
    # candles and one 1H candle are weak filters on their own: in choppy
    # markets both happen constantly by chance. Each can be switched off, and
    # the backtest measures whether they help. Set to 0 to disable.
    min_h1_body_ratio: float = 0.0   # 1H candle body as a share of its range
    min_trend_move_atr: float = 0.0  # 4H trend move, in 4H ATR
    max_extension_atr: float = 0.0   # distance from the 4H 20-period average


@dataclass
class RuleResult:
    passed: bool
    reason: str


@dataclass
class TrendRetracePlan:
    ticker: str
    direction: Optional[str]
    stage: str
    score: float
    grade: str
    current_price: Optional[float]
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    reward_risk: Optional[float] = None
    rules: Dict[str, RuleResult] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    features: Dict[str, object] = field(default_factory=dict)
    quality_fails: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# The three rules
# ---------------------------------------------------------------------

def four_hour_trend(df_4h: pd.DataFrame, n: int = 2) -> Tuple[Optional[str], str]:
    """Rule 1. Returns ("Long"|"Short"|None, reason), judged on closed candles.

    Needs n+1 candles: each of the last n must beat the one before it.
    """
    if df_4h is None or len(df_4h) < n + 1:
        return None, f"Need at least {n + 1} closed 4H candles."
    recent = df_4h.iloc[-(n + 1):]
    highs, lows = recent["High"].values, recent["Low"].values
    up = all(highs[i] > highs[i - 1] and lows[i] > lows[i - 1] for i in range(1, n + 1))
    down = all(highs[i] < highs[i - 1] and lows[i] < lows[i - 1] for i in range(1, n + 1))
    seq = " → ".join(f"H {_fp(h)} / L {_fp(l)}" for h, l in zip(highs, lows))
    if up:
        return "Long", f"Last {n} 4H candles made higher highs and higher lows ({seq})."
    if down:
        return "Short", f"Last {n} 4H candles made lower highs and lower lows ({seq})."
    return None, f"Last {n} 4H candles are not consistently HH/HL or LH/LL ({seq})."


def one_hour_confirms(df_1h: pd.DataFrame, direction: str) -> RuleResult:
    """Rule 2. The last closed 1H candle closes in the trade's direction."""
    if df_1h is None or df_1h.empty:
        return RuleResult(False, "No closed 1H candle available.")
    last = df_1h.iloc[-1]
    o, c = float(last["Open"]), float(last["Close"])
    if direction == "Long":
        ok = c > o
        word = "bullish" if ok else "not bullish"
    else:
        ok = c < o
        word = "bearish" if ok else "not bearish"
    return RuleResult(ok, f"Last 1H candle is {word} (open {_fp(o)}, close {_fp(c)}).")


def atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = float(tr.rolling(period).mean().iloc[-1])
    return v if np.isfinite(v) and v > 0 else None


def five_minute_support(df_5m: pd.DataFrame, direction: str, price: float,
                         params: StrategyParams) -> Tuple[Optional[float], str]:
    """Rule 3. The nearest previous support below price (resistance above,
    for a short) on the 5m chart, within the lookback window.

    Only CONFIRMED swing points count — a swing needs a few later candles
    before it is known to be a swing at all.
    """
    if df_5m is None or len(df_5m) < params.swing_left + params.swing_right + 1:
        return None, "Not enough 5m candles to find support."
    window = df_5m.iloc[-params.support_lookback:]
    swings = [s for s in find_swing_points(window, params.swing_left, params.swing_right)
              if s.confirmed]
    if direction == "Long":
        levels = sorted([s.price for s in swings if s.kind == "low" and s.price < price],
                        reverse=True)
        if not levels:
            return None, "No previous 5m support below the current price."
        return levels[0], f"Nearest previous 5m support below price is {_fp(levels[0])}."
    levels = sorted([s.price for s in swings if s.kind == "high" and s.price > price])
    if not levels:
        return None, "No previous 5m resistance above the current price."
    return levels[0], f"Nearest previous 5m resistance above price is {_fp(levels[0])}."


def stop_and_target(direction: str, entry: float, df_5m: pd.DataFrame,
                     df_4h: pd.DataFrame, params: StrategyParams
                     ) -> Tuple[Optional[float], Optional[float], str]:
    a = atr(df_5m) or 0.0
    if params.stop_mode == STOP_BELOW_4H_CANDLE and df_4h is not None and not df_4h.empty:
        last = df_4h.iloc[-1]
        stop = float(last["Low"]) if direction == "Long" else float(last["High"])
        how = "beyond the last closed 4H candle"
    else:
        buffer = a * params.stop_atr_mult
        stop = entry - buffer if direction == "Long" else entry + buffer
        how = f"{params.stop_atr_mult:g} x 5m ATR beyond the support"
    risk = (entry - stop) if direction == "Long" else (stop - entry)
    if risk <= 0:
        return None, None, "Stop would sit on the wrong side of entry."
    target = entry + risk * params.target_r if direction == "Long" else entry - risk * params.target_r
    return stop, target, f"Stop {_fp(stop)} ({how}); target {_fp(target)} ({params.target_r:g}R)."


def session_of(ts: pd.Timestamp) -> str:
    h = ts.tz_convert("UTC").hour if ts.tzinfo else ts.hour
    if h < 8:
        return "Asia"
    if h < 13:
        return "Europe"
    if h < 21:
        return "US"
    return "Late US"


def compute_features(direction: str, price: float, entry: float, stop: float,
                      df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_5m: pd.DataFrame,
                      when: pd.Timestamp, n_trend: int = 2) -> Dict[str, object]:
    """Snapshot of a setup at the moment it is signalled.

    Used to learn which KINDS of setup win. Computed from closed candles only,
    and identically in live scanning and the backtest — lessons learned from
    one only transfer to the other if the measurements are the same.
    """
    feats: Dict[str, object] = {}
    a5 = atr(df_5m)
    if df_4h is not None and len(df_4h) >= n_trend + 1:
        first = float(df_4h["High"].iloc[-(n_trend + 1)] if direction == "Long"
                      else df_4h["Low"].iloc[-(n_trend + 1)])
        last = float(df_4h["High"].iloc[-1] if direction == "Long" else df_4h["Low"].iloc[-1])
        if first:
            feats["trend_move_pct"] = abs(last - first) / first * 100
    if a5:
        feats["retrace_depth_atr"] = abs(price - entry) / a5
        feats["volatility_pct"] = a5 / price * 100 if price else None
    if entry:
        feats["risk_pct"] = abs(entry - stop) / entry * 100
    if df_1h is not None and not df_1h.empty:
        c = df_1h.iloc[-1]
        rng = float(c["High"] - c["Low"])
        feats["h1_body_ratio"] = abs(float(c["Close"] - c["Open"])) / rng if rng > 0 else 0.0
    feats["session"] = session_of(when)
    feats["direction"] = direction
    return {k: v for k, v in feats.items() if v is not None}


def quality_check(direction: str, price: float, df_4h: pd.DataFrame,
                   df_1h: pd.DataFrame, params: StrategyParams) -> List[str]:
    """Reasons this setup is weak, beyond the three core rules. Empty = fine."""
    fails: List[str] = []

    if params.min_h1_body_ratio > 0 and df_1h is not None and not df_1h.empty:
        c = df_1h.iloc[-1]
        rng = float(c["High"] - c["Low"])
        body = abs(float(c["Close"] - c["Open"])) / rng if rng > 0 else 0.0
        if body < params.min_h1_body_ratio:
            fails.append(f"The 1H confirmation candle is weak — its body is only "
                         f"{body * 100:.0f}% of its range (need "
                         f"{params.min_h1_body_ratio * 100:.0f}%). A small body means "
                         f"buyers and sellers finished roughly level.")

    if params.min_trend_move_atr > 0 and df_4h is not None and len(df_4h) >= 15:
        a4 = atr(df_4h)
        n = params.trend_candles
        if a4 and len(df_4h) >= n + 1:
            first = float(df_4h["High"].iloc[-(n + 1)] if direction == "Long"
                          else df_4h["Low"].iloc[-(n + 1)])
            last = float(df_4h["High"].iloc[-1] if direction == "Long"
                         else df_4h["Low"].iloc[-1])
            move = abs(last - first) / a4
            if move < params.min_trend_move_atr:
                fails.append(f"The 4H trend is barely a trend — it has moved {move:.2f} ATR "
                             f"over {n} candles (need {params.min_trend_move_atr:.2f}). "
                             f"Marginally higher highs happen by chance in a range.")

    if params.max_extension_atr > 0 and df_4h is not None and len(df_4h) >= 20:
        a4 = atr(df_4h)
        sma = float(df_4h["Close"].tail(20).mean())
        if a4 and sma:
            ext = abs(price - sma) / a4
            if ext > params.max_extension_atr:
                fails.append(f"Price is {ext:.1f} ATR from its 4H 20-period average (limit "
                             f"{params.max_extension_atr:.1f}) — the move is extended, and "
                             f"entering here is chasing.")
    return fails


def _grade(score: float) -> str:
    return "A+" if score >= 8 else "B" if score >= 6 else "C"


# ---------------------------------------------------------------------
# Live analysis
# ---------------------------------------------------------------------

def analyze(ticker: str, df_4h: pd.DataFrame, df_1h: pd.DataFrame,
            df_5m: pd.DataFrame, live_price: Optional[float] = None,
            params: Optional[StrategyParams] = None,
            direction_override: Optional[str] = None,
            extra_features: Optional[Dict[str, object]] = None) -> TrendRetracePlan:
    """Apply the three rules to closed candles and produce a plan.

    The candle frames passed in must contain CLOSED candles only; the caller
    is responsible for dropping a candle that is still forming.
    """
    params = params or StrategyParams()
    price = live_price
    if price is None and df_5m is not None and not df_5m.empty:
        price = float(df_5m["Close"].iloc[-1])
    if price is None or price <= 0:
        return TrendRetracePlan(ticker, None, NO_TREND, 0.0, "C", None,
                                 problems=["No usable price."])

    # Before anything else: do the candles describe the same market as the
    # live price?
    if live_price and df_5m is not None and not df_5m.empty:
        last_close = float(df_5m["Close"].iloc[-1])
        if last_close > 0:
            gap = abs(live_price - last_close) / last_close * 100
            if gap > params.max_price_gap_pct:
                return TrendRetracePlan(
                    ticker, None, DATA_MISMATCH, 0.0, "C", price,
                    problems=[f"The candles end at {_fp(last_close)} but the live price is "
                              f"{_fp(live_price)} — {gap:.0f}% apart. They are not the same "
                              f"market, so no levels can be trusted. Usually stale candles, "
                              f"a symbol that resolved to the wrong coin, or a bundled "
                              f"'k' market priced per 1,000."])

    trend, trend_reason = four_hour_trend(df_4h, params.trend_candles)
    direction = direction_override or trend
    rules: Dict[str, RuleResult] = {}
    rules["4h"] = RuleResult(trend is not None and trend == direction, trend_reason)

    if direction is None:
        return TrendRetracePlan(ticker, None, NO_TREND, 0.0, "C", price, rules=rules)

    rules["1h"] = one_hour_confirms(df_1h, direction)
    level, level_reason = five_minute_support(df_5m, direction, price, params)
    rules["5m"] = RuleResult(level is not None, level_reason)

    score = (POINTS_4H * rules["4h"].passed + POINTS_1H * rules["1h"].passed
             + POINTS_5M * rules["5m"].passed)
    plan = TrendRetracePlan(ticker, direction, NO_TREND, float(score), _grade(score),
                             price, rules=rules)

    plan.quality_fails = quality_check(direction, price, df_4h, df_1h, params)

    if not rules["4h"].passed:
        plan.stage = NO_TREND
    elif not rules["1h"].passed:
        plan.stage = WAIT_1H
    elif level is None:
        plan.stage = NO_SUPPORT
    else:
        a = atr(df_5m) or 0.0
        plan.stage = AT_ENTRY if abs(price - level) <= a * params.at_entry_atr else WAIT_RETRACE

    if plan.quality_fails and plan.stage in (WAIT_RETRACE, AT_ENTRY):
        # The core rules pass, but the setup is weak. Say so rather than
        # presenting it as a full-quality signal.
        plan.stage = LOW_QUALITY
        plan.score = min(plan.score, 7.0)
        plan.grade = _grade(plan.score)

    if level is not None:
        stop, target, how = stop_and_target(direction, level, df_5m, df_4h, params)
        if stop is not None:
            plan.entry, plan.stop, plan.target = level, stop, target
            plan.reward_risk = params.target_r
            # The stop/target explanation belongs to the plan, not the rule —
            # keeping it out of the rule's reason avoids stating it twice.
            _when = df_5m.index[-1] + FIVE_MIN if df_5m is not None and len(df_5m) else \
                pd.Timestamp.now(tz="UTC")
            plan.features = compute_features(direction, price, level, stop, df_4h, df_1h,
                                              df_5m, _when, params.trend_candles)
            if extra_features:
                # Market-wide context (e.g. the Fear & Greed band) recorded with
                # the setup, so the learner can test whether it matters.
                plan.features.update({k: v for k, v in extra_features.items()
                                      if v is not None})
        else:
            plan.problems.append(how)
    return plan


def reentry_plan_text(direction: str) -> List[str]:
    """The trader's re-entry rules, stated as a checklist."""
    word = "bullish" if direction == "Long" else "bearish"
    level = "support" if direction == "Long" else "resistance"
    return [
        f"Wait for a closed 4H candle that is {word} — do not re-enter before it.",
        f"On the 5m, wait for price to retrace to a previous {level}, then re-enter **50%**.",
        f"Wait for another closed 1H candle that is {word}, then add the remaining **50%**.",
    ]


# ---------------------------------------------------------------------
# Backtest — event-driven over 5m candles, including re-entry
#
# Every decision at a 5m bar uses only candles that had CLOSED by the end of
# that bar. Orders fill only on LATER bars. Ambiguous candles resolve to the
# worse outcome (a fill bar that also touches the stop is a loss; a bar that
# touches both stop and target is a loss).
# ---------------------------------------------------------------------

FIVE_MIN = pd.Timedelta(minutes=5)
ONE_HOUR = pd.Timedelta(hours=1)
FOUR_HOURS = pd.Timedelta(hours=4)

KIND_INITIAL = "Initial entry"
KIND_RE1 = "Re-entry (first 50%)"
KIND_RE2 = "Re-entry (second 50%)"


@dataclass
class TRTrade:
    ticker: str
    kind: str
    weight: float                 # share of a full 1R position (1.0 or 0.5)
    direction: str
    signal_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    status: str                   # WIN / LOSS / EXPIRED / OPEN
    r_result: Optional[float] = None       # per unit of this leg's own risk
    fill_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    features: Optional[Dict[str, object]] = None
    exit_reason: str = ""            # "stop" / "target" / "reversal" / "stale" / ""

    @property
    def weighted_r(self) -> Optional[float]:
        return None if self.r_result is None else self.r_result * self.weight


def _closed_index(open_times: np.ndarray, bar_len: pd.Timedelta, t: pd.Timestamp) -> int:
    """Index of the last candle closed by time t, or -1."""
    return int(np.searchsorted(open_times, (t - bar_len).to_datetime64(), side="right")) - 1


def run_backtest(df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                  ticker: str, params: Optional[StrategyParams] = None,
                  expiry_bars: int = 288, reentry_timeout_bars: int = 576,
                  fee_r: float = 0.0, enable_reentry: bool = True,
                  exit_on_reversal: bool = False,
                  exit_stale_hours: Optional[float] = None) -> List[TRTrade]:
    """exit_on_reversal / exit_stale_hours apply the trade review's two early-exit
    rules, so the backtest can measure whether they actually help:
      * reversal — close when the 4H candles start trending the other way
      * stale    — close a trade open this many hours that has gone nowhere
                   (under 0.5R either way, never reached +1R) while the last
                   closed 1H candle runs against it
    Early exits close at the 5m close and do not trigger the re-entry rules,
    which are for stop-outs."""
    params = params or StrategyParams()
    trades: List[TRTrade] = []
    if any(d is None or d.empty for d in (df_5m, df_1h, df_4h)):
        return trades

    t5 = df_5m.index
    o5, h5 = df_5m["Open"].values, df_5m["High"].values
    l5, c5 = df_5m["Low"].values, df_5m["Close"].values
    open_1h = df_1h.index.values
    open_4h = df_4h.index.values
    n = len(df_5m)
    start = max(params.support_lookback, 30)

    state = "FLAT"
    order = None           # dict describing the resting order / open leg(s)

    def _decision_frames(k):
        now = t5[k] + FIVE_MIN
        i4 = _closed_index(open_4h, FOUR_HOURS, now)
        i1 = _closed_index(open_1h, ONE_HOUR, now)
        return now, i4, i1

    def _new_order(k, direction, kind, weight, i4):
        lo = max(0, k - params.support_lookback + 1)
        win5 = df_5m.iloc[lo:k + 1]
        level, _ = five_minute_support(win5, direction, float(c5[k]), params)
        if level is None:
            return None
        stop, target, _ = stop_and_target(direction, level, win5,
                                           df_4h.iloc[:i4 + 1], params)
        if stop is None:
            return None
        i1 = _closed_index(open_1h, ONE_HOUR, t5[k] + FIVE_MIN)
        feats = compute_features(direction, float(c5[k]), level, stop,
                                  df_4h.iloc[:i4 + 1], df_1h.iloc[:i1 + 1] if i1 >= 0
                                  else pd.DataFrame(), win5, t5[k] + FIVE_MIN,
                                  params.trend_candles)
        return {"direction": direction, "kind": kind, "weight": weight,
                "entry": level, "stop": stop, "target": target,
                "placed": k, "signal_time": t5[k] + FIVE_MIN, "features": feats}

    def _record(o, status, r=None, fill=None, exit_=None, why=""):
        trades.append(TRTrade(ticker, o["kind"], o["weight"], o["direction"],
                               o["signal_time"], o["entry"], o["stop"], o["target"],
                               status, r, fill, exit_, o.get("features"), why))

    def _rr(o):
        risk = abs(o["entry"] - o["stop"])
        return abs(o["target"] - o["entry"]) / risk if risk else 0.0

    k = start
    legs: List[dict] = []          # open legs sharing one stop/target
    reentry_dir = None
    reentry_since = None
    stop_time = None

    trend_cache: Dict[int, Optional[str]] = {}
    h1_cache: Dict[Tuple[int, str], bool] = {}

    def _h1_ok(i1, direction):
        key = (i1, direction)
        if key not in h1_cache:
            h1_cache[key] = i1 >= 0 and one_hour_confirms(
                df_1h.iloc[:i1 + 1], direction).passed
        return h1_cache[key]

    while k < n:
        now, i4, i1 = _decision_frames(k)
        if i4 not in trend_cache:
            trend_cache[i4] = (four_hour_trend(df_4h.iloc[:i4 + 1], params.trend_candles)[0]
                               if i4 >= params.trend_candles else None)
        trend = trend_cache[i4]

        if state == "FLAT":
            _quality_ok = True
            if trend and (params.min_h1_body_ratio or params.min_trend_move_atr
                          or params.max_extension_atr):
                _quality_ok = not quality_check(
                    trend, float(c5[k]), df_4h.iloc[:i4 + 1],
                    df_1h.iloc[:i1 + 1] if i1 >= 0 else pd.DataFrame(), params)
            if trend and _quality_ok and _h1_ok(i1, trend):
                order = _new_order(k, trend, KIND_INITIAL, 1.0, i4)
                if order:
                    state = "PENDING"

        elif state in ("PENDING", "RE_PENDING"):
            if k > order["placed"]:
                long = order["direction"] == "Long"
                touched = l5[k] <= order["entry"] if long else h5[k] >= order["entry"]
                if touched:
                    stopped = l5[k] <= order["stop"] if long else h5[k] >= order["stop"]
                    if stopped:
                        _record(order, "LOSS", -1.0 - fee_r, t5[k], t5[k])
                        state, reentry_dir, stop_time = "RE_WAIT_4H", order["direction"], now
                        reentry_since = k
                    else:
                        order["fill"] = t5[k]
                        order["fill_k"] = k
                        order["mfe"] = 0.0
                        legs = [order]
                        state = "IN_TRADE" if state == "PENDING" else "RE_HALF"
                else:
                    # The retrace the strategy waits for usually prints a lower
                    # 4H low, which breaks the "two higher highs" pattern. That
                    # is expected, not a reason to cancel. Cancel only if the
                    # trend has actually REVERSED, or the order has expired.
                    reversed_ = trend is not None and trend != order["direction"]
                    if k - order["placed"] > expiry_bars or reversed_:
                        _record(order, "EXPIRED")
                        state = "FLAT"

        elif state in ("IN_TRADE", "RE_HALF", "RE_FULL"):
            base = legs[0]
            long = base["direction"] == "Long"
            hit_stop = l5[k] <= base["stop"] if long else h5[k] >= base["stop"]
            hit_tgt = h5[k] >= base["target"] if long else l5[k] <= base["target"]
            risk0 = abs(base["entry"] - base["stop"])
            if risk0 > 0:
                fav = (h5[k] - base["entry"]) if long else (base["entry"] - l5[k])
                base["mfe"] = max(base.get("mfe", 0.0), fav / risk0)
            early = None
            if not (hit_stop or hit_tgt):
                cur_r = (((c5[k] - base["entry"]) if long else (base["entry"] - c5[k]))
                         / risk0) if risk0 > 0 else 0.0
                opposite = "Short" if long else "Long"
                if exit_on_reversal and trend is not None and trend != base["direction"]:
                    early = "reversal"
                elif (exit_stale_hours and "fill_k" in base
                      and (k - base["fill_k"]) * 5 / 60 >= exit_stale_hours
                      and abs(cur_r) < 0.5 and base.get("mfe", 0.0) < 1.0
                      and _h1_ok(i1, opposite)):
                    early = "stale"
            if early:
                for leg in legs:
                    lr = abs(leg["entry"] - leg["stop"])
                    mv = (c5[k] - leg["entry"]) if long else (leg["entry"] - c5[k])
                    r_leg = (mv / lr if lr else 0.0) - fee_r
                    _record(leg, "WIN" if r_leg > 0 else "LOSS", r_leg, leg["fill"],
                            t5[k], early)
                legs = []
                state = "FLAT"
            elif hit_stop or hit_tgt:
                for leg in legs:
                    if hit_stop:
                        _record(leg, "LOSS", -1.0 - fee_r, leg["fill"], t5[k], "stop")
                    else:
                        _record(leg, "WIN", _rr(leg) - fee_r, leg["fill"], t5[k], "target")
                legs = []
                if hit_stop and enable_reentry:
                    state, reentry_dir, stop_time, reentry_since = (
                        "RE_WAIT_4H", base["direction"], now, k)
                else:
                    state = "FLAT"
            elif state == "RE_HALF" and i1 >= 0:
                # Add the second 50% once a 1H candle that CLOSED after the
                # first half filled confirms the direction.
                h1_close = df_1h.index[i1] + ONE_HOUR
                if h1_close > base["fill"] and _h1_ok(i1, base["direction"]):
                    price = float(c5[k])
                    second = dict(base)
                    second.update({"kind": KIND_RE2, "entry": price, "fill": t5[k],
                                   "signal_time": now, "fill_k": k})
                    valid = (price > base["stop"]) if long else (price < base["stop"])
                    if valid:
                        legs.append(second)
                    state = "RE_FULL"

        elif state == "RE_WAIT_4H":
            if not enable_reentry:
                state = "FLAT"
            elif trend is not None and trend != reentry_dir:
                state = "FLAT"                         # trend reversed: abandon
            elif k - reentry_since > reentry_timeout_bars:
                state = "FLAT"                         # waited too long
            elif i4 >= 0:
                c4 = df_4h.iloc[i4]
                closed_after_stop = df_4h.index[i4] + FOUR_HOURS > stop_time
                in_dir = (c4["Close"] > c4["Open"]) if reentry_dir == "Long" else \
                         (c4["Close"] < c4["Open"])
                if closed_after_stop and in_dir:
                    order = _new_order(k, reentry_dir, KIND_RE1, 0.5, i4)
                    if order:
                        state = "RE_PENDING"
        k += 1

    # Anything still open at the end of the data.
    for leg in legs:
        _record(leg, "OPEN", None, leg.get("fill"))
    if state in ("PENDING", "RE_PENDING") and order is not None:
        _record(order, "OPEN")
    return trades


@dataclass
class TRStats:
    group: str
    entries: int
    wins: int
    losses: int
    expired: int
    win_rate: Optional[float]
    total_r: float
    avg_r_per_entry: Optional[float]
    luck_range: Optional[float] = None    # ~95% band of total R expected from luck alone

    @property
    def verdict(self) -> str:
        """Is the total distinguishable from luck? Uses roughly two standard
        errors: a total inside +/- luck_range is what chance alone would
        routinely produce over this many trades."""
        if self.luck_range is None or self.entries < 30:
            return "Too few trades to judge"
        if self.total_r > self.luck_range:
            return "Clearly positive"
        if self.total_r < -self.luck_range:
            return "Clearly negative"
        return "Could be luck"


def backtest_stats(trades: List[TRTrade]) -> List[TRStats]:
    out = []
    for group in (KIND_INITIAL, KIND_RE1, KIND_RE2, "All"):
        g = trades if group == "All" else [t for t in trades if t.kind == group]
        resolved = [t for t in g if t.status in ("WIN", "LOSS")]
        wins = sum(t.status == "WIN" for t in resolved)
        total = sum(t.weighted_r for t in resolved if t.weighted_r is not None)
        rs = [t.weighted_r for t in resolved if t.weighted_r is not None]
        luck = None
        if len(rs) >= 2:
            luck = 2.0 * float(np.std(rs, ddof=1)) * float(np.sqrt(len(rs)))
        out.append(TRStats(
            group=group, entries=len(resolved), wins=wins,
            losses=len(resolved) - wins,
            expired=sum(t.status == "EXPIRED" for t in g),
            win_rate=(wins / len(resolved)) if resolved else None,
            total_r=total,
            avg_r_per_entry=(total / len(resolved)) if resolved else None,
            luck_range=luck))
    return out


# ---------------------------------------------------------------------
# Live scanning helpers
# ---------------------------------------------------------------------

BAR_LENGTH = {"5m": FIVE_MIN, "1h": ONE_HOUR, "4h": FOUR_HOURS}


def drop_forming(df: pd.DataFrame, bar: str, now: Optional[pd.Timestamp] = None
                  ) -> pd.DataFrame:
    """Remove the candle that is still forming.

    Exchanges return the current, unfinished candle as the last row. The rules
    are about CLOSED candles — a 1H candle that is green at minute 20 can close
    red — so an unfinished candle must never count as confirmation.
    """
    if df is None or df.empty:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + BAR_LENGTH[bar] <= now]


def scan_universe_tr(instruments, frame_loader, spot_loader=None,
                      params: Optional[StrategyParams] = None,
                      progress=None, now: Optional[pd.Timestamp] = None,
                      extra_features: Optional[Dict[str, object]] = None):
    """Apply the strategy to many instruments and rank them.

    instruments:  list of (label, ticker, fetch_key)
    frame_loader: fetch_key -> {"4h": df, "1h": df, "5m": df}
    Returns scanner.RankedCandidate objects so the existing results table,
    filters and forward tracking work unchanged.
    """
    from scanner import RankedCandidate
    params = params or StrategyParams()
    out = []
    total = len(instruments)
    for i, (label, ticker, key) in enumerate(instruments):
        if progress:
            progress(i, total, label)
        try:
            frames = frame_loader(key) or {}
            f4 = drop_forming(frames.get("4h"), "4h", now)
            f1 = drop_forming(frames.get("1h"), "1h", now)
            f5 = drop_forming(frames.get("5m"), "5m", now)
            if f4 is None or f4.empty or f5 is None or f5.empty:
                out.append(RankedCandidate(ticker, label, None, 0.0, "—", "unknown",
                                            None, None, None, None, error="No data returned."))
                continue
            live = None
            if spot_loader is not None:
                try:
                    live = spot_loader(key)
                except Exception:
                    live = None
            plan = analyze(ticker, f4, f1, f5, live_price=live, params=params,
                            extra_features=extra_features)
            if plan.stage == DATA_MISMATCH:
                # A data problem is not a weak setup — report it as a failure
                # so it can't be mistaken for a tradable plan.
                out.append(RankedCandidate(ticker, label, None, 0.0, "—", plan.stage,
                                            plan.current_price, None, None, None,
                                            error="; ".join(plan.problems)))
                continue
            out.append(RankedCandidate(
                ticker=ticker, label=label, direction=plan.direction, score=plan.score,
                grade=plan.grade, regime=plan.stage, price=plan.current_price,
                stop=plan.stop, target=plan.target, reward_risk=plan.reward_risk,
                note="; ".join(r.reason for r in plan.rules.values()),
                entry_status=plan.stage, entry=plan.entry,
                price_is_live=live is not None, features=plan.features or None))
        except Exception as e:
            out.append(RankedCandidate(ticker, label, None, 0.0, "—", "unknown",
                                        None, None, None, None,
                                        error=f"{type(e).__name__}: {e}"))
    if progress:
        progress(total, total, "done")

    stage_rank = {AT_ENTRY: 0, WAIT_RETRACE: 1, NO_SUPPORT: 2, WAIT_1H: 3, NO_TREND: 4}

    def key(r):
        dist = abs(r.entry - r.price) / r.price if (r.entry and r.price) else 9e9
        return (r.error is not None, -r.score, stage_rank.get(r.entry_status, 9), dist)

    out.sort(key=key)
    return out


def five_minute_levels(df_5m: pd.DataFrame, direction: str, price: float,
                        params: Optional[StrategyParams] = None,
                        limit: int = 6) -> List[float]:
    """Every confirmed 5m support below price (resistance above, for a short),
    nearest first. Used to build a laddered entry; the single-entry rule uses
    only the first of these."""
    params = params or StrategyParams()
    if df_5m is None or len(df_5m) < params.swing_left + params.swing_right + 1:
        return []
    window = df_5m.iloc[-params.support_lookback:]
    swings = [s for s in find_swing_points(window, params.swing_left, params.swing_right)
              if s.confirmed]
    kind = "low" if direction == "Long" else "high"
    levels = [s.price for s in swings if s.kind == kind
              and (s.price < price if direction == "Long" else s.price > price)]
    levels.sort(reverse=(direction == "Long"))
    return levels[:limit]


def frames_from_5m(df_5m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build the 1H and 4H frames by aggregating 5m candles.

    One request instead of three. The result is identical data — a 4H candle is
    by definition the open, high, low and close of its 48 five-minute candles —
    so nothing is approximated. It simply removes two round trips per coin,
    which is most of the scan time when a venue rate-limits requests.

    Needs enough history: 15 complete 4H candles (for ATR) is 720 5m candles.
    """
    if df_5m is None or df_5m.empty:
        return {"5m": pd.DataFrame(), "1h": pd.DataFrame(), "4h": pd.DataFrame()}
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df_5m.columns:
        agg["Volume"] = "sum"
    one_h = df_5m.resample("1h", label="left", closed="left").agg(agg).dropna(subset=["Close"])
    four_h = df_5m.resample("4h", label="left", closed="left").agg(agg).dropna(subset=["Close"])
    return {"5m": df_5m, "1h": one_h, "4h": four_h}
