"""
Automated candidate analysis: derives the scoring evidence and entry-sequence
evidence from ACTUAL multi-timeframe price data, so the user doesn't have to
tick every box by hand.

HONEST SCOPE NOTE — read this before trusting the output:

This is rule-based structural analysis computed from real OHLC candles. It is
NOT discretionary chart reading and it is NOT an oracle. Specifically:

  * "Liquidity" here means equal-highs/equal-lows clustering and prior swing
    levels. There is no order-book, volume-profile or order-flow data behind
    it, so a "sweep" is a close-based rule, not observed liquidation activity.
  * 4H candles are RESAMPLED from 1H data, because the provider does not serve
    a native 4H interval. Resampled bars can differ slightly from an exchange's
    own 4H candles (session boundary alignment).
  * Every derived level is computed from confirmed swing points only — nothing
    is invented — but a rule that says "this is support" is still a heuristic,
    not a fact about where other traders will act.
  * Auto-derived evidence is a STARTING POINT. Every field is overridable in
    the UI, and disagreement between you and the scanner is information, not
    an error.

Each piece of evidence returned carries a plain-English reason so you can audit
why it was set, rather than trusting a bare True/False.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from technical import (find_swing_points, label_structure, find_equal_levels,
                        fib_levels, SwingPoint)
from formatting import format_price as _fp


@dataclass
class EvidenceItem:
    value: Optional[bool]   # True / False / None (couldn't evaluate)
    reason: str             # why it was set this way


@dataclass
class CandidateAnalysis:
    ticker: str
    direction: Optional[str]                  # suggested "Long" / "Short" / None
    regime_1d: str                            # "bullish" / "bearish" / "ranging" / "unknown"
    current_price: Optional[float]
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    reward_risk: Optional[float]
    score_evidence: Dict[str, EvidenceItem] = field(default_factory=dict)
    sequence_evidence: Dict[str, EvidenceItem] = field(default_factory=dict)
    key_levels: Dict[str, float] = field(default_factory=dict)
    data_problems: List[str] = field(default_factory=list)
    level_reason: str = ""
    entry_plan: Optional["EntryPlan"] = None
    entry_is_planned: bool = False   # True when entry came from a zone, not spot
    alternative_zones: list = field(default_factory=list)
    price_source: Optional[str] = None
    price_time = None

    def score_dict(self) -> Dict[str, Optional[bool]]:
        return {k: v.value for k, v in self.score_evidence.items()}

    def sequence_dict(self) -> Dict[str, bool]:
        return {k: bool(v.value) for k, v in self.sequence_evidence.items()}


# ---------------------------------------------------------------------
# Data fetching / resampling
# ---------------------------------------------------------------------

def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Build 4H candles from 1H data. The provider has no native 4H interval,
    so this is a genuine resample — noted in the scope warning above.

    pandas changed its frequency alias casing in 2.2 ('4H' -> '4h'), and each
    version rejects the other spelling, so both are attempted.
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df_1h.columns:
        agg["Volume"] = "sum"
    last_error = None
    for freq in ("4h", "4H"):
        try:
            return df_1h.resample(freq).agg(agg).dropna(subset=["Close"])
        except ValueError as e:
            last_error = e
    raise ValueError(f"Could not resample to 4H with either alias: {last_error}")


def fetch_multi_timeframe(ticker: str, yf_module, ticker_builder=None) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Fetch 1D / 1H (and derive 4H). Returns (frames, problems).

    `ticker_builder` lets the caller inject a session-aware Ticker factory
    (and lets tests inject a fake) instead of hitting the network.
    """
    problems: List[str] = []
    frames: Dict[str, pd.DataFrame] = {}

    def _build(sym):
        if ticker_builder is not None:
            return ticker_builder(sym)
        return yf_module.Ticker(sym)

    specs = [("1d", "1y", "1d"), ("1h", "60d", "1h"), ("5m", "1d", "5m")]
    for label, period, interval in specs:
        try:
            t = _build(ticker)
            try:
                df = t.history(period=period, interval=interval)
            except TypeError:
                df = t.history(period=period)
            if df is None or df.empty:
                problems.append(f"No {label} data returned.")
                frames[label] = pd.DataFrame()
                continue
            if "Close" in df.columns:
                df = df[df["Close"].notna()]
            frames[label] = df
        except Exception as e:
            problems.append(f"{label} fetch failed: {type(e).__name__}: {e}")
            frames[label] = pd.DataFrame()

    frames["4h"] = resample_to_4h(frames.get("1h", pd.DataFrame()))
    if frames["4h"].empty and not frames.get("1h", pd.DataFrame()).empty:
        problems.append("Could not resample 1H data into 4H candles.")

    return frames, problems


# ---------------------------------------------------------------------
# Analysis primitives
# ---------------------------------------------------------------------

def detect_regime(df_1d: pd.DataFrame) -> Tuple[str, str]:
    """1D regime from moving-average alignment plus swing structure.
    Returns (regime, reason)."""
    if df_1d is None or len(df_1d) < 60:
        return "unknown", "Not enough daily history to judge regime (need ~60 bars)."

    close = df_1d["Close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    price = float(close.iloc[-1])

    swings = find_swing_points(df_1d, left=3, right=3)
    labeled = label_structure(swings)
    recent = [l for l in labeled if l.label][-4:]
    bullish_structure = sum(1 for l in recent if l.label in ("HH", "HL"))
    bearish_structure = sum(1 for l in recent if l.label in ("LH", "LL"))

    ma_bullish = price > sma20 > sma50
    ma_bearish = price < sma20 < sma50

    if ma_bullish and bullish_structure >= bearish_structure:
        return "bullish", (f"Price {_fp(price)} above SMA20 {_fp(sma20)} above SMA50 {_fp(sma50)}, "
                            f"with {bullish_structure}/{len(recent)} recent swings higher.")
    if ma_bearish and bearish_structure >= bullish_structure:
        return "bearish", (f"Price {_fp(price)} below SMA20 {_fp(sma20)} below SMA50 {_fp(sma50)}, "
                            f"with {bearish_structure}/{len(recent)} recent swings lower.")
    return "ranging", (f"No clean MA stack (price {_fp(price)}, SMA20 {_fp(sma20)}, SMA50 {_fp(sma50)}) "
                        f"or mixed swing structure — treat as ranging/transitioning.")


def structure_direction(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Judge whether a timeframe's swing structure supports Long or Short."""
    if df is None or len(df) < 20:
        return None, "Not enough bars to read structure."
    swings = find_swing_points(df, left=2, right=2)
    labeled = [l for l in label_structure(swings) if l.label]
    if len(labeled) < 2:
        return None, "Too few labelled swings to read structure."
    recent = labeled[-4:]
    up = sum(1 for l in recent if l.label in ("HH", "HL"))
    down = sum(1 for l in recent if l.label in ("LH", "LL"))
    seq = ", ".join(l.label for l in recent)
    if up > down:
        return "Long", f"Recent swing sequence [{seq}] favours longs."
    if down > up:
        return "Short", f"Recent swing sequence [{seq}] favours shorts."
    return None, f"Recent swing sequence [{seq}] is mixed — no structural bias."


def nearest_level(df: pd.DataFrame, price: float, kind: str,
                   tolerance_pct: float = 0.02) -> Optional[Tuple[float, int]]:
    """Nearest confirmed swing level of `kind` within tolerance of price.
    Returns (level_price, touch_count) or None."""
    if df is None or len(df) < 10 or price <= 0:
        return None
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed and s.kind == kind]
    if not swings:
        return None
    clusters = find_equal_levels(swings, tolerance_pct=0.003)
    candidates = [(c["price_avg"], c["touches"]) for c in clusters if c["kind"] == kind]
    candidates += [(s.price, 1) for s in swings]
    in_range = [(lvl, touches) for lvl, touches in candidates
                if abs(lvl - price) / price <= tolerance_pct]
    if not in_range:
        return None
    return min(in_range, key=lambda x: abs(x[0] - price))


def detect_sweep_and_reclaim(df: pd.DataFrame, direction: str,
                              lookback: int = 30) -> Tuple[Optional[bool], str]:
    """Close-based sweep + reclaim detection.

    For a Long: a recent bar's LOW pierced a prior confirmed swing low, but the
    bar CLOSED back above it (wick-only rejection is explicitly not enough on
    its own — the close must reclaim). Inverse for a Short.
    """
    if df is None or len(df) < lookback + 5:
        return None, "Not enough bars to evaluate a sweep/reclaim."

    window = df.iloc[-lookback:]
    prior = df.iloc[:-lookback]
    if prior.empty:
        return None, "No prior history to define the swept level."

    swings = [s for s in find_swing_points(prior, left=2, right=2) if s.confirmed]
    if direction == "Long":
        levels = [s.price for s in swings if s.kind == "low"]
        if not levels:
            return None, "No confirmed prior swing low to sweep."
        level = max(levels)  # most recent relevant support region
        pierced = window["Low"].min() < level
        reclaimed = float(window["Close"].iloc[-1]) > level
        if pierced and reclaimed:
            return True, (f"Low {_fp(window['Low'].min())} pierced prior swing low {_fp(level)}, "
                           f"and price closed back above at {_fp(window['Close'].iloc[-1])} — reclaim confirmed.")
        if pierced and not reclaimed:
            return False, (f"Level {_fp(level)} was pierced but price has NOT closed back above it "
                            f"(last close {_fp(window['Close'].iloc[-1])}) — no reclaim, so no points.")
        return False, f"No sweep of prior swing low {_fp(level)} in the last {lookback} bars."
    else:
        levels = [s.price for s in swings if s.kind == "high"]
        if not levels:
            return None, "No confirmed prior swing high to sweep."
        level = min(levels)
        pierced = window["High"].max() > level
        rejected = float(window["Close"].iloc[-1]) < level
        if pierced and rejected:
            return True, (f"High {_fp(window['High'].max())} pierced prior swing high {_fp(level)}, "
                           f"and price closed back below at {_fp(window['Close'].iloc[-1])} — rejection confirmed.")
        if pierced and not rejected:
            return False, (f"Level {_fp(level)} was pierced but price has NOT closed back below it "
                            f"(last close {_fp(window['Close'].iloc[-1])}) — no rejection, so no points.")
        return False, f"No sweep of prior swing high {_fp(level)} in the last {lookback} bars."


def fib_confluence(df: pd.DataFrame, price: float, direction: str,
                    tolerance_pct: float = 0.015) -> Tuple[Optional[bool], str, Dict[str, float]]:
    """Check whether price sits near a Fibonacci level drawn between the two
    most recent confirmed opposing swings."""
    if df is None or len(df) < 20 or price <= 0:
        return None, "Not enough data for Fibonacci anchors.", {}
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed]
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if not highs or not lows:
        return None, "No confirmed swing high/low pair to anchor Fibonacci.", {}

    last_high = max(highs, key=lambda s: s.index)
    last_low = max(lows, key=lambda s: s.index)
    if last_high.price <= last_low.price:
        return None, "Swing anchors are degenerate (high not above low).", {}

    try:
        levels = fib_levels(last_low.price, last_high.price, direction)
    except ValueError as e:
        return None, f"Could not compute Fibonacci levels: {e}", {}

    near = [(name, lvl) for name, lvl in levels.items()
            if lvl > 0 and abs(lvl - price) / price <= tolerance_pct]
    if near:
        name, lvl = min(near, key=lambda x: abs(x[1] - price))
        return True, (f"Price {_fp(price)} is within {tolerance_pct:.1%} of {name} at {_fp(lvl)} "
                       f"(anchored on swing low {_fp(last_low.price)} / high {_fp(last_high.price)})."), levels
    return False, (f"Price {_fp(price)} is not near any Fibonacci level from the latest confirmed "
                    f"swing pair ({_fp(last_low.price)} → {_fp(last_high.price)})."), levels


def atr_value(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Average True Range, or None if it can't be computed."""
    if df is None or len(df) < period + 1:
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                     (low - prev_close).abs()], axis=1).max(axis=1)
    val = float(tr.rolling(period).mean().iloc[-1])
    return val if np.isfinite(val) and val > 0 else None


def derive_levels(df_4h: pd.DataFrame, entry: float, direction: str,
                   min_rr: float = 2.0, stop_buffer_atr: float = 0.25,
                   current_price: Optional[float] = None
                   ) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Pick a structural stop and a target that actually clears min_rr.

    The stop is the nearest confirmed swing beyond entry, pushed out by a
    fraction of ATR so it sits *past* the level rather than exactly on it
    (resting precisely on an obvious swing is where stop hunts live).

    The target is NOT simply the nearest opposing swing — doing that forces
    reward:risk to roughly 1:1 no matter what the chart looks like, which
    makes the 2:1 requirement unreachable by construction. Instead the
    candidate levels beyond entry are walked outward and the first one that
    meets min_rr is chosen. If none qualifies, the furthest real level is
    returned along with its true (sub-minimum) R:R, so the setup fails the
    R:R test honestly rather than being handed a fabricated target.

    Returns (stop, target, reward_risk, explanation).
    """
    if df_4h is None or df_4h.empty or entry <= 0 or direction not in ("Long", "Short"):
        return None, None, None, "No usable 4H data or direction for level derivation."

    swings = [s for s in find_swing_points(df_4h, left=2, right=2) if s.confirmed]
    if not swings:
        return None, None, None, "No confirmed 4H swing points available."

    atr = atr_value(df_4h) or 0.0
    buffer = atr * stop_buffer_atr

    # Targets: confirmed swings PLUS the live extreme of the current leg (the
    # obvious retest objective for a pullback entry) and Fibonacci extensions
    # beyond it. Stops still come only from confirmed swings — invalidation
    # must rest on structure that has actually been established.
    extreme = recent_extreme(df_4h, direction)
    fibs = _fib_levels_for(df_4h, direction)
    extensions = [v for k, v in fibs.items() if k.startswith("ext_")]

    if direction == "Long":
        # Invalidation for a long is the last confirmed swing LOW below entry —
        # the higher low that the trend depends on. An old swing high below
        # entry is not structural invalidation for this trade.
        lows = sorted([s.price for s in swings if s.kind == "low" and s.price < entry],
                      reverse=True)
        if not lows:
            return None, None, None, "No confirmed swing low below entry to anchor a stop."
        stop = lows[0] - buffer
        pool = [s.price for s in swings if s.price > entry] + extensions
        if extreme is not None:
            pool.append(extreme)
        # A target must be a level price has NOT already cleared. With a planned
        # entry below the live price, every level between the two was broken on
        # the way up — it is support now, not resistance, so it cannot be where
        # you take profit. Targets therefore start from the live price upward.
        floor = max(entry, current_price) if current_price else entry
        candidates = sorted({round(p, 12) for p in pool if p > floor})
    else:
        highs = sorted([s.price for s in swings if s.kind == "high" and s.price > entry])
        if not highs:
            return None, None, None, "No confirmed swing high above entry to anchor a stop."
        stop = highs[0] + buffer
        pool = [s.price for s in swings if s.price < entry] + extensions
        if extreme is not None:
            pool.append(extreme)
        ceiling = min(entry, current_price) if current_price else entry
        candidates = sorted({round(p, 12) for p in pool if p < ceiling}, reverse=True)

    risk = abs(entry - stop)
    if risk <= 0:
        return None, None, None, "Degenerate stop distance (zero risk)."

    if not candidates:
        return stop, None, None, (
            f"Stop at {_fp(stop)} (nearest confirmed swing plus {stop_buffer_atr:g} ATR buffer), "
            f"but no confirmed opposing swing beyond entry to use as a target."
        )

    for level in candidates:
        rr = abs(level - entry) / risk
        if rr >= min_rr:
            return stop, level, rr, (
                f"Stop {_fp(stop)} (last confirmed swing {'low' if direction == 'Long' else 'high'} "
                f"±{stop_buffer_atr:g} ATR). Target {_fp(level)} is the first objective "
                f"(prior swing, current-leg extreme, or Fib extension) clearing "
                f"{min_rr:.1f}:1, giving {rr:.2f}:1."
            )

    furthest = candidates[-1]
    rr = abs(furthest - entry) / risk
    return stop, furthest, rr, (
        f"Stop {_fp(stop)}. No confirmed level reaches {min_rr:.1f}:1 — the furthest available "
        f"({_fp(furthest)}) gives only {rr:.2f}:1, so this setup fails the reward:risk test."
    )


def is_extended(df: pd.DataFrame, price: float, atr_mult: float = 2.5,
                 period: int = 14) -> Tuple[Optional[bool], str]:
    """Flag 'chasing an extended move': price stretched far from its 20-SMA in
    ATR terms, i.e. entering after the move rather than at a location."""
    if df is None or len(df) < period + 20:
        return None, "Not enough bars to judge extension."
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                    axis=1).max(axis=1)
    atr = float(tr.rolling(period).mean().iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    if atr <= 0 or not np.isfinite(atr):
        return None, "ATR unavailable — cannot judge extension."
    distance_atr = abs(price - sma20) / atr
    if distance_atr >= atr_mult:
        return True, (f"Price is {distance_atr:.1f} ATR from its 20-SMA ({_fp(sma20)}) — "
                       f"that is an extended move; entering here is chasing.")
    return False, f"Price is {distance_atr:.1f} ATR from its 20-SMA — not extended."


def opposing_liquidity_ahead(df: pd.DataFrame, price: float, target: Optional[float],
                              direction: str) -> Tuple[Optional[bool], str]:
    """Is there a clustered equal-high/low pool between price and the target?"""
    if df is None or target is None or price <= 0:
        return None, "No target defined — cannot check for liquidity in the path."
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed]
    clusters = find_equal_levels(swings, tolerance_pct=0.003)
    if not clusters:
        return False, "No clustered equal highs/lows detected in the path to target."

    if direction == "Long":
        blocking = [c for c in clusters if c["kind"] == "high" and price < c["price_avg"] < target]
    else:
        blocking = [c for c in clusters if c["kind"] == "low" and target < c["price_avg"] < price]

    if blocking:
        nearest = min(blocking, key=lambda c: abs(c["price_avg"] - price))
        return True, (f"A cluster of {nearest['touches']} equal {nearest['kind']}s sits at "
                       f"{_fp(nearest['price_avg'])}, between entry and target — price may stall there.")
    return False, "No clustered opposing liquidity between entry and target."


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def analyze_candidate(ticker: str, frames: Dict[str, pd.DataFrame],
                       direction_override: Optional[str] = None,
                       min_rr: float = 2.0,
                       data_problems: Optional[List[str]] = None,
                       live_price: Optional[float] = None) -> CandidateAnalysis:
    """Derive score + sequence evidence from real multi-timeframe data."""
    problems = list(data_problems or [])
    df_1d = frames.get("1d", pd.DataFrame())
    df_4h = frames.get("4h", pd.DataFrame())
    df_1h = frames.get("1h", pd.DataFrame())

    price = None
    price_source = None
    price_time = None
    for label, df in (("5m", frames.get("5m")), ("1h", df_1h), ("4h", df_4h), ("1d", df_1d)):
        if df is not None and not df.empty:
            price = float(df["Close"].iloc[-1])
            price_source = label
            try:
                price_time = df.index[-1].to_pydatetime()
            except Exception:
                price_time = None
            break

    # A genuine live spot price beats the last candle close — a 4H close can
    # be up to four hours old.
    if live_price is not None and live_price > 0:
        price = float(live_price)
        price_source = "live spot"

    regime, regime_reason = detect_regime(df_1d)
    regime_from_4h = False
    if regime == "unknown" and df_4h is not None and len(df_4h) >= 60:
        # Daily data missing (e.g. a rate-limited request). Rather than letting
        # that cascade into "no direction -> everything unevaluated -> 0/10",
        # read the higher-timeframe bias from 4H and say so plainly. This is a
        # weaker read than a true daily regime and is labelled as such.
        alt_regime, alt_reason = detect_regime(df_4h)
        if alt_regime != "unknown":
            regime = alt_regime
            regime_from_4h = True
            regime_reason = (f"[Daily data unavailable — regime inferred from 4H instead, "
                              f"which is a weaker read.] {alt_reason}")
            problems.append(
                "1D candles were unavailable, so the regime was inferred from 4H. "
                "Treat the regime-alignment component with extra caution."
            )
    dir_4h, dir_4h_reason = structure_direction(df_4h)
    dir_1h, dir_1h_reason = structure_direction(df_1h)

    # Direction: explicit override wins; otherwise regime, then 4H structure.
    if direction_override:
        direction = direction_override
    elif regime == "bullish":
        direction = "Long"
    elif regime == "bearish":
        direction = "Short"
    else:
        direction = dir_4h

    analysis = CandidateAnalysis(
        ticker=ticker, direction=direction, regime_1d=regime, current_price=price,
        entry=None, stop=None, target=None, reward_risk=None, data_problems=problems,
    )
    analysis.price_source = price_source
    analysis.price_time = price_time

    if price is None:
        problems.append("No usable price on any timeframe — analysis cannot proceed.")
        return analysis
    if direction is None:
        problems.append("No directional bias could be determined (regime ranging, structure mixed).")

    # --- derive levels from confirmed structure -----------------------
    entry = price
    entry_plan = None
    entry_is_planned = False
    stop = target = rr = None
    level_reason = "No direction — levels not derived."

    if direction:
        # Prefer a PLANNED entry at a confluence zone over entering at market.
        # Reward:risk measured from the zone is the reward:risk the setup
        # actually offers; measured from spot it is just an accident of timing.
        # Consider several candidate zones and pick the first (in rank order)
        # whose entry actually clears the minimum reward:risk. The nearest zone
        # is not automatically the right one: if a shallow pullback does not
        # offer 2:1, the strategy's answer is to wait for a deeper one — not to
        # take the shallow one anyway.
        zones = [z for z in derive_entry_zones(df_4h, price, direction, limit=5)
                 if z.status != ENTRY_MISSED]
        evaluated = []
        for z in zones:
            zs, zt, zrr, zreason = derive_levels(df_4h, z.reference, direction,
                                                  min_rr=min_rr, current_price=price)
            evaluated.append((z, zs, zt, zrr, zreason))

        chosen = next((e for e in evaluated if e[3] is not None and e[3] >= min_rr), None)
        if chosen is None and evaluated:
            # Nothing clears the bar — show the zone with the best R:R so the
            # shortfall is visible, rather than hiding the plan entirely.
            scored = [e for e in evaluated if e[3] is not None]
            chosen = max(scored, key=lambda e: e[3]) if scored else evaluated[0]

        if chosen is not None:
            entry_plan, stop, target, rr, level_reason = chosen
            entry = entry_plan.reference
            entry_is_planned = True
            nearest = zones[0] if zones else None
            if nearest is not None and nearest is not entry_plan:
                level_reason = (
                    f"The nearest zone ({_fp(nearest.reference)}) did not offer "
                    f"{min_rr:.1f}:1, so a deeper zone was chosen. {level_reason}")
            analysis.alternative_zones = [e[0] for e in evaluated if e[0] is not entry_plan]
        else:
            stop, target, rr, level_reason = derive_levels(df_4h, entry, direction,
                                                            min_rr=min_rr,
                                                            current_price=price)

    analysis.entry, analysis.stop, analysis.target, analysis.reward_risk = entry, stop, target, rr
    analysis.level_reason = level_reason
    analysis.entry_plan = entry_plan
    analysis.entry_is_planned = entry_is_planned
    if entry_plan is not None:
        analysis.key_levels["entry_zone_low"] = entry_plan.zone_low
        analysis.key_levels["entry_zone_high"] = entry_plan.zone_high
    if stop is not None:
        analysis.key_levels["structural_stop"] = stop
    if target is not None:
        analysis.key_levels["structural_target"] = target

    # --- scoring evidence ---------------------------------------------
    ev: Dict[str, EvidenceItem] = {}

    if regime == "unknown" or direction is None:
        ev["regime_alignment_1d_4h"] = EvidenceItem(None, f"{regime_reason} {dir_4h_reason}")
    elif regime_from_4h:
        # Alignment means two INDEPENDENT timeframes agree. When the "daily"
        # regime was itself read from 4H, checking it against 4H structure is
        # comparing a thing with itself — and the 4H structure point below
        # already rewards that same observation. Awarding both double-counted
        # one 4H uptrend as three points, which inflated universe-scan grades.
        ev["regime_alignment_1d_4h"] = EvidenceItem(
            None,
            "Daily data unavailable, so 1D/4H alignment cannot be independently "
            "confirmed. Not scored — 4H structure is already counted separately.")
    else:
        aligned = ((regime == "bullish" and direction == "Long" and dir_4h == "Long")
                   or (regime == "bearish" and direction == "Short" and dir_4h == "Short"))
        ev["regime_alignment_1d_4h"] = EvidenceItem(
            aligned, f"1D: {regime_reason} 4H: {dir_4h_reason}"
        )

    sr = nearest_level(df_4h, price, "low" if direction == "Long" else "high") if direction else None
    if direction is None:
        ev["support_resistance"] = EvidenceItem(None, "No direction — cannot pick the relevant level.")
    elif sr:
        lvl, touches = sr
        ev["support_resistance"] = EvidenceItem(
            True, f"Price is within 2% of a confirmed 4H level at {_fp(lvl)} ({touches} touch(es))."
        )
    else:
        ev["support_resistance"] = EvidenceItem(
            False, "Price is not near a confirmed 4H swing level (within 2%)."
        )

    if direction:
        swept, sweep_reason = detect_sweep_and_reclaim(df_4h, direction)
        ev["liquidity_sweep_reclaim"] = EvidenceItem(swept, sweep_reason)
    else:
        ev["liquidity_sweep_reclaim"] = EvidenceItem(None, "No direction — cannot evaluate a sweep.")

    if direction:
        fib_ok, fib_reason, fib_lv = fib_confluence(df_4h, price, direction)
        ev["fib_confluence"] = EvidenceItem(fib_ok, fib_reason)
        for k, v in fib_lv.items():
            analysis.key_levels[f"fib_{k}"] = v
    else:
        ev["fib_confluence"] = EvidenceItem(None, "No direction — cannot anchor Fibonacci.")

    if dir_4h is None:
        ev["structure_4h_supports"] = EvidenceItem(None, dir_4h_reason)
    else:
        ev["structure_4h_supports"] = EvidenceItem(dir_4h == direction, dir_4h_reason)

    if dir_1h is None:
        ev["entry_confirmation_1h"] = EvidenceItem(None, dir_1h_reason)
    else:
        ev["entry_confirmation_1h"] = EvidenceItem(dir_1h == direction, dir_1h_reason)

    if stop is None:
        ev["invalidation_defined"] = EvidenceItem(
            False, "No confirmed swing level available to place a structural stop."
        )
    else:
        ev["invalidation_defined"] = EvidenceItem(
            True, f"Structural invalidation at {_fp(stop)} (nearest confirmed 4H swing)."
        )

    if rr is None:
        ev["rr_at_least_2"] = EvidenceItem(
            None, "Reward:risk not computable — entry, stop or target missing."
        )
    else:
        ev["rr_at_least_2"] = EvidenceItem(
            rr >= min_rr,
            f"Reward:risk {rr:.2f}:1 vs {min_rr:.1f}:1 minimum. {level_reason}"
        )

    if direction:
        opp, opp_reason = opposing_liquidity_ahead(df_4h, price, target, direction)
        ev["opposing_liquidity_ahead"] = EvidenceItem(opp, opp_reason)
    else:
        ev["opposing_liquidity_ahead"] = EvidenceItem(None, "No direction — cannot check the path.")

    extended, ext_reason = is_extended(df_4h, price)
    if entry_plan is not None and entry_plan.status == ENTRY_MISSED:
        ev["chasing_extended_move"] = EvidenceItem(
            True, f"Price has already run through the entry zone. {ext_reason}")
    else:
        ev["chasing_extended_move"] = EvidenceItem(extended, ext_reason)

    analysis.score_evidence = ev

    # --- entry-sequence evidence --------------------------------------
    seq: Dict[str, EvidenceItem] = {}
    if entry_plan is None:
        seq["location"] = EvidenceItem(
            False, "No confluence zone could be derived from confirmed structure.")
    else:
        seq["location"] = EvidenceItem(
            entry_plan.status == ENTRY_AT_ZONE, entry_plan.rationale)
    seq["liquidity_event"] = EvidenceItem(
        bool(ev["liquidity_sweep_reclaim"].value),
        ev["liquidity_sweep_reclaim"].reason
    )
    seq["reclaim_or_rejection"] = EvidenceItem(
        bool(ev["liquidity_sweep_reclaim"].value),
        "Reclaim/rejection is part of the sweep test above (close-based, not wick-only)."
    )
    seq["confirmation"] = EvidenceItem(
        bool(ev["entry_confirmation_1h"].value), dir_1h_reason
    )
    seq["structural_invalidation"] = EvidenceItem(
        bool(ev["invalidation_defined"].value), ev["invalidation_defined"].reason
    )
    seq["acceptable_rr"] = EvidenceItem(
        bool(ev["rr_at_least_2"].value), ev["rr_at_least_2"].reason
    )
    seq["execution_plan"] = EvidenceItem(
        stop is not None and target is not None,
        "Entry, stop and target are all defined."
        if (stop is not None and target is not None)
        else "Entry, stop and/or target could not be derived from structure."
    )
    analysis.sequence_evidence = seq

    return analysis


# ---------------------------------------------------------------------
# Universe scanning: rank many instruments instead of analysing one.
#
# HONEST TRADE-OFF: a full multi-timeframe analysis costs three API calls per
# instrument, so scanning 90 coins would need ~270 calls and be rate-limited
# into uselessness. A universe scan therefore fetches ONLY the 4H frame (one
# call each) and reads the regime from it. That means:
#
#   * "1D/4H regime alignment" is judged from 4H alone — a weaker read
#   * "Entry confirmation (1H)" cannot be evaluated at all, so that point is
#     never awarded and the practical ceiling is 9/10, not 10/10
#
# Ranked results are a SHORTLIST, not a verdict. Re-run a full single-instrument
# scan on anything that looks interesting before acting on it.
# ---------------------------------------------------------------------

@dataclass
class RankedCandidate:
    ticker: str
    label: str
    direction: Optional[str]
    score: float
    grade: str
    regime: str
    price: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    reward_risk: Optional[float]
    note: str = ""
    error: Optional[str] = None
    entry_status: Optional[str] = None
    entry: Optional[float] = None
    price_is_live: bool = False
    features: Optional[dict] = None     # setup snapshot, used by the learner


def scan_universe(instruments, frame_loader, min_rr: float = 2.0,
                   direction_override: Optional[str] = None,
                   progress_callback=None, spot_loader=None):
    """Score a list of instruments and return them ranked, best first.

    instruments:    list of (label, ticker, fetch_key) tuples
    frame_loader:   callable(fetch_key) -> (frames_dict, problems_list)
    progress_callback: optional callable(index, total, label)

    Instruments that fail to load are returned with an error rather than being
    silently dropped — a coin missing from the list because its fetch failed
    would be indistinguishable from one that scored badly.
    """
    from scoring import score_setup

    results = []
    total = len(instruments)
    for i, (label, ticker, fetch_key) in enumerate(instruments):
        if progress_callback:
            progress_callback(i, total, label)
        try:
            frames, problems = frame_loader(fetch_key)
            if not frames or all(f is None or f.empty for f in frames.values()):
                results.append(RankedCandidate(
                    ticker=ticker, label=label, direction=None, score=0.0, grade="—",
                    regime="unknown", price=None, stop=None, target=None,
                    reward_risk=None, error="No data returned."))
                continue

            live = None
            if spot_loader is not None:
                try:
                    live = spot_loader(fetch_key)
                except Exception:
                    live = None
            analysis = analyze_candidate(ticker, frames,
                                          direction_override=direction_override,
                                          min_rr=min_rr, data_problems=problems,
                                          live_price=live)
            scored = score_setup(analysis.score_dict())
            results.append(RankedCandidate(
                ticker=ticker, label=label, direction=analysis.direction,
                score=scored.normalized_score, grade=scored.label,
                regime=analysis.regime_1d, price=analysis.current_price,
                stop=analysis.stop, target=analysis.target,
                reward_risk=analysis.reward_risk,
                note=analysis.level_reason,
                entry_status=(analysis.entry_plan.status
                              if analysis.entry_plan else None),
                entry=analysis.entry,
                price_is_live=(analysis.price_source == "live spot")))
        except Exception as e:
            results.append(RankedCandidate(
                ticker=ticker, label=label, direction=None, score=0.0, grade="—",
                regime="unknown", price=None, stop=None, target=None,
                reward_risk=None, error=f"{type(e).__name__}: {e}"))

    if progress_callback:
        progress_callback(total, total, "done")

    # Rank by score, then by reward:risk as a tiebreak. Failures sort last.
    results.sort(key=lambda r: (r.error is None, r.score, r.reward_risk or 0),
                  reverse=True)
    return results


# ---------------------------------------------------------------------
# Entry zones
#
# The strategy is "wait for price to reach a location, then confirm" — not
# "enter at whatever the market is right now". Entering at market takes
# whatever reward:risk happens to exist at this second; entering at a planned
# level takes the reward:risk the setup actually offers.
#
# An entry zone is built from CONFLUENCE: a confirmed swing level that also
# coincides with a Fibonacci retracement and/or an equal-high/low cluster is
# stronger than any one of those alone. Confluence count is reported so a
# one-source zone is never presented as though it were three.
# ---------------------------------------------------------------------

ENTRY_AT_ZONE = "AT_ZONE"
ENTRY_APPROACHING = "APPROACHING"
ENTRY_FAR = "FAR"
ENTRY_MISSED = "MISSED"


@dataclass
class EntryPlan:
    zone_low: float
    zone_high: float
    reference: float               # midpoint — used for R:R maths
    status: str                    # AT_ZONE / APPROACHING / FAR / MISSED
    order_type: str                # "Market" / "Limit" / "Wait" / "None"
    distance_pct: float            # current price to the nearest zone edge
    distance_atr: float
    confluence: int                # how many independent sources agree
    sources: List[str] = field(default_factory=list)
    rationale: str = ""


def recent_extreme(df: pd.DataFrame, direction: str) -> Optional[float]:
    """The extreme of the CURRENT leg, confirmed or not.

    For a Long this is the highest High since the last confirmed swing low —
    the top of the move you would draw a pullback Fibonacci from, and the
    obvious first target for a pullback entry. Fractal confirmation requires
    bars AFTER a swing, so the live leg's high is never confirmed yet; ignoring
    it anchored Fibs on an older, smaller leg and hid the natural target.

    Using it is legitimate for live analysis (the high is already known). It
    must not be used inside the backtester, where it would leak future bars.
    """
    if df is None or df.empty:
        return None
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed]
    if direction == "Long":
        lows = [s for s in swings if s.kind == "low"]
        start = max(lows, key=lambda s: s.index).index if lows else 0
        return float(df["High"].iloc[start:].max())
    highs = [s for s in swings if s.kind == "high"]
    start = max(highs, key=lambda s: s.index).index if highs else 0
    return float(df["Low"].iloc[start:].min())


def _fib_levels_for(df: pd.DataFrame, direction: str) -> Dict[str, float]:
    """Fibonacci levels anchored on the CURRENT leg: last confirmed swing
    point to the live extreme of the move."""
    swings = [s for s in find_swing_points(df, left=2, right=2) if s.confirmed]
    extreme = recent_extreme(df, direction)
    if extreme is None:
        return {}
    try:
        if direction == "Long":
            lows = [s for s in swings if s.kind == "low"]
            if not lows:
                return {}
            anchor = max(lows, key=lambda s: s.index).price
            if extreme <= anchor:
                return {}
            return fib_levels(anchor, extreme, "Long")
        highs = [s for s in swings if s.kind == "high"]
        if not highs:
            return {}
        anchor = max(highs, key=lambda s: s.index).price
        if extreme >= anchor:
            return {}
        return fib_levels(extreme, anchor, "Short")
    except ValueError:
        return {}


def derive_entry_zone(df_4h: pd.DataFrame, current_price: float, direction: str,
                       zone_atr: float = 0.35, approach_atr: float = 3.0,
                       confluence_tolerance: float = 0.01,
                       max_distance_atr: float = 8.0) -> Optional[EntryPlan]:
    """Find the best confluence level to enter FROM, rather than entering at market.

    For a Long this looks BELOW current price (you want to buy a pullback into
    support); for a Short, ABOVE it. Levels already passed through are not
    offered — if price has traded through the zone the setup is MISSED, which
    is information, not a reason to enter late.
    """
    if df_4h is None or df_4h.empty or current_price <= 0 or direction not in ("Long", "Short"):
        return None

    atr = atr_value(df_4h)
    if not atr:
        return None

    swings = [s for s in find_swing_points(df_4h, left=2, right=2) if s.confirmed]
    if not swings:
        return None

    clusters = find_equal_levels(swings, tolerance_pct=0.003)
    fibs = _fib_levels_for(df_4h, direction)

    # Candidate levels on the correct side of price.
    candidates: Dict[float, List[str]] = {}

    def _add(price_level: float, source: str):
        if price_level <= 0:
            return
        on_correct_side = (price_level < current_price if direction == "Long"
                           else price_level > current_price)
        if not on_correct_side:
            return
        for existing in list(candidates):
            if abs(existing - price_level) / existing <= confluence_tolerance:
                candidates[existing].append(source)
                return
        candidates[price_level] = [source]

    wanted_kind = "low" if direction == "Long" else "high"
    for s in swings:
        if s.kind == wanted_kind:
            _add(s.price, "confirmed swing")
    for c in clusters:
        if c["kind"] == wanted_kind and c["touches"] >= 2:
            _add(c["price_avg"], f"equal {wanted_kind}s x{c['touches']}")
    for name, level in fibs.items():
        if name.startswith("retr_"):
            _add(level, f"fib {name.replace('retr_', '')}")

    if not candidates:
        return None

    plans = [_build_plan(level, sorted(set(srcs)), current_price, atr, direction,
                          zone_atr, approach_atr)
             for level, srcs in candidates.items()]
    plans = [p for p in plans if p is not None]
    if not plans:
        return None
    return rank_zones(plans, max_distance_atr=max_distance_atr)[0]


def _build_plan(level: float, sources: List[str], current_price: float, atr: float,
                 direction: str, zone_atr: float, approach_atr: float) -> Optional[EntryPlan]:
    """Turn one candidate level into a fully described EntryPlan."""
    half = atr * zone_atr
    zone_low, zone_high = level - half, level + half
    confluence = len(sources)

    if zone_low <= current_price <= zone_high:
        status, order_type, distance = ENTRY_AT_ZONE, "Market", 0.0
    else:
        nearest_edge = zone_high if current_price > zone_high else zone_low
        distance = abs(current_price - nearest_edge)
        passed_through = (current_price < zone_low if direction == "Long"
                          else current_price > zone_high)
        if passed_through:
            status, order_type = ENTRY_MISSED, "None"
        elif distance <= atr * approach_atr:
            status, order_type = ENTRY_APPROACHING, "Limit"
        else:
            # Still a valid planned entry — price simply hasn't arrived yet.
            # A resting limit order is still the right instrument, it will just
            # take longer to fill (or may never fill, which is also fine).
            status, order_type = ENTRY_FAR, "Limit (distant)"

    distance_pct = distance / current_price * 100 if current_price else 0.0
    distance_atr = distance / atr if atr else 0.0
    src_text = ", ".join(sources)

    if status == ENTRY_AT_ZONE:
        rationale = (f"Price is inside the zone {_fp(zone_low)}–{_fp(zone_high)} "
                      f"({confluence} source(s): {src_text}). Wait for confirmation on "
                      f"the entry timeframe before acting.")
    elif status == ENTRY_APPROACHING:
        rationale = (f"Zone {_fp(zone_low)}–{_fp(zone_high)} is {distance_pct:.2f}% "
                      f"({distance_atr:.1f} ATR) away. Place a limit order at the zone "
                      f"rather than buying at market.")
    elif status == ENTRY_FAR:
        rationale = (f"Zone {_fp(zone_low)}–{_fp(zone_high)} is {distance_pct:.2f}% "
                      f"({distance_atr:.1f} ATR) away. That is the planned entry — "
                      f"a distant limit order, or simply wait. Entering at market "
                      f"instead would give up the better price the setup is based on.")
    else:
        rationale = (f"Price already traded through {_fp(zone_low)}–{_fp(zone_high)}. "
                      f"That location is spent.")

    return EntryPlan(zone_low=zone_low, zone_high=zone_high, reference=level,
                      status=status, order_type=order_type, distance_pct=distance_pct,
                      distance_atr=distance_atr, confluence=confluence,
                      sources=sources, rationale=rationale)


def rank_zones(plans: List[EntryPlan], max_distance_atr: float = 8.0) -> List[EntryPlan]:
    """Rank candidate zones by confluence AND proximity.

    Picking purely by confluence chose distant levels that price may never
    revisit; picking purely by proximity chose weak single-source levels. The
    score trades them off: each extra agreeing source is worth about 3 ATR of
    extra distance. Zones beyond max_distance_atr only rank if nothing nearer
    exists, and MISSED zones always sort last.
    """
    def score(p: EntryPlan) -> float:
        return p.confluence - p.distance_atr / 3.0

    usable = [p for p in plans if p.status != ENTRY_MISSED]
    in_range = [p for p in usable if p.distance_atr <= max_distance_atr]
    pool = in_range or usable
    ranked = sorted(pool, key=score, reverse=True)
    rest = [p for p in plans if p not in ranked]
    return ranked + sorted(rest, key=score, reverse=True)


def derive_entry_zones(df_4h: pd.DataFrame, current_price: float, direction: str,
                        limit: int = 3, **kwargs) -> List[EntryPlan]:
    """Top-ranked candidate zones, so alternatives can be shown alongside the
    primary one rather than hidden."""
    if df_4h is None or df_4h.empty or current_price <= 0 or direction not in ("Long", "Short"):
        return []
    atr = atr_value(df_4h)
    if not atr:
        return []
    swings = [s for s in find_swing_points(df_4h, left=2, right=2) if s.confirmed]
    if not swings:
        return []
    zone_atr = kwargs.get("zone_atr", 0.35)
    approach_atr = kwargs.get("approach_atr", 3.0)
    tol = kwargs.get("confluence_tolerance", 0.01)

    clusters = find_equal_levels(swings, tolerance_pct=0.003)
    fibs = _fib_levels_for(df_4h, direction)
    candidates: Dict[float, List[str]] = {}

    def _add(level, source):
        if level <= 0:
            return
        if not (level < current_price if direction == "Long" else level > current_price):
            return
        for existing in list(candidates):
            if abs(existing - level) / existing <= tol:
                candidates[existing].append(source)
                return
        candidates[level] = [source]

    kind = "low" if direction == "Long" else "high"
    for sp in swings:
        if sp.kind == kind:
            _add(sp.price, "confirmed swing")
    for c in clusters:
        if c["kind"] == kind and c["touches"] >= 2:
            _add(c["price_avg"], f"equal {kind}s x{c['touches']}")
    for name, lv in fibs.items():
        if name.startswith("retr_"):
            _add(lv, f"fib {name.replace('retr_', '')}")

    plans = [_build_plan(lv, sorted(set(srcs)), current_price, atr, direction,
                          zone_atr, approach_atr) for lv, srcs in candidates.items()]
    return rank_zones([p for p in plans if p], kwargs.get("max_distance_atr", 8.0))[:limit]
