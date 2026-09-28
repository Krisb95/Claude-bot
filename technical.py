"""
Technical structure detection from actual OHLC data.

HONEST SCOPE NOTE: this is a simplified, rule-based fractal/pivot detector —
real, testable, and NOT hardcoded — but it is not equivalent to discretionary
chart reading. It has no volume-profile/order-flow liquidity data, so
"liquidity pools" here means equal-highs/equal-lows clustering only, and
sweep/reclaim confirmation is a close-based rule, not a full market-structure
model. Treat this as a first-pass structural overlay, not ground truth.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
import pandas as pd


@dataclass
class SwingPoint:
    index: int          # positional index into the source DataFrame
    price: float
    kind: Literal["high", "low"]
    confirmed: bool      # False if it's within `right` bars of the end (not yet confirmed)


def find_swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[SwingPoint]:
    """Fractal swing detection: a bar is a swing high if its High is the
    max within [i-left, i+right], and a swing low if its Low is the min
    within that window. Requires `right` bars after a point to confirm it —
    swings near the very end of the series are marked confirmed=False since
    a future bar could still invalidate them (this avoids look-ahead bias
    when this function is reused inside the backtester).
    """
    if len(df) < left + right + 1:
        return []

    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    points: List[SwingPoint] = []

    for i in range(left, n - right if n - right > left else n):
        window_high = highs[max(0, i - left): i + right + 1]
        window_low = lows[max(0, i - left): i + right + 1]
        confirmed = (i + right) < n

        if highs[i] == window_high.max() and (window_high == highs[i]).sum() == 1:
            points.append(SwingPoint(index=i, price=float(highs[i]), kind="high", confirmed=confirmed))
        if lows[i] == window_low.min() and (window_low == lows[i]).sum() == 1:
            points.append(SwingPoint(index=i, price=float(lows[i]), kind="low", confirmed=confirmed))

    points.sort(key=lambda p: p.index)
    return points


@dataclass
class LabeledSwing:
    index: int
    price: float
    kind: Literal["high", "low"]
    label: str  # "HH", "HL", "LH", "LL", or "" for the first of its kind


def label_structure(points: List[SwingPoint]) -> List[LabeledSwing]:
    """Label each swing relative to the previous swing of the SAME kind
    (HH/LH for highs, HL/LL for lows)."""
    labeled: List[LabeledSwing] = []
    last_high: Optional[float] = None
    last_low: Optional[float] = None

    for p in points:
        if p.kind == "high":
            if last_high is None:
                label = ""
            else:
                label = "HH" if p.price > last_high else "LH"
            last_high = p.price
        else:
            if last_low is None:
                label = ""
            else:
                label = "HL" if p.price > last_low else "LL"
            last_low = p.price
        labeled.append(LabeledSwing(index=p.index, price=p.price, kind=p.kind, label=label))

    return labeled


def find_equal_levels(points: List[SwingPoint], tolerance_pct: float = 0.001) -> List[dict]:
    """Cluster swing points of the same kind that sit within tolerance_pct
    of each other — a simple, real proxy for 'equal highs/equal lows'
    liquidity pools (not a substitute for actual order-book/volume data)."""
    clusters = []
    for kind in ("high", "low"):
        kind_points = sorted([p for p in points if p.kind == kind], key=lambda p: p.price)
        used = [False] * len(kind_points)
        for i, p in enumerate(kind_points):
            if used[i]:
                continue
            group = [p]
            used[i] = True
            for j in range(i + 1, len(kind_points)):
                if used[j]:
                    continue
                if p.price == 0:
                    is_close = kind_points[j].price == 0
                else:
                    is_close = abs(kind_points[j].price - p.price) / p.price <= tolerance_pct
                if is_close:
                    group.append(kind_points[j])
                    used[j] = True
            if len(group) >= 2:
                clusters.append({
                    "kind": kind,
                    "price_avg": sum(g.price for g in group) / len(group),
                    "touches": len(group),
                    "indices": [g.index for g in group],
                })
    return clusters


FIB_RETRACEMENT_RATIOS = [0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION_RATIOS = [1.272, 1.618]


def fib_levels(swing_low: float, swing_high: float, direction: str) -> dict:
    """Fibonacci retracement/extension levels computed from two REAL swing
    anchors — never invented. `direction` = "Long" (retracement measured
    down from the high) or "Short" (retracement measured up from the low)."""
    direction = direction.capitalize()
    span = swing_high - swing_low
    if span <= 0:
        raise ValueError("swing_high must be greater than swing_low")

    levels = {}
    if direction == "Long":
        for r in FIB_RETRACEMENT_RATIOS:
            levels[f"retr_{r}"] = swing_high - span * r
        for r in FIB_EXTENSION_RATIOS:
            levels[f"ext_{r}"] = swing_high + span * (r - 1)
    elif direction == "Short":
        for r in FIB_RETRACEMENT_RATIOS:
            levels[f"retr_{r}"] = swing_low + span * r
        for r in FIB_EXTENSION_RATIOS:
            levels[f"ext_{r}"] = swing_low - span * (r - 1)
    else:
        raise ValueError("direction must be 'Long' or 'Short'")

    return levels
