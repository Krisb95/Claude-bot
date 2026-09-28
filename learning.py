"""
Learning from past trades: which KINDS of setup lose, so they can be skipped.

HOW IT LEARNS
  1. Every trade carries a snapshot of its setup at signal time (trend
     strength, retrace depth, stop width, 1H candle strength, volatility,
     session, direction).
  2. Trades are split by time: the earlier 70% to learn from, the later 30%
     held back, unseen.
  3. On the earlier trades it looks for a condition whose trades lost clearly
     more than the rest — e.g. "retrace deeper than 2.4 ATR".
  4. Each candidate lesson is then checked on the held-back later trades. It
     is only adopted if those trades ALSO lost, by a margin chance rarely
     produces.

WHY THE HOLD-OUT MATTERS
  Search enough conditions and some will look meaningful purely by luck —
  in random data too. A lesson that only works on the trades it was found in
  is memorised noise, and filtering on it would make results worse, not
  better. The held-back check is what separates a real lesson from a
  coincidence. When nothing survives it, the right answer is to change
  nothing — and the model says so.

WHAT IT CANNOT DO
  It only learns to SKIP losing kinds of setup; it doesn't invent new rules.
  It needs plenty of trades — lessons from a few dozen are not trustworthy.
  Markets change, so lessons should be re-learned as new trades arrive, and
  a lesson that held last quarter can fade.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

NUMERIC_FEATURES = {
    "trend_move_pct": "4H trend move (%)",
    "retrace_depth_atr": "Retrace depth to entry (x 5m ATR)",
    "risk_pct": "Stop distance (% of entry)",
    "h1_body_ratio": "1H confirmation candle strength (body ÷ range)",
    "volatility_pct": "Market volatility (5m ATR, % of price)",
}
CATEGORICAL_FEATURES = {"session": "Trading session", "direction": "Direction"}

MIN_TRADES = 80          # below this, nothing is learned
TRAIN_FRACTION = 0.7
MIN_BUCKET_TRAIN = 20
MIN_BUCKET_TEST = 12
TRAIN_MARGIN_R = 0.2     # bucket must trail the rest by this much in training
Z_CRITICAL = 2.0         # held-back losses must be this many standard errors below zero
MAX_RULES = 2


@dataclass
class Rule:
    feature: str
    kind: str                         # "low" | "high" | "category"
    threshold: Optional[float] = None
    category: Optional[str] = None
    description: str = ""
    train_n: int = 0
    train_avg: Optional[float] = None
    train_rest_avg: Optional[float] = None
    test_n: int = 0
    test_avg: Optional[float] = None
    test_rest_avg: Optional[float] = None
    test_z: Optional[float] = None
    accepted: bool = False
    verdict: str = ""

    def matches(self, features: Optional[Dict]) -> bool:
        """True if a setup falls in the group this rule says to skip."""
        if not features or self.feature not in features:
            return False                       # can't judge: don't skip
        v = features[self.feature]
        if self.kind == "category":
            return v == self.category
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        return v < self.threshold if self.kind == "low" else v >= self.threshold


@dataclass
class LearnedModel:
    rules: List[Rule] = field(default_factory=list)          # adopted lessons
    candidates: List[Rule] = field(default_factory=list)     # everything tested
    n_trades: int = 0
    n_train: int = 0
    n_test: int = 0
    test_avg_before: Optional[float] = None
    test_avg_after: Optional[float] = None
    test_kept: int = 0
    summary: str = ""
    source: str = ""

    def check(self, features: Optional[Dict]) -> Tuple[bool, List[str]]:
        """(passes, reasons). A setup fails if it matches any adopted lesson."""
        hits = [r.description for r in self.rules if r.matches(features)]
        return (len(hits) == 0), hits

    def to_dict(self) -> Dict:
        return {"rules": [asdict(r) for r in self.rules],
                "candidates": [asdict(r) for r in self.candidates],
                **{k: getattr(self, k) for k in ("n_trades", "n_train", "n_test",
                                                  "test_avg_before", "test_avg_after",
                                                  "test_kept", "summary", "source")}}

    @classmethod
    def from_dict(cls, d: Dict) -> "LearnedModel":
        m = cls(rules=[Rule(**r) for r in d.get("rules", [])],
                candidates=[Rule(**r) for r in d.get("candidates", [])])
        for k in ("n_trades", "n_train", "n_test", "test_avg_before", "test_avg_after",
                  "test_kept", "summary", "source"):
            setattr(m, k, d.get(k))
        return m


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _z(xs) -> Optional[float]:
    """How many standard errors the mean sits from zero."""
    if len(xs) < 2:
        return None
    sd = float(np.std(xs, ddof=1))
    if sd == 0:
        return None
    return float(np.mean(xs)) / (sd / np.sqrt(len(xs)))


def _describe(feature: str, kind: str, threshold=None, category=None) -> str:
    if kind == "category":
        label = CATEGORICAL_FEATURES.get(feature, feature)
        return f"{label} is {category}"
    label = NUMERIC_FEATURES.get(feature, feature)
    word = "below" if kind == "low" else "at or above"
    return f"{label} {word} {threshold:.3g}"


def learn(trades: List, source: str = "", min_trades: int = MIN_TRADES) -> LearnedModel:
    """Find conditions that separate losing setups, validated on unseen trades.

    trades: objects with .status ("WIN"/"LOSS"), .signal_time, .features, and
    .r_result. Only resolved trades that carry features are used; for strategies
    with re-entry legs, pass initial entries only so each trade stands alone.
    """
    rows = []
    for t in trades:
        if getattr(t, "status", None) not in ("WIN", "LOSS"):
            continue
        feats = getattr(t, "features", None)
        r = getattr(t, "r_result", None)
        if not feats or r is None:
            continue
        rows.append((pd.Timestamp(t.signal_time), feats, float(r)))

    model = LearnedModel(n_trades=len(rows), source=source)
    if len(rows) < min_trades:
        model.summary = (f"Not enough trades to learn from yet — {len(rows)} usable, about "
                         f"{min_trades} needed. With fewer, any 'lesson' would mostly be luck.")
        return model

    rows.sort(key=lambda x: x[0])
    cut = int(len(rows) * TRAIN_FRACTION)
    train, test = rows[:cut], rows[cut:]
    model.n_train, model.n_test = len(train), len(test)

    def split(data, rule):
        hit = [r for _, f, r in data if rule.matches(f)]
        rest = [r for _, f, r in data if not rule.matches(f)]
        return hit, rest

    # --- candidate lessons, from the TRAINING trades only ----------------
    candidates: List[Rule] = []
    for feat in NUMERIC_FEATURES:
        vals = [float(f[feat]) for _, f, _ in train if feat in f]
        if len(vals) < 3 * MIN_BUCKET_TRAIN:
            continue
        lo, hi = np.percentile(vals, [33.3, 66.7])
        candidates.append(Rule(feat, "low", threshold=float(lo),
                               description=_describe(feat, "low", lo)))
        candidates.append(Rule(feat, "high", threshold=float(hi),
                               description=_describe(feat, "high", hi)))
    for feat in CATEGORICAL_FEATURES:
        cats = sorted({f[feat] for _, f, _ in train if feat in f})
        if len(cats) < 2:
            continue
        for c in cats:
            candidates.append(Rule(feat, "category", category=c,
                                   description=_describe(feat, "category", category=c)))

    # --- screen on training data, then validate on held-back data --------
    for rule in candidates:
        hit, rest = split(train, rule)
        rule.train_n, rule.train_avg, rule.train_rest_avg = len(hit), _mean(hit), _mean(rest)
        if (len(hit) < MIN_BUCKET_TRAIN or rule.train_avg is None or rule.train_rest_avg is None
                or rule.train_avg >= 0
                or rule.train_avg > rule.train_rest_avg - TRAIN_MARGIN_R):
            rule.verdict = "No clear losing pattern in the learning trades."
            continue
        thit, trest = split(test, rule)
        rule.test_n, rule.test_avg, rule.test_rest_avg = len(thit), _mean(thit), _mean(trest)
        rule.test_z = _z(thit)
        if len(thit) < MIN_BUCKET_TEST:
            rule.verdict = "Looked like a loser, but too few held-back trades to confirm."
        elif (rule.test_avg is not None and rule.test_avg < 0
              and rule.test_rest_avg is not None and rule.test_avg < rule.test_rest_avg
              and rule.test_z is not None and rule.test_z <= -Z_CRITICAL):
            rule.accepted = True
            rule.verdict = "Confirmed on unseen trades — adopted."
        else:
            rule.verdict = ("Looked like a loser in the learning trades but did NOT hold up on "
                            "unseen trades — probably coincidence, so not adopted.")

    adopted = sorted([r for r in candidates if r.accepted], key=lambda r: r.test_z)[:MAX_RULES]
    for r in candidates:
        if r.accepted and r not in adopted:
            r.accepted = False
            r.verdict = "Confirmed, but a stronger lesson already covers it (limit of 2)."
    model.rules = adopted
    model.candidates = candidates

    before = [r for _, _, r in test]
    after = [r for _, f, r in test if not any(rule.matches(f) for rule in adopted)]
    model.test_avg_before, model.test_avg_after = _mean(before), _mean(after)
    model.test_kept = len(after)

    if adopted:
        model.summary = (f"Learned {len(adopted)} lesson(s) that held up on {len(test)} unseen "
                         f"trades. On those trades, skipping these setups moved the average "
                         f"from {model.test_avg_before:+.2f}R to {model.test_avg_after:+.2f}R "
                         f"per trade.")
    else:
        model.summary = (f"No lesson survived testing on {len(test)} unseen trades, so nothing "
                         f"is changed. Patterns that appeared in the learning trades didn't "
                         f"repeat — acting on them would have been acting on luck.")
    return model
