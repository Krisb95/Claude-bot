"""
Forward tracking: record the scanner's live signals, then resolve each one
against the prices that actually followed.

This is the most honest test available. The plan (entry, stop, target) is
frozen at the moment the signal is generated, before anyone knows the result.
Nothing can be tuned to fit the outcome afterwards.

It works even though the app sleeps: outcomes are resolved by fetching the
candles printed SINCE each signal was created, whenever the app is next
opened. A trade that filled and hit its target at 3am is still found.

Resolution uses the same limit-order simulator as the backtest, with the same
deliberately pessimistic handling of ambiguous candles, so tracked results and
backtest results are directly comparable.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
import json
import pandas as pd

import storage
from trade_sim import simulate_limit_trade, PENDING, FILLED, WIN, LOSS, EXPIRED

DEFAULT_EXPIRY_HOURS = 72
TRACKED_GRADES = ("A+", "B")
UNRESOLVED = (PENDING, FILLED)


def record_from_ranked(ranked, tags: Optional[Dict[str, List[str]]] = None,
                        expiry_hours: float = DEFAULT_EXPIRY_HOURS,
                        grades=TRACKED_GRADES, db_path: str = storage.DEFAULT_DB_PATH,
                        min_rr: float = 0.0) -> int:
    """grades=None records every setup with a complete plan, whatever its grade.

    Recording the weaker ones too is what lets you compare grades later — and
    it means a setup you looked at last week is still there to re-check, rather
    than only the ones that happened to be A+ at the time."""
    """Store every qualifying setup from a universe scan. Returns count added.

    min_rr: setups below this reward:risk are not recorded — tracking only
    what you would actually trade keeps the track record meaningful."""
    added = 0
    for r in ranked:
        if r.error or (grades is not None and r.grade not in grades):
            continue
        if r.reward_risk is None or r.reward_risk < min_rr - 1e-9:
            continue
        if None in (r.entry, r.stop, r.target, r.direction):
            continue
        new_id = storage.record_signal({
            "ticker": r.ticker, "label": r.label, "direction": r.direction,
            "score": r.score, "grade": r.grade, "entry": r.entry, "stop": r.stop,
            "target": r.target, "planned_rr": r.reward_risk,
            "expiry_hours": expiry_hours, "source": "universe",
            "features": json.dumps(r.features) if getattr(r, "features", None) else None,
        }, db_path=db_path)
        if new_id is not None:
            added += 1
    return added


def _bars_since(bars: pd.DataFrame, created_utc: str) -> pd.DataFrame:
    """Only candles that OPENED after the signal was created. A candle already
    in progress at signal time is excluded — its earlier part happened before
    the order existed and could not have filled it."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    created = pd.Timestamp(created_utc)
    if created.tzinfo is None:
        created = created.tz_localize("UTC")
    return bars[bars.index > created]


def resolve_signal(row: Dict, bars: pd.DataFrame, bar_hours: float = 1.0):
    """Resolve one signal. Returns the updates to apply, or None if unchanged."""
    after = _bars_since(bars, row["created_utc"])
    expiry_bars = max(1, int(round(float(row.get("expiry_hours") or DEFAULT_EXPIRY_HOURS)
                                   / bar_hours)))
    sim = simulate_limit_trade(after, row["direction"], float(row["entry"]),
                                float(row["stop"]), float(row["target"]),
                                expiry_bars=expiry_bars)
    updates = {"status": sim.status,
               "last_checked_utc": datetime.now(timezone.utc).isoformat()}
    if sim.fill_time is not None:
        updates["fill_utc"] = sim.fill_time.isoformat()
    if sim.exit_time is not None:
        updates["exit_utc"] = sim.exit_time.isoformat()
    if sim.r_result is not None:
        updates["r_result"] = sim.r_result
    return updates


def update_all(bar_loader: Callable[[str], Optional[pd.DataFrame]],
                bar_hours: float = 1.0, db_path: str = storage.DEFAULT_DB_PATH,
                progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, int]:
    """Resolve every open signal. bar_loader(ticker) returns recent OHLC bars.

    Returns counts of what changed, so the UI can report it."""
    df = storage.get_signals_df(db_path)
    open_rows = df[df["status"].isin(UNRESOLVED)] if not df.empty else df
    counts = {"checked": 0, "resolved": 0, "failed": 0}
    total = len(open_rows)
    for i, (_, row) in enumerate(open_rows.iterrows()):
        if progress:
            progress(i, total, row["ticker"])
        counts["checked"] += 1
        try:
            bars = bar_loader(row["ticker"])
            if bars is None or bars.empty:
                counts["failed"] += 1
                continue
            updates = resolve_signal(row.to_dict(), bars, bar_hours=bar_hours)
            if updates["status"] in (WIN, LOSS, EXPIRED):
                counts["resolved"] += 1
            storage.update_signal(int(row["id"]), db_path=db_path, **updates)
        except Exception:
            counts["failed"] += 1
    if progress:
        progress(total, total, "done")
    return counts


def tracked_stats(db_path: str = storage.DEFAULT_DB_PATH):
    """Results by grade, in the same shape the backtest reports."""
    from strategy_backtest import BacktestTrade, stats_by_grade
    df = storage.get_signals_df(db_path)
    trades = []
    for _, r in df.iterrows():
        trades.append(BacktestTrade(
            ticker=r["ticker"], signal_time=pd.Timestamp(r["created_utc"]),
            direction=r["direction"], score=float(r["score"] or 0), grade=r["grade"],
            entry=float(r["entry"]), stop=float(r["stop"]), target=float(r["target"]),
            planned_rr=float(r["planned_rr"] or 0), status=r["status"],
            r_result=None if pd.isna(r["r_result"]) else float(r["r_result"])))
    return stats_by_grade(trades)



@dataclass
class TrackedTrade:
    """A resolved tracked signal, in the shape the learner expects."""
    signal_time: pd.Timestamp
    features: Optional[dict]
    r_result: Optional[float]
    status: str
    ticker: str = ""
    direction: str = ""


def tracked_trades(db_path: str = storage.DEFAULT_DB_PATH) -> List[TrackedTrade]:
    """The app's own live trades — plan frozen at signal time, outcome resolved
    later — with the setup snapshot each was taken with."""
    df = storage.get_signals_df(db_path)
    out = []
    for _, r in df.iterrows():
        raw = r.get("features")
        feats = None
        if isinstance(raw, str) and raw:
            try:
                feats = json.loads(raw)
            except ValueError:
                feats = None
        out.append(TrackedTrade(
            signal_time=pd.Timestamp(r["created_utc"]), features=feats,
            r_result=None if pd.isna(r["r_result"]) else float(r["r_result"]),
            status=r["status"], ticker=r["ticker"], direction=r["direction"]))
    return out


def record_one(plan_like, source: str = "manual", expiry_hours: float = DEFAULT_EXPIRY_HOURS,
               db_path: str = storage.DEFAULT_DB_PATH) -> Optional[int]:
    """Save a single setup from a one-off check, so it can be re-checked later."""
    if None in (getattr(plan_like, "entry", None), getattr(plan_like, "stop", None),
                getattr(plan_like, "target", None), getattr(plan_like, "direction", None)):
        return None
    return storage.record_signal({
        "ticker": plan_like.ticker, "label": getattr(plan_like, "label", plan_like.ticker),
        "direction": plan_like.direction, "score": getattr(plan_like, "score", None),
        "grade": getattr(plan_like, "grade", None), "entry": plan_like.entry,
        "stop": plan_like.stop, "target": plan_like.target,
        "planned_rr": getattr(plan_like, "reward_risk", None),
        "expiry_hours": expiry_hours, "source": source,
        "features": json.dumps(getattr(plan_like, "features", None) or {}) or None,
    }, db_path=db_path)
