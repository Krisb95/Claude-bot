"""
Trades you actually take: log them from a recommendation, complete them when
they close, and let the learner learn from them.

THE LOOP
  1. Take    — "I took this trade" on a scanner setup records it in the journal
               as Open, with the plan (entry, stop, target) AND the setup
               snapshot the learner needs.
  2. Complete — when it closes, enter the exit price (plus the final stop and
               target if you moved them). The result is worked out in R: how
               many times your ORIGINAL risk you made or lost.
  3. Learn   — completed trades feed the same learner used for backtests,
               which looks for kinds of setup that lose.

WHY R, MEASURED AGAINST THE ORIGINAL STOP
  Dollar results depend on position size, which changes trade to trade. R —
  profit divided by the risk you planned when entering — makes every trade
  comparable. The ORIGINAL stop is used even if you later moved it, because
  that is the risk you actually took on when you decided to enter.

WHY "FOLLOWED THE RULES" MATTERS
  A trade entered early, on a hunch, or with a stop moved further away isn't
  a test of the strategy. Only rule-following trades teach it by default;
  mixing in discretionary trades would teach it about your habits, not its
  rules. (That's useful too, and can be switched on separately.)
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import storage

SOURCE_SCANNER = "scanner"

OUTCOME_TARGET = "Hit take profit"
OUTCOME_STOP = "Hit stop loss"
OUTCOME_MANUAL = "Closed manually"
OUTCOMES = [OUTCOME_TARGET, OUTCOME_STOP, OUTCOME_MANUAL]


def realized_r(direction: str, entry: float, initial_stop: float, exit_price: float
               ) -> Optional[float]:
    """Result as a multiple of the original risk. +3.0 = made three times the
    risk; -1.0 = lost exactly the risk; beyond -1 means the exit was worse
    than the stop (slippage, or a stop moved further away)."""
    risk = abs(entry - initial_stop)
    if risk <= 0:
        return None
    move = (exit_price - entry) if direction == "Long" else (entry - exit_price)
    return move / risk


def pl_status_for(r: Optional[float], tolerance: float = 0.05) -> str:
    if r is None:
        return "Closed"
    if r > tolerance:
        return "Win"
    if r < -tolerance:
        return "Loss"
    return "Breakeven"


def default_exit_price(outcome: str, stop: float, target: float,
                       manual: Optional[float] = None) -> Optional[float]:
    if outcome == OUTCOME_TARGET:
        return target
    if outcome == OUTCOME_STOP:
        return stop
    return manual


def take_trade(*, ticker: str, direction: str, planned_entry: float, stop: float,
               target: float, actual_entry: Optional[float] = None,
               quantity: Optional[float] = None, leverage: float = 1.0,
               features: Optional[Dict] = None, followed_rules: bool = True,
               strategy_config: str = "", reason: str = "", score: Optional[float] = None,
               grade: Optional[str] = None, db_path: str = storage.DEFAULT_DB_PATH) -> int:
    """Record a trade you took from a recommendation. Returns the journal id."""
    if direction not in ("Long", "Short"):
        raise ValueError("direction must be Long or Short")
    entry = actual_entry if actual_entry else planned_entry
    wrong_side = stop >= entry if direction == "Long" else stop <= entry
    if wrong_side:
        raise ValueError("The stop must be below entry for a long and above it for a short.")
    risk_per_unit = abs(entry - stop)
    qty = quantity or 0.0
    return storage.add_journal_entry({
        "status": "Open", "asset": ticker, "direction": direction, "leverage": leverage,
        "entry": entry, "planned_entry": planned_entry, "sl": stop, "initial_stop": stop,
        "tp": target, "quantity": qty,
        "size_notional_usd": entry * qty,
        "potential_loss_at_sl": -risk_per_unit * qty,
        "potential_profit_at_tp": abs(target - entry) * qty,
        "entry_reason": reason,
        "score_breakdown": (f"{score:.0f}/10 {grade}" if score is not None else ""),
        "features": json.dumps(features) if features else None,
        "followed_rules": 1 if followed_rules else 0,
        "strategy_config": strategy_config, "source": SOURCE_SCANNER,
        "pl_status": "Open",
    }, db_path=db_path)


def complete_trade(entry_id: int, *, exit_price: float, actual_entry: Optional[float] = None,
                   final_stop: Optional[float] = None, final_target: Optional[float] = None,
                   fees: float = 0.0, exit_reason: str = "",
                   db_path: str = storage.DEFAULT_DB_PATH) -> Dict:
    """Close an open journal trade and work out its result."""
    df = storage.get_journal_df(db_path)
    rows = df[df["id"] == entry_id]
    if rows.empty:
        raise KeyError(f"No journal entry {entry_id}")
    row = rows.iloc[0]
    direction = row["direction"]
    entry = float(actual_entry) if actual_entry else float(row["entry"])
    initial_stop = row.get("initial_stop")
    if initial_stop is None or pd.isna(initial_stop):
        initial_stop = row["sl"]
    initial_stop = float(initial_stop)

    r = realized_r(direction, entry, initial_stop, float(exit_price))
    qty = row.get("quantity")
    qty = 0.0 if qty is None or pd.isna(qty) else float(qty)
    move = (exit_price - entry) if direction == "Long" else (entry - exit_price)
    pnl = move * qty - (fees or 0.0) if qty else None

    updates = {
        "status": "Closed", "exit_price": float(exit_price), "entry": entry,
        "realized_r": r, "pl_status": pl_status_for(r),
        "exit_reason": exit_reason, "closed_utc": datetime.now(timezone.utc).isoformat(),
    }
    if pnl is not None:
        updates["realized_pnl"] = pnl
    if final_stop is not None and final_stop != float(row["sl"]):
        updates["updated_sl"] = float(final_stop)
    if final_target is not None and final_target != float(row["tp"]):
        updates["updated_tp"] = float(final_target)
    storage.update_journal_entry(entry_id, db_path=db_path, **updates)
    return updates


@dataclass
class JournalTrade:
    """A completed real trade, in the shape the learner expects."""
    signal_time: pd.Timestamp
    features: Optional[Dict]
    r_result: Optional[float]
    status: str
    ticker: str
    direction: str


def learning_trades(db_path: str = storage.DEFAULT_DB_PATH, config: Optional[str] = None,
                    followed_only: bool = True) -> List[JournalTrade]:
    """Completed scanner trades for the learner.

    config:        only trades taken under these strategy settings, since a
                   lesson about one stop/target setup says nothing about another
    followed_only: only trades where you followed the rules
    """
    df = storage.get_journal_df(db_path)
    out: List[JournalTrade] = []
    if df.empty or "realized_r" not in df.columns:
        return out
    for _, row in df.iterrows():
        if row.get("status") != "Closed" or row.get("source") != SOURCE_SCANNER:
            continue
        r = row.get("realized_r")
        if r is None or pd.isna(r):
            continue
        if followed_only and not row.get("followed_rules"):
            continue
        if config is not None and row.get("strategy_config") != config:
            continue
        feats = None
        raw = row.get("features")
        if isinstance(raw, str) and raw:
            try:
                feats = json.loads(raw)
            except ValueError:
                feats = None
        out.append(JournalTrade(
            signal_time=pd.Timestamp(row["timestamp_utc"]), features=feats,
            r_result=float(r), status="WIN" if float(r) > 0 else "LOSS",
            ticker=row["asset"], direction=row["direction"]))
    return out


def open_scanner_trades(db_path: str = storage.DEFAULT_DB_PATH) -> pd.DataFrame:
    df = storage.get_journal_df(db_path)
    if df.empty or "source" not in df.columns:
        return df.iloc[0:0]
    return df[(df["status"] == "Open") & (df["source"] == SOURCE_SCANNER)]
