"""
SQLite persistence for the trade journal and open positions.

WHY THIS EXISTS: previously the journal and positions lived in Streamlit
session state, which is wiped whenever the app restarts, sleeps, or the
browser tab is closed. That is unacceptable for a trade record.

IMPORTANT CAVEAT ABOUT FREE HOSTING: on Hugging Face Spaces (free tier) the
container filesystem is EPHEMERAL. This database survives page refreshes,
reruns and app sleeps, but it is wiped when the Space is rebuilt (i.e. when
you push new code or restart the container from settings). Treat the CSV
export as your real backup — export after any session that matters. For
durable storage you would need HF persistent storage (paid) or an external
database. This module does not pretend otherwise.
"""

import sqlite3
import os
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import io
import pandas as pd

DEFAULT_DB_PATH = os.environ.get("TRADING_DB_PATH", "trading_data.db")

JOURNAL_COLUMNS = [
    "id", "status", "asset", "direction", "leverage", "entry", "size_notional_usd",
    "margin", "current_pl", "tp", "sl", "updated_tp", "updated_sl",
    "potential_profit_at_tp", "potential_loss_at_sl", "realized_pnl",
    "pl_status", "entry_reason", "score_breakdown", "exit_reason",
    "timestamp_utc", "management_notes",
    # Added so trades you actually take can teach the learner:
    "planned_entry", "initial_stop", "exit_price", "realized_r", "quantity",
    "features", "followed_rules", "strategy_config", "source", "closed_utc",
]


def _ensure_columns(conn, table: str, columns: Dict[str, str]) -> None:
    """Add any missing columns to an existing table (a lightweight migration)."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


@contextmanager
def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create tables if they don't exist. Safe to call on every run."""
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT DEFAULT 'Open',
                asset TEXT,
                direction TEXT,
                leverage REAL DEFAULT 1.0,
                entry REAL DEFAULT 0.0,
                size_notional_usd REAL DEFAULT 0.0,
                margin REAL DEFAULT 0.0,
                current_pl REAL DEFAULT 0.0,
                tp REAL DEFAULT 0.0,
                sl REAL DEFAULT 0.0,
                updated_tp REAL,
                updated_sl REAL,
                potential_profit_at_tp REAL DEFAULT 0.0,
                potential_loss_at_sl REAL DEFAULT 0.0,
                realized_pnl REAL,
                pl_status TEXT DEFAULT 'Open',
                entry_reason TEXT DEFAULT '',
                score_breakdown TEXT DEFAULT '',
                exit_reason TEXT DEFAULT '',
                timestamp_utc TEXT,
                management_notes TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT,
                asset_class TEXT,
                direction TEXT,
                entry REAL,
                stop REAL,
                quantity REAL,
                contract_multiplier REAL DEFAULT 1.0,
                opened_utc TEXT,
                target REAL,
                leverage REAL DEFAULT 1.0,
                notes TEXT DEFAULT ''
            )
        """)
        # Columns added after the first release. Existing databases are
        # migrated in place rather than dropped, so a schema change never
        # silently destroys someone's open positions.
        _ensure_columns(conn, "positions", {
            "target": "REAL",
            "leverage": "REAL DEFAULT 1.0",
            "fee_rate": "REAL DEFAULT 0.0",
            "slippage_pct": "REAL DEFAULT 0.0",
            "entry_reason": "TEXT DEFAULT ''",
            "notes": "TEXT DEFAULT ''",
        })
        _ensure_columns(conn, "journal", {
            "planned_entry": "REAL",
            "initial_stop": "REAL",
            "exit_price": "REAL",
            "realized_r": "REAL",
            "quantity": "REAL",
            "features": "TEXT",
            "followed_rules": "INTEGER",
            "strategy_config": "TEXT",
            "source": "TEXT",
            "closed_utc": "TEXT",
        })


# ---------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------

def add_journal_entry(data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert a journal row. Returns its database id."""
    payload = dict(data)
    payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    payload.setdefault("status", "Open")
    payload.setdefault("pl_status", "Open")

    fields = [k for k in payload if k in JOURNAL_COLUMNS and k != "id"]
    placeholders = ", ".join("?" for _ in fields)
    sql = f"INSERT INTO journal ({', '.join(fields)}) VALUES ({placeholders})"
    with _connect(db_path) as conn:
        cur = conn.execute(sql, [payload[f] for f in fields])
        return cur.lastrowid


def update_journal_entry(entry_id: int, db_path: str = DEFAULT_DB_PATH, **changes) -> None:
    """Update named columns on one journal row."""
    valid = {k: v for k, v in changes.items() if k in JOURNAL_COLUMNS and k != "id"}
    if not valid:
        raise ValueError(f"No valid journal columns in {list(changes)}")
    assignments = ", ".join(f"{k} = ?" for k in valid)
    with _connect(db_path) as conn:
        cur = conn.execute(f"UPDATE journal SET {assignments} WHERE id = ?",
                            list(valid.values()) + [entry_id])
        if cur.rowcount == 0:
            raise KeyError(f"No journal entry with id {entry_id}")


def close_journal_entry(entry_id: int, realized_pnl: float, exit_reason: str,
                         db_path: str = DEFAULT_DB_PATH) -> None:
    pl_status = "Win" if realized_pnl > 0 else ("Loss" if realized_pnl < 0 else "Breakeven")
    update_journal_entry(entry_id, db_path=db_path, status="Closed",
                          realized_pnl=realized_pnl, exit_reason=exit_reason,
                          pl_status=pl_status)


def delete_journal_entry(entry_id: int, db_path: str = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM journal WHERE id = ?", (entry_id,))


def get_journal_df(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM journal ORDER BY id").fetchall()
    if not rows:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    return pd.DataFrame([dict(r) for r in rows])


def journal_to_csv_bytes(db_path: str = DEFAULT_DB_PATH) -> bytes:
    buf = io.StringIO()
    get_journal_df(db_path).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------

POSITION_EDITABLE = {"stop", "target", "quantity", "leverage", "fee_rate",
                      "slippage_pct", "entry", "notes", "entry_reason"}


def add_position(asset: str, asset_class: str, direction: str, entry: float,
                  stop: float, quantity: float, contract_multiplier: float = 1.0,
                  target: Optional[float] = None, leverage: float = 1.0,
                  fee_rate: float = 0.0, slippage_pct: float = 0.0,
                  entry_reason: str = "", db_path: str = DEFAULT_DB_PATH) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO positions
               (asset, asset_class, direction, entry, stop, quantity,
                contract_multiplier, opened_utc, target, leverage, fee_rate,
                slippage_pct, entry_reason, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')""",
            (asset, asset_class, direction, entry, stop, quantity,
             contract_multiplier, datetime.now(timezone.utc).isoformat(),
             target, leverage, fee_rate, slippage_pct, entry_reason),
        )
        return cur.lastrowid


def update_position(position_id: int, db_path: str = DEFAULT_DB_PATH, **changes) -> None:
    """Edit an open position (stop, target, size, costs, notes)."""
    valid = {k: v for k, v in changes.items() if k in POSITION_EDITABLE}
    if not valid:
        raise ValueError(f"No editable position fields in {list(changes)}")
    assignments = ", ".join(f"{k} = ?" for k in valid)
    with _connect(db_path) as conn:
        cur = conn.execute(f"UPDATE positions SET {assignments} WHERE id = ?",
                            list(valid.values()) + [position_id])
        if cur.rowcount == 0:
            raise KeyError(f"No position with id {position_id}")


def update_position_stop(position_id: int, new_stop: float,
                          db_path: str = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        cur = conn.execute("UPDATE positions SET stop = ? WHERE id = ?",
                            (new_stop, position_id))
        if cur.rowcount == 0:
            raise KeyError(f"No position with id {position_id}")


def delete_position(position_id: int, db_path: str = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM positions WHERE id = ?", (position_id,))


def get_positions(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def clear_all(db_path: str = DEFAULT_DB_PATH) -> None:
    """Wipe both tables. Used by the explicit reset control in the UI."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM journal")
        conn.execute("DELETE FROM positions")


# ---------------------------------------------------------------------
# Forward-tracked signals
#
# Every B-or-better setup the scanner produces is recorded here with the
# exact plan (entry, stop, target) at the moment it was generated. Later, the
# outcome is resolved against real prices that came afterwards. This builds a
# genuine out-of-sample track record — the plan is fixed before the result is
# known, so there is no way to fit the rules to the outcome.
# ---------------------------------------------------------------------

SIGNAL_COLUMNS = [
    "id", "created_utc", "ticker", "label", "direction", "score", "grade",
    "entry", "stop", "target", "planned_rr", "expiry_hours", "status",
    "fill_utc", "exit_utc", "r_result", "last_checked_utc", "source", "features",
]


def _ensure_signals_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_utc TEXT NOT NULL,
            ticker TEXT NOT NULL,
            label TEXT,
            direction TEXT NOT NULL,
            score REAL,
            grade TEXT,
            entry REAL NOT NULL,
            stop REAL NOT NULL,
            target REAL NOT NULL,
            planned_rr REAL,
            expiry_hours REAL DEFAULT 72,
            status TEXT DEFAULT 'PENDING',
            fill_utc TEXT,
            exit_utc TEXT,
            r_result REAL,
            last_checked_utc TEXT,
            source TEXT DEFAULT 'scan',
            features TEXT
        )
    """)
    # Databases created before setup snapshots existed gain the column.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(signals)")}
    if "features" not in cols:
        conn.execute("ALTER TABLE signals ADD COLUMN features TEXT")


def record_signal(data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> Optional[int]:
    """Record a signal unless an unresolved one already exists for the same
    ticker and direction — rescanning every hour must not log the same setup
    over and over, which would multiply its weight in the statistics."""
    with _connect(db_path) as conn:
        _ensure_signals_table(conn)
        dup = conn.execute(
            "SELECT id FROM signals WHERE ticker = ? AND direction = ? "
            "AND status IN ('PENDING', 'FILLED')",
            (data["ticker"], data["direction"])).fetchone()
        if dup:
            return None
        payload = dict(data)
        payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        payload.setdefault("status", "PENDING")
        fields = [k for k in payload if k in SIGNAL_COLUMNS and k != "id"]
        cur = conn.execute(
            f"INSERT INTO signals ({', '.join(fields)}) "
            f"VALUES ({', '.join('?' for _ in fields)})",
            [payload[f] for f in fields])
        return cur.lastrowid


def update_signal(signal_id: int, db_path: str = DEFAULT_DB_PATH, **changes) -> None:
    valid = {k: v for k, v in changes.items() if k in SIGNAL_COLUMNS and k != "id"}
    if not valid:
        return
    with _connect(db_path) as conn:
        _ensure_signals_table(conn)
        conn.execute(f"UPDATE signals SET {', '.join(f'{k} = ?' for k in valid)} "
                      f"WHERE id = ?", list(valid.values()) + [signal_id])


def get_signals_df(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        _ensure_signals_table(conn)
        rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
    if not rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return pd.DataFrame([dict(r) for r in rows])


def signals_to_csv_bytes(db_path: str = DEFAULT_DB_PATH) -> bytes:
    buf = io.StringIO()
    get_signals_df(db_path).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def import_signals_csv(raw: bytes, db_path: str = DEFAULT_DB_PATH) -> int:
    """Restore tracked signals from an exported CSV. Needed because free
    hosting wipes the database on every redeploy — without this, a track
    record weeks in the making would vanish with the next code push.
    Rows already present (same ticker, direction and created time) are skipped."""
    df = pd.read_csv(io.BytesIO(raw))
    missing = {"ticker", "direction", "entry", "stop", "target", "created_utc"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    added = 0
    with _connect(db_path) as conn:
        _ensure_signals_table(conn)
        for _, row in df.iterrows():
            exists = conn.execute(
                "SELECT 1 FROM signals WHERE ticker = ? AND direction = ? AND created_utc = ?",
                (row["ticker"], row["direction"], row["created_utc"])).fetchone()
            if exists:
                continue
            record = {c: (None if pd.isna(row[c]) else row[c])
                      for c in SIGNAL_COLUMNS if c in row.index and c != "id"}
            fields = list(record)
            conn.execute(f"INSERT INTO signals ({', '.join(fields)}) "
                          f"VALUES ({', '.join('?' for _ in fields)})",
                          [record[f] for f in fields])
            added += 1
    return added


def clear_signals(db_path: str = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        _ensure_signals_table(conn)
        conn.execute("DELETE FROM signals")


# ---------------------------------------------------------------------
# Small key/value store (e.g. the latest backtest evidence), so results
# survive a page refresh. Like everything here, wiped on redeploy.
# ---------------------------------------------------------------------
import json as _json


def _ensure_kv(conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, "
                 "updated_utc TEXT)")


def save_value(key: str, value: Any, db_path: str = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        _ensure_kv(conn)
        conn.execute("INSERT OR REPLACE INTO kv (key, value, updated_utc) VALUES (?, ?, ?)",
                     (key, _json.dumps(value), datetime.now(timezone.utc).isoformat()))


def load_value(key: str, db_path: str = DEFAULT_DB_PATH):
    """Returns (value, updated_utc) or (None, None)."""
    with _connect(db_path) as conn:
        _ensure_kv(conn)
        row = conn.execute("SELECT value, updated_utc FROM kv WHERE key = ?",
                           (key,)).fetchone()
    if not row:
        return None, None
    return _json.loads(row["value"]), row["updated_utc"]


def import_journal_csv(raw: bytes, db_path: str = DEFAULT_DB_PATH) -> int:
    """Restore journal entries from an exported CSV.

    Essential on free hosting, where every redeploy wipes the database: without
    it, weeks of real trades — the most valuable data the learner has — would
    vanish with each app update. Rows already present (same asset, timestamp
    and entry) are skipped, so importing twice is harmless.
    """
    df = pd.read_csv(io.BytesIO(raw))
    if "asset" not in df.columns or "timestamp_utc" not in df.columns:
        raise ValueError("This doesn't look like a journal export (missing asset/timestamp).")
    added = 0
    with _connect(db_path) as conn:
        for _, row in df.iterrows():
            exists = conn.execute(
                "SELECT 1 FROM journal WHERE asset IS ? AND timestamp_utc IS ? AND entry IS ?",
                (None if pd.isna(row.get("asset")) else row.get("asset"),
                 None if pd.isna(row.get("timestamp_utc")) else row.get("timestamp_utc"),
                 None if pd.isna(row.get("entry")) else float(row.get("entry")))).fetchone()
            if exists:
                continue
            record = {}
            for col in JOURNAL_COLUMNS:
                if col == "id" or col not in row.index:
                    continue
                v = row[col]
                record[col] = None if (not isinstance(v, str) and pd.isna(v)) else v
            if not record:
                continue
            fields = list(record)
            conn.execute(f"INSERT INTO journal ({', '.join(fields)}) "
                         f"VALUES ({', '.join('?' for _ in fields)})",
                         [record[f] for f in fields])
            added += 1
    return added
