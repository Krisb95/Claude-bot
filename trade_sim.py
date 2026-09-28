"""
Limit-order trade simulation, shared by the strategy backtest and forward
tracking so both apply identical rules about what counts as a fill, a win,
and a loss.

The strategy places a LIMIT order at a planned entry zone. So a signal is not
a trade until price actually comes back to the entry. Many planned entries
never fill — that is normal and is reported as EXPIRED, not hidden.

CONSERVATIVE ASSUMPTIONS, chosen so results err pessimistic rather than
flattering. With only OHLC bars we cannot see the order of events inside a
bar, so whenever a bar is ambiguous the worse outcome is assumed:

  * On the bar that fills the order, if that same bar also reaches the stop,
    it is counted as a LOSS (price may have filled you and then stopped you).
  * On the fill bar, reaching the target is NOT counted — price may have
    touched the target before dipping to your entry, in which case you would
    never have been filled at all.
  * After the fill, if one bar touches both stop and target, the STOP wins.

These assumptions make results worse than a tick-by-tick replay would. That is
deliberate: an optimistic backtest is worse than none.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

PENDING = "PENDING"
FILLED = "FILLED"      # entered, not yet resolved (still open at end of data)
WIN = "WIN"
LOSS = "LOSS"
EXPIRED = "EXPIRED"    # limit order never filled within its validity window


@dataclass
class SimResult:
    status: str
    fill_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    r_result: Optional[float] = None      # +planned R on a win, -1 on a loss
    bars_to_fill: Optional[int] = None
    bars_in_trade: Optional[int] = None
    last_r: Optional[float] = None        # mark-to-market R if still open


def planned_reward_r(direction: str, entry: float, stop: float, target: float) -> Optional[float]:
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return abs(target - entry) / risk


def simulate_limit_trade(bars: pd.DataFrame, direction: str, entry: float,
                          stop: float, target: float, expiry_bars: int,
                          fee_r: float = 0.0, breakeven_at_r: Optional[float] = None,
                          trail_atr_mult: Optional[float] = None,
                          atr_value: Optional[float] = None) -> SimResult:
    """Replay a limit order against bars that occur STRICTLY AFTER the signal.

    bars:        OHLC dataframe, oldest first, containing only post-signal bars
    expiry_bars: how many bars the limit order may rest before being cancelled
    fee_r:       round-trip fees+slippage expressed in R, subtracted from results
    """
    if direction not in ("Long", "Short"):
        raise ValueError(f"direction must be Long or Short, got {direction!r}")
    if bars is None or bars.empty:
        return SimResult(status=PENDING)

    reward_r = planned_reward_r(direction, entry, stop, target)
    if reward_r is None:
        raise ValueError("entry and stop must differ")

    long = direction == "Long"
    highs, lows = bars["High"].values, bars["Low"].values
    closes = bars["Close"].values
    index = bars.index
    n = len(bars)

    # --- phase 1: wait for the limit order to fill ----------------------
    fill_i = None
    for i in range(min(n, expiry_bars)):
        touched = lows[i] <= entry if long else highs[i] >= entry
        if touched:
            fill_i = i
            break

    if fill_i is None:
        if n >= expiry_bars:
            return SimResult(status=EXPIRED)
        return SimResult(status=PENDING)   # still inside its validity window

    # Conservative: a fill bar that also reaches the stop is a loss.
    stopped_on_fill = lows[fill_i] <= stop if long else highs[fill_i] >= stop
    if stopped_on_fill:
        return SimResult(status=LOSS, fill_time=index[fill_i], exit_time=index[fill_i],
                          r_result=-1.0 - fee_r, bars_to_fill=fill_i, bars_in_trade=0)

    # --- phase 2: manage the open position ------------------------------
    # Optional stop management. The stop is moved using only bars that have
    # ALREADY closed, then tested on the next one — moving it with the same
    # bar's high and then testing that bar's low would be using the future.
    risk0 = abs(entry - stop)
    stop_now = stop
    best_favourable = 0.0

    for j in range(fill_i + 1, n):
        hit_stop = lows[j] <= stop_now if long else highs[j] >= stop_now
        hit_target = highs[j] >= target if long else lows[j] <= target
        if hit_stop:   # checked first: ambiguous bars resolve to the stop
            # A trailed stop can sit beyond entry, so this is not always -1R.
            moved = (stop_now - entry) if long else (entry - stop_now)
            r = (moved / risk0) if risk0 else -1.0
            return SimResult(status=WIN if r > 0 else LOSS, fill_time=index[fill_i],
                              exit_time=index[j], r_result=r - fee_r,
                              bars_to_fill=fill_i, bars_in_trade=j - fill_i)
        if hit_target:
            return SimResult(status=WIN, fill_time=index[fill_i], exit_time=index[j],
                              r_result=reward_r - fee_r, bars_to_fill=fill_i,
                              bars_in_trade=j - fill_i)

        favourable = (highs[j] - entry) if long else (entry - lows[j])
        best_favourable = max(best_favourable, favourable)
        if breakeven_at_r and risk0 and best_favourable / risk0 >= breakeven_at_r:
            stop_now = max(stop_now, entry) if long else min(stop_now, entry)
        if trail_atr_mult and atr_value:
            candidate = ((highs[j] - trail_atr_mult * atr_value) if long
                         else (lows[j] + trail_atr_mult * atr_value))
            stop_now = max(stop_now, candidate) if long else min(stop_now, candidate)

    # Filled but unresolved by the end of the data.
    risk = abs(entry - stop)
    move = (closes[-1] - entry) if long else (entry - closes[-1])
    return SimResult(status=FILLED, fill_time=index[fill_i], bars_to_fill=fill_i,
                      bars_in_trade=n - 1 - fill_i, last_r=move / risk if risk else 0.0)
