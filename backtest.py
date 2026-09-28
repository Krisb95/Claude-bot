"""
Backtesting engine.

HONEST SCOPE NOTE: this engine correctly replays a given set of trade
signals against historical OHLC bars without look-ahead (it only acts on
information available at or before the signal bar, and fills happen on a
LATER bar), and it models fees/slippage and computes the requested metrics.
It does NOT itself contain the full multi-timeframe regime/structure/
liquidity strategy from section 6 — wiring that strategy's signal generation
into this engine is real additional work (the strategy has to run bar-by-bar
too, not just be applied to a whole DataFrame, or it reintroduces look-ahead).
What's here is a correct, tested replay/metrics engine plus one example
signal generator (SMA crossover) used only to validate the engine itself.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Callable
import pandas as pd
import numpy as np


@dataclass
class TradeSignal:
    """A signal generated using only data up to and including `signal_index`.
    The engine fills at the NEXT bar's open (signal_index + 1), never at the
    signal bar's own close, to avoid look-ahead."""
    signal_index: int
    direction: Literal["Long", "Short"]
    stop: float
    target: float


@dataclass
class TradeResult:
    entry_index: int
    exit_index: int
    direction: str
    entry_price: float
    exit_price: float
    stop: float
    target: float
    exit_reason: Literal["target", "stop", "end_of_data"]
    r_multiple: float
    pnl: float
    mfe_r: float   # max favorable excursion, in R
    mae_r: float   # max adverse excursion, in R


def run_backtest(
    df: pd.DataFrame,
    signals: List[TradeSignal],
    fee_rate: float = 0.0,
    slippage_pct: float = 0.0,
    risk_per_trade: float = 1.0,
) -> List[TradeResult]:
    """Replay signals against df (must have Open/High/Low/Close, sorted
    ascending by time, integer-positional index 0..n-1). One position at a
    time per signal (signals are independent trades, not a portfolio sim).

    risk_per_trade is an abstract "1R" unit (e.g. dollars) used to normalize
    pnl into R multiples; pass account-currency risk amount if you want pnl
    in that currency too.
    """
    results: List[TradeResult] = []
    n = len(df)
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    for sig in signals:
        entry_index = sig.signal_index + 1
        if entry_index >= n:
            continue  # no bar left to fill on — skip, don't fabricate a fill

        entry_price = float(opens[entry_index])
        # Apply slippage against the trader (worse fill).
        if sig.direction == "Long":
            entry_price *= (1 + slippage_pct)
        else:
            entry_price *= (1 - slippage_pct)

        stop_distance = abs(entry_price - sig.stop)
        if stop_distance == 0:
            continue  # degenerate signal, skip rather than divide by zero

        mfe = 0.0
        mae = 0.0
        exit_index = n - 1
        exit_price = float(closes[-1])
        exit_reason: Literal["target", "stop", "end_of_data"] = "end_of_data"

        for i in range(entry_index, n):
            bar_high = float(highs[i])
            bar_low = float(lows[i])

            if sig.direction == "Long":
                favorable = bar_high - entry_price
                adverse = entry_price - bar_low
            else:
                favorable = entry_price - bar_low
                adverse = bar_high - entry_price

            mfe = max(mfe, favorable / stop_distance)
            mae = max(mae, adverse / stop_distance)

            hit_stop = (bar_low <= sig.stop) if sig.direction == "Long" else (bar_high >= sig.stop)
            hit_target = (bar_high >= sig.target) if sig.direction == "Long" else (bar_low <= sig.target)

            # Conservative assumption when both could have hit intrabar:
            # assume the stop was hit first (worst case, not the flattering one).
            if hit_stop:
                exit_index, exit_price, exit_reason = i, sig.stop, "stop"
                break
            if hit_target:
                exit_index, exit_price, exit_reason = i, sig.target, "target"
                break

        if sig.direction == "Long":
            raw_r = (exit_price - entry_price) / stop_distance
        else:
            raw_r = (entry_price - exit_price) / stop_distance

        fee_cost_r = (fee_rate * 2) * (entry_price / stop_distance)  # round-trip, expressed in R
        r_multiple = raw_r - fee_cost_r
        pnl = r_multiple * risk_per_trade

        results.append(TradeResult(
            entry_index=entry_index, exit_index=exit_index, direction=sig.direction,
            entry_price=entry_price, exit_price=exit_price, stop=sig.stop, target=sig.target,
            exit_reason=exit_reason, r_multiple=r_multiple, pnl=pnl, mfe_r=mfe, mae_r=mae,
        ))

    return results


@dataclass
class BacktestMetrics:
    trade_count: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float
    max_drawdown_r: float
    profit_factor: float
    avg_mfe_r: float
    avg_mae_r: float
    max_losing_streak: int


def compute_metrics(results: List[TradeResult]) -> BacktestMetrics:
    if not results:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    r_values = [t.r_multiple for t in results]
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]

    win_rate = len(wins) / len(r_values)
    avg_win_r = float(np.mean(wins)) if wins else 0.0
    avg_loss_r = float(np.mean(losses)) if losses else 0.0
    expectancy_r = float(np.mean(r_values))

    equity_curve = np.cumsum(r_values)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    max_drawdown_r = float(drawdowns.max()) if len(drawdowns) else 0.0

    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    max_streak = 0
    current_streak = 0
    for r in r_values:
        if r <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return BacktestMetrics(
        trade_count=len(results),
        win_rate=win_rate,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        expectancy_r=expectancy_r,
        max_drawdown_r=max_drawdown_r,
        profit_factor=profit_factor,
        avg_mfe_r=float(np.mean([t.mfe_r for t in results])),
        avg_mae_r=float(np.mean([t.mae_r for t in results])),
        max_losing_streak=max_streak,
    )


def split_in_out_sample(df: pd.DataFrame, split_ratio: float = 0.7):
    """Chronological split — in-sample is the earlier portion, out-of-sample
    is the later portion. No shuffling (that would leak future info)."""
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1")
    cutoff = int(len(df) * split_ratio)
    return df.iloc[:cutoff].reset_index(drop=True), df.iloc[cutoff:].reset_index(drop=True)


def example_sma_crossover_signals(df: pd.DataFrame, fast: int = 10, slow: int = 30,
                                   stop_atr_mult: float = 1.5, target_r: float = 2.0,
                                   atr_period: int = 14) -> List[TradeSignal]:
    """EXAMPLE ONLY — a simple moving-average crossover used to validate the
    backtest engine's mechanics (fills, no-lookahead, fees). This is NOT the
    multi-timeframe regime/structure/liquidity strategy from section 6."""
    closes = df["Close"]
    fast_ma = closes.rolling(fast).mean()
    slow_ma = closes.rolling(slow).mean()

    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    signals: List[TradeSignal] = []
    prev_state = None
    for i in range(1, len(df)):
        if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        state = "above" if fast_ma.iloc[i] > slow_ma.iloc[i] else "below"
        if prev_state is not None and state != prev_state:
            price = float(close.iloc[i])
            atr_val = float(atr.iloc[i])
            if atr_val <= 0:
                prev_state = state
                continue
            if state == "above":
                stop = price - stop_atr_mult * atr_val
                target = price + target_r * stop_atr_mult * atr_val
                signals.append(TradeSignal(signal_index=i, direction="Long", stop=stop, target=target))
            else:
                stop = price + stop_atr_mult * atr_val
                target = price - target_r * stop_atr_mult * atr_val
                signals.append(TradeSignal(signal_index=i, direction="Short", stop=stop, target=target))
        prev_state = state

    return signals
