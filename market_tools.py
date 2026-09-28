"""
Market-wide tools: RSI across timeframes, the altcoin season index, and
funding / open-interest leaders.

These describe market CONDITIONS. None of them is a trade signal, and none has
been shown to improve this strategy — they're context you read alongside the
scanner, not a reason to enter. Where a reading has a conventional
interpretation (RSI above 70 as "overbought"), that convention is reported as
convention, not as a prediction: strong trends routinely stay overbought for
weeks while price keeps rising.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

RSI_PERIOD = 14
OVERBOUGHT = 70
OVERSOLD = 30


def rsi(closes: pd.Series, period: int = RSI_PERIOD) -> Optional[float]:
    """Wilder's RSI on the most recent bar, 0-100. None if too little data.

    Uses Wilder's smoothing (the original definition), which differs slightly
    from a simple moving average of gains and losses — charting platforms use
    Wilder's, so this matches what you see on a chart.
    """
    if closes is None or len(closes) < period + 1:
        return None
    delta = pd.Series(closes).astype(float).diff().dropna()
    if delta.empty:
        return None
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    value = 100 - (100 / (1 + rs))
    return float(value) if np.isfinite(value) else None


def rsi_state(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value >= OVERBOUGHT:
        return "Overbought"
    if value <= OVERSOLD:
        return "Oversold"
    return "Neutral"


def rsi_row(frames: Dict[str, pd.DataFrame], period: int = RSI_PERIOD
            ) -> Dict[str, Optional[float]]:
    """RSI per timeframe for one coin: {"1h": 58.2, "4h": None, ...}."""
    out = {}
    for tf, df in (frames or {}).items():
        out[tf] = rsi(df["Close"], period) if df is not None and not df.empty else None
    return out


# ---------------------------------------------------------------------
# Altcoin season index
# ---------------------------------------------------------------------

ALT_SEASON_THRESHOLD = 75      # conventional cut-offs used by the published index
BTC_SEASON_THRESHOLD = 25


@dataclass
class AltSeason:
    index: int                 # % of the sample that beat Bitcoin
    label: str
    sample: int
    outperformers: int
    btc_change_pct: float


def altcoin_season_index(changes: Dict[str, float], btc_symbol: str = "BTC",
                         top_n: int = 50) -> Optional[AltSeason]:
    """Share of the top coins that outperformed Bitcoin over the period.

    changes: {symbol: percent change} in market-cap order, Bitcoin included.
    The published index uses the top 50 excluding stablecoins over 90 days;
    75+ is called "altcoin season", 25 or below "Bitcoin season". It is a
    description of what has already happened, not a forecast.
    """
    if not changes or btc_symbol not in changes:
        return None
    btc = float(changes[btc_symbol])
    alts = [(s, float(c)) for s, c in changes.items() if s != btc_symbol][:top_n]
    if not alts:
        return None
    beat = sum(1 for _, c in alts if c > btc)
    index = round(beat / len(alts) * 100)
    if index >= ALT_SEASON_THRESHOLD:
        label = "Altcoin season"
    elif index <= BTC_SEASON_THRESHOLD:
        label = "Bitcoin season"
    else:
        label = "Neither — mixed"
    return AltSeason(index, label, len(alts), beat, btc)


# ---------------------------------------------------------------------
# Perp flow: funding and open interest
# ---------------------------------------------------------------------

@dataclass
class FlowRow:
    name: str
    funding_rate: float          # hourly
    funding_annual_pct: float
    open_interest_usd: float
    volume_usd: float
    crowded: Optional[str]       # "Longs crowded" / "Shorts crowded" / None


CROWDED_HOURLY = 0.0002          # ~0.02%/h — about 175% a year; clearly stretched


def flow_rows(contexts: Sequence) -> List[FlowRow]:
    """Funding and open interest per market, with a crowding note.

    Positive funding means longs pay shorts — more traders are positioned long,
    and strongly positive funding often precedes a flush of those longs. It
    indicates positioning, not direction: crowded longs can stay crowded while
    price keeps rising.
    """
    out = []
    for c in contexts:
        f = float(getattr(c, "funding_rate", 0.0) or 0.0)
        crowded = None
        if f >= CROWDED_HOURLY:
            crowded = "Longs crowded"
        elif f <= -CROWDED_HOURLY:
            crowded = "Shorts crowded"
        out.append(FlowRow(
            name=getattr(c, "name", "?"), funding_rate=f,
            funding_annual_pct=f * 24 * 365 * 100,
            open_interest_usd=float(getattr(c, "open_interest_usd", 0.0) or 0.0),
            volume_usd=float(getattr(c, "day_volume_usd", 0.0) or 0.0),
            crowded=crowded))
    return out


def most_crowded(rows: Sequence[FlowRow], limit: int = 5) -> Tuple[List[FlowRow], List[FlowRow]]:
    """(most positive funding, most negative funding) — crowded longs, crowded shorts."""
    ranked = sorted(rows, key=lambda r: r.funding_rate, reverse=True)
    return ranked[:limit], list(reversed(ranked[-limit:]))


def turnover_ratio(row: FlowRow) -> Optional[float]:
    """24h volume divided by open interest. High means fast churn relative to
    positions held; low means positions are being sat on."""
    if not row.open_interest_usd:
        return None
    return row.volume_usd / row.open_interest_usd
