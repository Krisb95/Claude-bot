"""
Hyperliquid market data: 24-hour volume for every perp, plus candles.

WHY HYPERLIQUID'S OWN VOLUME: the trader executes on Hyperliquid, so its volume
is the liquidity they would actually trade into. A coin with huge volume on
other exchanges can still be thin here, and vice versa. Hyperliquid also lists
coins outside the market-cap top 100, which this reaches.

RATE LIMITS (public /info endpoint): 1,200 weight per minute per IP.
metaAndAssetCtxs costs 20; candleSnapshot costs 20 plus extra per 60 candles
returned. Scanning one coin (4H, 1H and 5m candles) costs roughly 70, so about
17 coins per minute is the ceiling. Requests are paced at one per 1.5 seconds
(about 13 coins a minute) to stay safely under it, and a 429 response triggers
a backoff rather than a failure.

"k" COINS: some low-priced coins trade on Hyperliquid in bundles of 1,000 —
kPEPE is 1,000 PEPE, so its price is 1,000x PEPE's. Prices from Hyperliquid
are internally consistent, but must not be mixed with another exchange's
per-coin price for the same token. Tickers from this module are therefore
prefixed "HL:" (e.g. "HL:kPEPE") so every later step knows where to fetch.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import math
import threading
import time
import pandas as pd
import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
PREFIX = "HL:"
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000,
               "4h": 14_400_000, "1d": 86_400_000}

# ~70 weight per coin (three candle requests) at one request per 1.5s is about
# 930 weight a minute — comfortably under the 1,200 limit, with headroom for
# the volume request and the occasional retry.
# Hyperliquid allows 1,200 weight per minute per IP. What matters is the TOTAL
# weight spent in any 60-second window, not the gap between requests — so
# rather than pausing between calls, requests go out as fast as the budget
# allows and only wait when the window is genuinely full. Scanning 20 coins
# costs about 740 weight, which fits inside one window with room to spare.
WEIGHT_LIMIT_PER_MIN = 1200
SAFETY_MARGIN = 0.80                  # only spend 80% of the published limit
INFO_BASE_WEIGHT = 20                 # most /info requests
CANDLES_PER_WEIGHT_UNIT = 60          # candleSnapshot adds 1 per 60 candles

_SETTINGS = {"spacing": 0.0, "max_retries": 3, "backoff": 3.0,
             "weight_limit": WEIGHT_LIMIT_PER_MIN, "workers": 6}
_last = {"t": 0.0}
_spent: List[Tuple[float, int]] = []   # (timestamp, weight) in the last minute
_budget_lock = threading.Lock()


def configure(spacing: float = 0.0, max_retries: int = 3, backoff: float = 3.0,
              weight_limit: int = WEIGHT_LIMIT_PER_MIN, workers: int = 6) -> None:
    """Tune pacing. Tests set everything to zero so nothing actually sleeps."""
    _SETTINGS.update(spacing=spacing, max_retries=max_retries, backoff=backoff,
                     weight_limit=weight_limit, workers=workers)


def candle_weight(n_bars: int) -> int:
    """What a candleSnapshot request costs against the budget."""
    return INFO_BASE_WEIGHT + math.ceil(max(n_bars, 1) / CANDLES_PER_WEIGHT_UNIT)


def _spend(weight: int, max_wait: float = 60.0) -> None:
    """Wait if this request would breach the rolling weight budget, then record it.

    Waits at most once. Looping until the window clears risks spinning forever
    if the clock or the budget misbehaves, and a single wait plus the existing
    429 backoff is enough: overshooting the budget slightly costs one retry,
    while a stuck loop would hang the whole scan.
    """
    allowance = _SETTINGS["weight_limit"] * SAFETY_MARGIN
    with _budget_lock:
        now = time.time()
        _spent[:] = [(t, w) for t, w in _spent if now - t < 60.0]
        used = sum(w for _t, w in _spent)
        wait = 0.0
        if _spent and used + weight > allowance:
            oldest = min(t for t, _w in _spent)
            wait = min(max(0.0, 60.0 - (now - oldest)), max_wait)
    if wait > 0:
        time.sleep(wait)
        with _budget_lock:
            cutoff = time.time()
            _spent[:] = [(t, w) for t, w in _spent if cutoff - t < 60.0]
    with _budget_lock:
        _spent.append((time.time(), weight))


def budget_used() -> int:
    """Weight spent in the last minute — for diagnostics."""
    now = time.time()
    with _budget_lock:
        return sum(w for t, w in _spent if now - t < 60.0)


def _post(body: Dict, timeout: int = 15, weight: int = INFO_BASE_WEIGHT):
    """POST to /info within the weight budget, retrying on 429. Returns (json, error)."""
    last_error = None
    for attempt in range(_SETTINGS["max_retries"] + 1):
        if _SETTINGS["weight_limit"]:
            _spend(weight)
        wait = _SETTINGS["spacing"] - (time.time() - _last["t"])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.post(INFO_URL, json=body, timeout=timeout)
            _last["t"] = time.time()
            if resp.status_code == 429:
                last_error = "429 rate limited by Hyperliquid"
                if attempt < _SETTINGS["max_retries"]:
                    time.sleep(_SETTINGS["backoff"] * (attempt + 1))
                    continue
                return None, last_error
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            _last["t"] = time.time()
            last_error = f"{type(e).__name__}: {e}"
            if attempt < _SETTINGS["max_retries"]:
                time.sleep(_SETTINGS["backoff"] * (attempt + 1))
    return None, last_error


def is_hl_ticker(ticker: str) -> bool:
    return isinstance(ticker, str) and ticker.startswith(PREFIX)


def hl_name(ticker: str) -> str:
    """'HL:kPEPE' -> 'kPEPE'."""
    return ticker[len(PREFIX):] if is_hl_ticker(ticker) else ticker


def base_symbol(name: str) -> str:
    """Underlying token: 'kPEPE' -> 'PEPE', 'BTC' -> 'BTC'."""
    if len(name) > 1 and name[0] == "k" and name[1:].isupper():
        return name[1:]
    return name.upper()


def bundle_size(name: str) -> int:
    """How many tokens one Hyperliquid unit represents (1,000 for 'k' coins)."""
    return 1000 if base_symbol(name) != name.upper() else 1


@dataclass
class MarketContext:
    name: str
    day_volume_usd: float
    mark_price: float
    open_interest_usd: float
    funding_rate: float           # hourly funding rate
    change_24h_pct: Optional[float]
    max_leverage: Optional[int]

    @property
    def ticker(self) -> str:
        return PREFIX + self.name

    @property
    def base(self) -> str:
        return base_symbol(self.name)

    @property
    def label(self) -> str:
        suffix = " (per 1,000)" if bundle_size(self.name) > 1 else ""
        return f"{self.name}{suffix}"


def fetch_market_contexts() -> Tuple[List[MarketContext], Optional[str]]:
    """Every active Hyperliquid perp with its 24h volume, price, OI and funding."""
    data, err = _post({"type": "metaAndAssetCtxs"})
    if data is None:
        return [], err
    try:
        meta, ctxs = data[0], data[1]
        universe = meta.get("universe", [])
    except (IndexError, KeyError, TypeError, AttributeError):
        return [], "Unexpected response shape from Hyperliquid."
    out: List[MarketContext] = []
    for info, ctx in zip(universe, ctxs):
        if info.get("isDelisted"):
            continue
        try:
            mark = float(ctx.get("markPx") or 0)
            if mark <= 0:
                continue
            prev = float(ctx.get("prevDayPx") or 0)
            out.append(MarketContext(
                name=str(info["name"]),
                day_volume_usd=float(ctx.get("dayNtlVlm") or 0),
                mark_price=mark,
                open_interest_usd=float(ctx.get("openInterest") or 0) * mark,
                funding_rate=float(ctx.get("funding") or 0),
                change_24h_pct=((mark - prev) / prev * 100) if prev > 0 else None,
                max_leverage=info.get("maxLeverage")))
        except (TypeError, ValueError, KeyError):
            continue
    if not out:
        return [], "Hyperliquid returned no active markets."
    return out, None


def rank_by_volume(contexts: List[MarketContext], exclude_bases: Set[str],
                   min_volume_usd: float = 0.0, limit: int = 20) -> List[MarketContext]:
    """Highest 24h volume first, skipping excluded tokens and thin markets."""
    excl = {b.upper() for b in exclude_bases}
    kept = [c for c in contexts
            if c.base.upper() not in excl and c.day_volume_usd >= min_volume_usd]
    kept.sort(key=lambda c: c.day_volume_usd, reverse=True)
    return kept[:limit]


def fetch_candles(name: str, interval: str, n_bars: int,
                  end_ms: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """OHLCV candles for one Hyperliquid perp, oldest first, indexed by open
    time in UTC. Hyperliquid serves at most the most recent 5,000 candles."""
    if interval not in INTERVAL_MS:
        return None, f"Unsupported interval {interval!r}"
    end_ms = end_ms or int(time.time() * 1000)
    start_ms = end_ms - INTERVAL_MS[interval] * (n_bars + 1)
    data, err = _post({"type": "candleSnapshot",
                       "req": {"coin": hl_name(name), "interval": interval,
                               "startTime": start_ms, "endTime": end_ms}},
                      weight=candle_weight(n_bars))
    if data is None:
        return None, err
    if not isinstance(data, list) or not data:
        return None, f"No {interval} candles returned for {hl_name(name)}."
    try:
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("ts").rename(columns={"o": "Open", "h": "High", "l": "Low",
                                                 "c": "Close", "v": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df[~df.index.duplicated()].sort_index()
        df = df[df["Close"].notna()]
    except (KeyError, ValueError, TypeError) as e:
        return None, f"Malformed candle data: {type(e).__name__}"
    return (df, None) if not df.empty else (None, "All candles were empty.")


def fetch_candles_many(names: Sequence[str], interval: str, n_bars: int,
                        progress: Optional[Callable[[int, int, str], None]] = None
                        ) -> Dict[str, Optional[pd.DataFrame]]:
    """Candles for several markets at once.

    The rate limit is a weight budget per minute, not a minimum gap between
    calls, so requests run in parallel and only wait when the budget is
    genuinely full. Scanning 20 coins costs roughly 740 of the 1,200 weight
    available, so it finishes in seconds rather than a request at a time.

    The progress callback runs on the CALLING thread, never a worker. Streamlit
    widgets belong to the thread running the script and raise NoSessionContext
    if touched from anywhere else — so workers only fetch, and results are
    collected here as each finishes.

    Failures come back as None for that market rather than raising, so one bad
    symbol can't sink the scan.
    """
    results: Dict[str, Optional[pd.DataFrame]] = {}
    total = len(names)
    if not names:
        return results

    def _one(name):
        df, _err = fetch_candles(name, interval, n_bars)
        return name, df

    workers = max(1, min(_SETTINGS["workers"], len(names)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, n) for n in names]
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                name, df = future.result()
                results[name] = df
            except Exception:
                name = None
            if progress:
                # Reporting must never lose data that was already fetched, so a
                # failing callback is ignored rather than allowed to abort the
                # loop. It runs on this thread, so Streamlit updates work
                # normally — this is only a safety net.
                try:
                    progress(count, total, name or "")
                except Exception:
                    pass
    for n in names:
        results.setdefault(n, None)
    return results
