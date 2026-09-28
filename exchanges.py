"""
Exchange market data: Binance first, Kraken as fallback.

WHY THIS EXISTS: CoinGecko's free tier rate-limits hard (HTTP 429), has no
true daily OHLC (one close per day), and serves no 1-hour or 5-minute
interval. Public exchange endpoints have none of those problems — they are
free, need no key, allow far more requests, and return genuine OHLCV at every
interval this strategy uses.

WHY TWO EXCHANGES: api.binance.com geo-blocks some regions (notably the US)
with HTTP 451. Streamlit Cloud runs in the US, so Binance may be unreachable
there. Kraken's public endpoints have no such restriction, so every call falls
back to Kraken automatically. Whichever answered is reported in the UI rather
than left ambiguous — the two exchanges can differ slightly in price.

WHAT THIS DOES NOT COVER: coins not listed on either exchange. Those still
fall back to CoinGecko. Nothing here touches stocks or commodities, which
remain on Yahoo Finance.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import time
import pandas as pd
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price"
KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"
KRAKEN_TICKER = "https://api.kraken.com/0/public/Ticker"

# Binance uses USDT pairs for almost everything; a few majors also have USD.
BINANCE_QUOTE = "USDT"

# Kraken uses legacy codes for a couple of assets.
KRAKEN_ASSET_ALIASES = {"BTC": "XBT", "DOGE": "XDG"}

# Interval strings per exchange. Kraken takes minutes as an integer.
BINANCE_INTERVALS = {"5m": "5m", "1h": "1h", "4h": "4h", "1d": "1d"}
KRAKEN_INTERVALS = {"5m": 5, "1h": 60, "4h": 240, "1d": 1440}

_MIN_SPACING = 0.15          # exchanges tolerate far more traffic than CoinGecko
_last_call = {"t": 0.0}
_SETTINGS = {"max_retries": 2, "backoff": 1.0, "spacing": _MIN_SPACING}

# A geo-block is permanent for the life of the process: the server's location
# will not change between requests. Retrying one — three attempts with backoff,
# for every coin and every timeframe — turned a 20-coin scan into a 15-minute
# wait. Once a host answers 403 or 451 it is skipped instantly thereafter.
PERMANENT_STATUSES = (401, 403, 451)
_blocked_hosts: Dict[str, str] = {}


def _host_of(url: str) -> str:
    from urllib.parse import urlsplit
    return urlsplit(url).netloc


def blocked_hosts() -> Dict[str, str]:
    """Hosts found unreachable this session, and why."""
    return dict(_blocked_hosts)


def reset_blocked_hosts() -> None:
    _blocked_hosts.clear()


def configure(max_retries: int = 2, backoff: float = 1.0, spacing: float = 0.15) -> None:
    """Tune retry behaviour. Tests set these to zero so no real sleeping occurs."""
    _SETTINGS["max_retries"] = max_retries
    _SETTINGS["backoff"] = backoff
    _SETTINGS["spacing"] = spacing


def _get(url, params, timeout=10):
    """GET with light spacing and retry. Returns (response, error).

    Hosts that have already refused on location grounds are skipped without a
    request or a pause.
    """
    host = _host_of(url)
    if host in _blocked_hosts:
        return None, _blocked_hosts[host]

    last_error = None
    for attempt in range(_SETTINGS["max_retries"] + 1):
        wait = _SETTINGS["spacing"] - (time.time() - _last_call["t"])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            _last_call["t"] = time.time()
            if resp.status_code in PERMANENT_STATUSES:
                msg = (f"{resp.status_code} — this exchange is geo-blocked from this "
                       f"server.")
                _blocked_hosts[host] = msg      # never try this host again
                return None, msg
            if resp.status_code == 429:
                last_error = "429 rate limited"
                if attempt < _SETTINGS["max_retries"]:
                    time.sleep(_SETTINGS["backoff"] * (attempt + 1))
                    continue
                return None, last_error
            resp.raise_for_status()
            return resp, None
        except Exception as e:
            _last_call["t"] = time.time()
            last_error = f"{type(e).__name__}: {e}"
            if attempt < _SETTINGS["max_retries"]:
                time.sleep(_SETTINGS["backoff"] * (attempt + 1))
                continue
    return None, last_error


def base_asset(ticker: str) -> str:
    """'BTC-USD' -> 'BTC', and 'HL:SOL' -> 'SOL'.

    The venue prefix has to come off too: without it, a Hyperliquid ticker
    became 'HL:SOLUSDT' on Bybit and 'HL:SOL-USD' on Coinbase — symbols no
    exchange recognises, so those coins silently fell through to another
    source.
    """
    raw = ticker.upper().strip()
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    return raw.replace("-USD", "").replace("USD", "").strip()


def to_binance_symbol(ticker: str) -> str:
    return f"{base_asset(ticker)}{BINANCE_QUOTE}"


def to_kraken_pair(ticker: str) -> str:
    asset = base_asset(ticker)
    return f"{KRAKEN_ASSET_ALIASES.get(asset, asset)}USD"


# ---------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------

def fetch_binance_klines(ticker: str, interval: str, limit: int = 500
                          ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """True OHLCV candles from Binance. Returns (df, error)."""
    if interval not in BINANCE_INTERVALS:
        return None, f"Unsupported interval {interval!r}"
    resp, err = _get(BINANCE_KLINES, {
        "symbol": to_binance_symbol(ticker),
        "interval": BINANCE_INTERVALS[interval],
        "limit": limit,
    })
    if resp is None:
        return None, err
    try:
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return None, "Binance returned no candles (symbol may not be listed)."
        df = pd.DataFrame(rows, columns=[
            "openTime", "Open", "High", "Low", "Close", "Volume",
            "closeTime", "qav", "trades", "tbb", "tbq", "ignore"])
        df["ts"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df[df["Close"].notna()]
        return (df, None) if not df.empty else (None, "All candles had null closes.")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_binance_price(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    resp, err = _get(BINANCE_TICKER, {"symbol": to_binance_symbol(ticker)})
    if resp is None:
        return None, err
    try:
        price = float(resp.json().get("price"))
        return (price, None) if price > 0 else (None, f"Non-positive price {price}")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------
# Kraken
# ---------------------------------------------------------------------

def _kraken_result(payload):
    """Kraken reports errors in a JSON field, not the HTTP status."""
    errors = payload.get("error") or []
    if errors:
        return None, "; ".join(str(e) for e in errors)
    result = payload.get("result") or {}
    data_keys = [k for k in result if k != "last"]
    if not data_keys:
        return None, "Kraken returned no data for this pair."
    return result[data_keys[0]], None


def fetch_kraken_ohlc(ticker: str, interval: str
                       ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if interval not in KRAKEN_INTERVALS:
        return None, f"Unsupported interval {interval!r}"
    resp, err = _get(KRAKEN_OHLC, {
        "pair": to_kraken_pair(ticker),
        "interval": KRAKEN_INTERVALS[interval],
    })
    if resp is None:
        return None, err
    try:
        rows, kerr = _kraken_result(resp.json())
        if rows is None:
            return None, kerr
        df = pd.DataFrame(rows, columns=[
            "time", "Open", "High", "Low", "Close", "vwap", "Volume", "count"])
        df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df[df["Close"].notna()]
        return (df, None) if not df.empty else (None, "All candles had null closes.")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_kraken_price(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    resp, err = _get(KRAKEN_TICKER, {"pair": to_kraken_pair(ticker)})
    if resp is None:
        return None, err
    try:
        rows, kerr = _kraken_result(resp.json())
        if rows is None:
            return None, kerr
        price = float(rows["c"][0])   # 'c' is [last trade price, lot volume]
        return (price, None) if price > 0 else (None, f"Non-positive price {price}")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------
# Combined interface
# ---------------------------------------------------------------------

@dataclass
class ExchangeResult:
    frames: Dict[str, pd.DataFrame]
    source: Optional[str]          # "Binance" | "Kraken" | None
    problems: List[str]


def fetch_spot(ticker: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Spot price. Returns (price, source, error)."""
    price, b_err = fetch_binance_price(ticker)
    if price is not None:
        return price, "Binance", None
    price, k_err = fetch_kraken_price(ticker)
    if price is not None:
        return price, "Kraken", None
    return None, None, f"Binance: {b_err} | Kraken: {k_err}"


def build_frames(ticker: str) -> ExchangeResult:
    """All four timeframes as true OHLCV, from whichever exchange answers.

    Both exchanges are tried per timeframe, but the source is pinned to
    whichever served the 4H frame so a single scan does not silently mix
    prices from two venues.
    """
    problems: List[str] = []
    wanted = ["1d", "4h", "1h", "5m"]

    # Decide the venue on the 4H frame — the strategy's primary timeframe.
    frames: Dict[str, pd.DataFrame] = {}
    df, b_err = fetch_binance_klines(ticker, "4h")
    source = None
    if df is not None:
        source = "Binance"
        frames["4h"] = df
    else:
        df, k_err = fetch_kraken_ohlc(ticker, "4h")
        if df is not None:
            source = "Kraken"
            frames["4h"] = df
            problems.append(f"Binance unavailable ({b_err}) — using Kraken.")
        else:
            return ExchangeResult(
                frames={k: pd.DataFrame() for k in wanted}, source=None,
                problems=[f"Neither exchange served 4H candles. "
                          f"Binance: {b_err} | Kraken: {k_err}"])

    fetch = fetch_binance_klines if source == "Binance" else fetch_kraken_ohlc
    for tf in wanted:
        if tf in frames:
            continue
        got, err = fetch(ticker, tf)
        if got is None:
            frames[tf] = pd.DataFrame()
            problems.append(f"{source} {tf} unavailable: {err}")
        else:
            frames[tf] = got

    return ExchangeResult(frames=frames, source=source, problems=problems)


INTERVAL_MS = {"5m": 300_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def fetch_binance_history(ticker: str, interval: str, total_bars: int
                           ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """More than 1,000 candles by paging backwards through Binance history.

    Binance caps one request at 1,000 candles. A backtest wanting ~160 days of
    1H candles needs ~3,800, so this walks backwards in 1,000-candle pages and
    stitches them together, oldest first, with duplicates removed.
    """
    if interval not in BINANCE_INTERVALS:
        return None, f"Unsupported interval {interval!r}"
    frames = []
    end_time = None
    remaining = total_bars
    while remaining > 0:
        params = {"symbol": to_binance_symbol(ticker),
                  "interval": BINANCE_INTERVALS[interval],
                  "limit": min(1000, remaining)}
        if end_time is not None:
            params["endTime"] = end_time
        resp, err = _get(BINANCE_KLINES, params)
        if resp is None:
            if frames:
                break          # keep what we have rather than discard it
            return None, err
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            break
        df = pd.DataFrame(rows, columns=[
            "openTime", "Open", "High", "Low", "Close", "Volume",
            "closeTime", "qav", "trades", "tbb", "tbq", "ignore"])
        frames.append(df)
        remaining -= len(rows)
        earliest = int(df["openTime"].iloc[0])
        end_time = earliest - 1
        if len(rows) < params["limit"]:
            break              # reached the start of the listing
    if not frames:
        return None, "Binance returned no candles."
    raw = pd.concat(frames).drop_duplicates(subset="openTime")
    raw["ts"] = pd.to_datetime(raw["openTime"], unit="ms", utc=True)
    out = raw.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    out = out[out["Close"].notna()].sort_index()
    return out, None


# ---------------------------------------------------------------------
# Bybit
#
# Bybit blocks requests from some regions (notably the US) with HTTP 403.
# Streamlit Cloud is US-hosted, so these calls usually fail there while
# working fine from a local machine elsewhere. api.bytick.com is Bybit's
# alternate domain and is tried automatically in case it is reachable.
# ---------------------------------------------------------------------

BYBIT_HOSTS = ["https://api.bybit.com", "https://api.bytick.com"]
BYBIT_INTERVALS = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}


def to_bybit_symbol(ticker: str) -> str:
    return f"{base_asset(ticker)}USDT"


def _bybit_get(path: str, params: dict):
    """Try each Bybit host. Returns (payload, error)."""
    last_error = None
    for host in BYBIT_HOSTS:
        resp, err = _get(host + path, params)
        if resp is None:
            last_error = err
            continue
        try:
            payload = resp.json()
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
        if payload.get("retCode") not in (0, None):
            last_error = f"Bybit error: {payload.get('retMsg')}"
            continue
        return payload, None
    if last_error and ("403" in last_error or "451" in last_error):
        last_error = ("Bybit blocks requests from this server's location (the app is "
                      "hosted in the US). It works from a machine in a region Bybit serves.")
    return None, last_error


def fetch_bybit_tickers() -> Tuple[Dict[str, dict], Optional[str]]:
    """Every USDT perp with price, 24h change, volume, OI and funding."""
    payload, err = _bybit_get("/v5/market/tickers", {"category": "linear"})
    if payload is None:
        return {}, err
    out = {}
    for row in (payload.get("result") or {}).get("list") or []:
        sym = row.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        try:
            last = float(row.get("lastPrice") or 0)
            prev = float(row.get("prevPrice24h") or 0)
            if last <= 0:
                continue
            out[sym[:-4]] = {
                "symbol": sym, "last": last, "prev_24h": prev,
                "change_pct": float(row.get("price24hPcnt") or 0) * 100,
                "change_abs": last - prev if prev else 0.0,
                "high_24h": float(row.get("highPrice24h") or 0),
                "low_24h": float(row.get("lowPrice24h") or 0),
                "turnover_24h": float(row.get("turnover24h") or 0),
                "open_interest_usd": float(row.get("openInterestValue") or 0),
                "funding_rate": float(row.get("fundingRate") or 0),
            }
        except (TypeError, ValueError):
            continue
    return (out, None) if out else ({}, "Bybit returned no USDT perpetuals.")


def fetch_bybit_klines(ticker: str, interval: str, limit: int = 500
                        ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """OHLCV candles from Bybit, oldest first (Bybit returns newest first)."""
    if interval not in BYBIT_INTERVALS:
        return None, f"Unsupported interval {interval!r}"
    payload, err = _bybit_get("/v5/market/kline", {
        "category": "linear", "symbol": to_bybit_symbol(ticker),
        "interval": BYBIT_INTERVALS[interval], "limit": min(limit, 1000)})
    if payload is None:
        return None, err
    rows = (payload.get("result") or {}).get("list") or []
    if not rows:
        return None, "Bybit returned no candles (symbol may not be listed)."
    try:
        df = pd.DataFrame(rows, columns=["start", "Open", "High", "Low", "Close",
                                          "Volume", "turnover"])
        df["ts"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df[df["Close"].notna()].sort_index()
    except (KeyError, ValueError, TypeError) as e:
        return None, f"Malformed Bybit candles: {type(e).__name__}"
    return (df, None) if not df.empty else (None, "All Bybit candles were empty.")


def fetch_bybit_price(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    payload, err = _bybit_get("/v5/market/tickers",
                              {"category": "linear", "symbol": to_bybit_symbol(ticker)})
    if payload is None:
        return None, err
    rows = (payload.get("result") or {}).get("list") or []
    if not rows:
        return None, "Not listed on Bybit."
    try:
        price = float(rows[0].get("lastPrice"))
        return (price, None) if price > 0 else (None, "Non-positive price.")
    except (TypeError, ValueError) as e:
        return None, f"{type(e).__name__}"


# ---------------------------------------------------------------------
# Coinbase
#
# US-hosted and therefore reachable from Streamlit Cloud, unlike Bybit and
# Binance. Coverage is smaller than Binance but good for majors, so it fills
# the gap left when the two big venues refuse.
# ---------------------------------------------------------------------

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{product}/candles"
COINBASE_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 21600,
                        "1d": 86400}


def to_coinbase_product(ticker: str) -> str:
    return f"{base_asset(ticker)}-USD"


def fetch_coinbase_candles(ticker: str, interval: str, limit: int = 300
                            ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """OHLCV candles from Coinbase, oldest first.

    Note Coinbase has no true 4h granularity — 21600 is six hours — so 4h is
    not offered here; build it from 5m or 1h instead.
    """
    if interval not in COINBASE_GRANULARITY or interval == "4h":
        return None, f"Coinbase has no {interval} candles."
    resp, err = _get(COINBASE_CANDLES.format(product=to_coinbase_product(ticker)),
                     {"granularity": COINBASE_GRANULARITY[interval]})
    if resp is None:
        return None, err
    try:
        rows = resp.json()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(rows, list) or not rows:
        return None, "Not listed on Coinbase."
    try:
        df = pd.DataFrame(rows, columns=["time", "Low", "High", "Open", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df.sort_index().tail(limit)
    except (KeyError, ValueError, TypeError) as e:
        return None, f"Malformed Coinbase candles: {type(e).__name__}"
    return (df, None) if not df.empty else (None, "Coinbase returned no candles.")
