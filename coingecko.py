"""
CoinGecko fallback data source.

Yahoo Finance does not list every crypto in the top 100 — newer listings such
as Hyperliquid (HYPE) return nothing at any interval, which previously left
the app showing UNAVAILABLE with no way forward. CoinGecko serves its own OHLC
candles, so it is used as a fallback whenever Yahoo has no data for a coin.

CANDLE GRANULARITY IS FIXED BY THE API — you cannot request an arbitrary
interval. CoinGecko's /ohlc endpoint returns:
    days=1        -> 30-minute candles
    days=2..30    -> 4-hour candles
    days=31+      -> 4-day candles

That is genuinely useful here: days<=30 gives NATIVE 4H candles (better than
resampling 1H, which is what the Yahoo path has to do), and days=1 gives 30m
for a near-live price. But there is no 1-hour or 5-minute option, so the
"1h" frame is unavailable on this source and the daily frame is approximated
by 4-day candles. Those limits are reported rather than papered over.

The free API is rate-limited (roughly 5-15 calls/minute). Callers should cache.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
import os
import time
import pandas as pd
import requests

# CoinGecko's free tier is tightly rate limited and answers 429 when exceeded.
# Requests are spaced out and retried with exponential backoff rather than
# letting a single 429 knock out a whole frame (which previously cascaded into
# "regime unknown" -> no direction -> every check unevaluated -> 0.0/10).
_MIN_SECONDS_BETWEEN_CALLS = 1.2
_last_call_at = {"t": 0.0}

# Overridable so tests can disable real sleeping (a retry storm with genuine
# backoff makes the suite take minutes).
_RETRY_SETTINGS = {"max_retries": 3, "backoff": 2.0}

# A free CoinGecko "demo" API key raises the rate limit substantially
# (roughly 30 calls/min vs ~5-15 anonymous). Set it via the COINGECKO_API_KEY
# environment variable or Streamlit secrets. Everything works without one —
# you just hit 429s sooner.
_API_KEY = {"value": os.environ.get("COINGECKO_API_KEY", "").strip()}


def set_api_key(key: str) -> None:
    _API_KEY["value"] = (key or "").strip()


def has_api_key() -> bool:
    return bool(_API_KEY["value"])


def _auth_headers():
    return {"x-cg-demo-api-key": _API_KEY["value"]} if _API_KEY["value"] else {}


def configure_retries(max_retries: int = 3, backoff: float = 2.0,
                       min_spacing: float = 1.2) -> None:
    """Tune rate-limit handling. Tests set these to zero for speed."""
    global _MIN_SECONDS_BETWEEN_CALLS
    _RETRY_SETTINGS["max_retries"] = max_retries
    _RETRY_SETTINGS["backoff"] = backoff
    _MIN_SECONDS_BETWEEN_CALLS = min_spacing


def _throttled_get(url, params, timeout, max_retries=None, backoff=None):
    """GET with inter-call spacing and 429-aware exponential backoff."""
    max_retries = _RETRY_SETTINGS["max_retries"] if max_retries is None else max_retries
    backoff = _RETRY_SETTINGS["backoff"] if backoff is None else backoff
    last_error = None
    for attempt in range(max_retries + 1):
        wait = _MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call_at["t"])
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                 headers=_auth_headers())
            _last_call_at["t"] = time.time()
            if resp.status_code == 429:
                last_error = "429 Too Many Requests (CoinGecko free-tier rate limit)"
                if attempt < max_retries:
                    time.sleep(backoff ** (attempt + 1))
                    continue
                return None, last_error
            resp.raise_for_status()
            return resp, None
        except Exception as e:
            _last_call_at["t"] = time.time()
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(backoff ** (attempt + 1))
                continue
    return None, last_error

OHLC_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"

# What each `days` value actually returns, per CoinGecko's documented behaviour.
GRANULARITY_BY_DAYS = {
    1: ("30m", 1800),
    7: ("4h", 14400),
    14: ("4h", 14400),
    30: ("4h", 14400),
    90: ("4d", 345600),
    180: ("4d", 345600),
    365: ("4d", 345600),
}


def fetch_ohlc(coin_id: str, days: int = 30, timeout: int = 10
                ) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
    """Fetch OHLC candles for a CoinGecko coin id.

    Returns (dataframe, granularity_label, error). The dataframe has
    Open/High/Low/Close columns and a UTC DatetimeIndex, matching the shape
    the rest of the app expects from yfinance. Volume is not provided by this
    endpoint, so it is absent rather than faked.
    """
    if days not in GRANULARITY_BY_DAYS:
        days = min(GRANULARITY_BY_DAYS, key=lambda d: abs(d - days))
    label, _seconds = GRANULARITY_BY_DAYS[days]

    try:
        resp, err = _throttled_get(
            OHLC_URL.format(coin_id=coin_id),
            {"vs_currency": "usd", "days": days}, timeout)
        if resp is None:
            return None, label, err
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return None, label, "CoinGecko returned no candles."

        df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        df = df[df["Close"].notna()]
        if df.empty:
            return None, label, "All candles had null closes."
        return df, label, None

    except Exception as e:
        return None, label, f"{type(e).__name__}: {e}"


def fetch_spot_price(coin_id: str, timeout: int = 10
                      ) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
    """Current spot price with its last-updated timestamp.

    Returns (price, updated_at_utc, error). The timestamp is CoinGecko's own
    last_updated_at, not the time of our request — so freshness can be judged
    honestly rather than assumed.
    """
    try:
        resp, err = _throttled_get(
            PRICE_URL,
            {"ids": coin_id, "vs_currencies": "usd",
             "include_last_updated_at": "true"}, timeout)
        if resp is None:
            return None, None, err
        data = resp.json()
        entry = data.get(coin_id)
        if not entry or "usd" not in entry:
            return None, None, f"No price for coin id '{coin_id}'."
        price = float(entry["usd"])
        if not (price > 0):
            return None, None, f"Non-positive price returned: {price}"
        ts = entry.get("last_updated_at")
        updated = (datetime.fromtimestamp(ts, tz=timezone.utc)
                    if isinstance(ts, (int, float)) else None)
        return price, updated, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def build_frames(coin_id: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Assemble the 1d / 4h / 5m-equivalent frames the scanner expects.

    Mapping to what CoinGecko can actually provide:
        "1d" <- 4-day candles over a year  (coarser than true daily)
        "4h" <- native 4-hour candles over 30 days
        "5m" <- 30-minute candles over 1 day (finest available, not 5m)

    Every substitution is reported in the returned problems list so the UI can
    say what the analysis is really based on.
    """
    frames: Dict[str, pd.DataFrame] = {}
    problems: List[str] = []

    daily, label, err = fetch_ohlc(coin_id, days=365)
    if daily is None:
        problems.append(f"CoinGecko daily data unavailable: {err}")
        frames["1d"] = pd.DataFrame()
    else:
        frames["1d"] = daily
        problems.append(
            "1D regime is based on CoinGecko 4-day candles — Yahoo has no data for this "
            "coin and CoinGecko serves no true daily interval. Regime reads will be coarser."
        )

    four_h, label, err = fetch_ohlc(coin_id, days=30)
    if four_h is None:
        problems.append(f"CoinGecko 4H data unavailable: {err}")
        frames["4h"] = pd.DataFrame()
    else:
        frames["4h"] = four_h

    fine, label, err = fetch_ohlc(coin_id, days=1)
    if fine is None:
        problems.append(f"CoinGecko intraday data unavailable: {err}")
        frames["5m"] = pd.DataFrame()
        frames["1h"] = pd.DataFrame()
    else:
        frames["5m"] = fine
        frames["1h"] = fine
        problems.append(
            "1H confirmation uses CoinGecko 30-minute candles (its finest free interval); "
            "there is no 5-minute or 1-hour data on this source."
        )

    return frames, problems


# ---------------------------------------------------------------------
# market_chart: price series, which can be resampled into finer candles
# than the fixed /ohlc buckets allow.
#
# CoinGecko's price-point spacing for market_chart is:
#     days=1      -> ~5-minute points
#     days=2..90  -> hourly points
#     days=91+    -> daily points
#
# Resampling 5-minute points up to 1H produces genuine OHLC (12 points per
# bucket). Resampling to the SAME spacing as the source does not — each bucket
# holds one point, so Open=High=Low=Close. That is flagged wherever it applies,
# because swing detection reads High/Low and will simply be reading closes.
# ---------------------------------------------------------------------

def fetch_price_series(coin_id: str, days: int, timeout: int = 15
                        ) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Fetch a timestamped price series. Returns (series, error)."""
    try:
        resp, err = _throttled_get(
            MARKET_CHART_URL.format(coin_id=coin_id),
            {"vs_currency": "usd", "days": days}, timeout)
        if resp is None:
            return None, err
        data = resp.json()
        points = data.get("prices") if isinstance(data, dict) else None
        if not points:
            return None, "CoinGecko returned no price points."
        df = pd.DataFrame(points, columns=["ts", "price"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        series = df.set_index("ts")["price"].dropna()
        if series.empty:
            return None, "All price points were null."
        return series, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def series_to_ohlc(series: pd.Series, rule: str) -> pd.DataFrame:
    """Resample a price series into OHLC candles at `rule` (e.g. '1h')."""
    if series is None or series.empty:
        return pd.DataFrame()
    out = series.resample(rule).agg(["first", "max", "min", "last"]).dropna()
    out.columns = ["Open", "High", "Low", "Close"]
    return out


def build_frames_v2(coin_id: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Best-quality frame set CoinGecko can provide.

        "1d" <- daily closes over a year (market_chart, days=365)
        "4h" <- NATIVE 4-hour OHLC candles (ohlc, days=30)
        "1h" <- 5-minute points resampled to 1H (real OHLC within each hour)
        "5m" <- 5-minute points (finest available)

    The 4H frame is genuinely better than the Yahoo path, which has to resample
    1H into 4H. The daily frame is weaker: one close per day means
    Open=High=Low=Close, so daily swing detection reads closes only.
    """
    frames: Dict[str, pd.DataFrame] = {}
    problems: List[str] = []

    # --- 4H: native OHLC, the most important frame for this strategy ---
    four_h, _label, err = fetch_ohlc(coin_id, days=30)
    if four_h is None:
        problems.append(f"CoinGecko 4H unavailable: {err}")
        frames["4h"] = pd.DataFrame()
    else:
        frames["4h"] = four_h

    # --- Daily: one close per day ---
    daily_series, err = fetch_price_series(coin_id, days=365)
    if daily_series is None:
        problems.append(f"CoinGecko daily unavailable: {err}")
        frames["1d"] = pd.DataFrame()
    else:
        frames["1d"] = series_to_ohlc(daily_series, "1D")
        problems.append(
            "1D candles are built from one closing price per day, so Open/High/Low all "
            "equal the Close. Daily swing detection is therefore reading closes only — "
            "regime and moving averages are unaffected, but daily wicks are invisible."
        )

    # --- Intraday: 5-minute points, also resampled up to 1H ---
    fine, err = fetch_price_series(coin_id, days=1)
    if fine is None:
        problems.append(f"CoinGecko intraday unavailable: {err}")
        frames["5m"] = pd.DataFrame()
        frames["1h"] = pd.DataFrame()
    else:
        frames["5m"] = series_to_ohlc(fine, "5min")
        frames["1h"] = series_to_ohlc(fine, "1h")

    return frames, problems


MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"


def fetch_period_changes(days: str = "90d", limit: int = 100, timeout: int = 15):
    """{symbol: percent change} over the period, in market-cap order.

    Used for the altcoin season index, which compares how many large coins
    outperformed Bitcoin. Stablecoins are excluded — they never move, so
    including them would drag the index down artificially.
    """
    from universe import EXCLUDED_SYMBOLS
    try:
        resp, err = _throttled_get(MARKETS_URL, {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": min(limit, 250), "page": 1, "sparkline": "false",
            "price_change_percentage": days,
        }, timeout)
        if resp is None:
            return {}, err
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return {}, "CoinGecko returned no market data."
        key = f"price_change_percentage_{days}_in_currency"
        out = {}
        for r in rows:
            sym = str(r.get("symbol", "")).upper()
            change = r.get(key)
            if not sym or change is None or sym in EXCLUDED_SYMBOLS:
                continue
            try:
                out[sym] = float(change)
            except (TypeError, ValueError):
                continue
        return (out, None) if out else ({}, "No usable change data.")
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


SEARCH_URL = "https://api.coingecko.com/api/v3/search"


def search_coins(symbol: str, timeout: int = 10):
    """Coins whose ticker exactly matches `symbol`, best-known first.

    Tickers are reused constantly — several projects use DRV — so this returns
    every exact match with its full name and market-cap rank rather than
    picking one. The caller shows them and lets the trader choose.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return [], "No symbol given."
    try:
        resp, err = _throttled_get(SEARCH_URL, {"query": sym}, timeout)
        if resp is None:
            return [], err
        coins = (resp.json() or {}).get("coins") or []
        exact = [{"id": c.get("id"), "symbol": str(c.get("symbol", "")).upper(),
                  "name": c.get("name"), "rank": c.get("market_cap_rank")}
                 for c in coins if str(c.get("symbol", "")).upper() == sym and c.get("id")]
        exact.sort(key=lambda c: (c["rank"] is None, c["rank"] or 0))
        return (exact, None) if exact else ([], f"No coin on CoinGecko has the ticker {sym}.")
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


COIN_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}"


def fetch_description(coin_id: str, timeout: int = 15):
    """A coin's own description from CoinGecko, trimmed to a readable length.

    One call per coin, which is why the app only does this on request rather
    than for a whole list. Returns (text, homepage, error).
    """
    try:
        resp, err = _throttled_get(COIN_URL.format(coin_id=coin_id), {
            "localization": "false", "tickers": "false", "market_data": "false",
            "community_data": "false", "developer_data": "false", "sparkline": "false",
        }, timeout)
        if resp is None:
            return None, None, err
        data = resp.json() or {}
        text = ((data.get("description") or {}).get("en") or "").strip()
        if not text:
            return None, None, "CoinGecko has no description for this coin."
        import re as _re
        text = _re.sub(r"<[^>]+>", "", text)           # strip the HTML links
        text = _re.sub(r"\s+", " ", text).strip()
        links = (data.get("links") or {}).get("homepage") or []
        home = next((h for h in links if h), None)
        return text, home, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


SIMPLE_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"


def fetch_prices_bulk(coin_ids, timeout: int = 15):
    """Current USD price for many coins in one request.

    CoinGecko aggregates across exchanges, so this is the best available answer
    to "what is this coin actually worth right now" — and it is reachable from
    anywhere, unlike the exchange APIs that block US-hosted servers. Used as the
    reference every candle source is checked against.

    Returns ({coin_id: price}, error).
    """
    ids = [c for c in dict.fromkeys(coin_ids) if c]
    if not ids:
        return {}, None
    out = {}
    # The URL is length-limited, so ask in batches.
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        resp, err = _throttled_get(SIMPLE_PRICE_URL,
                                   {"ids": ",".join(batch), "vs_currencies": "usd"}, timeout)
        if resp is None:
            return out, err
        try:
            for coin_id, payload in (resp.json() or {}).items():
                price = (payload or {}).get("usd")
                if price:
                    out[coin_id] = float(price)
        except Exception as e:
            return out, f"{type(e).__name__}: {e}"
    return out, (None if out else "CoinGecko returned no prices.")
