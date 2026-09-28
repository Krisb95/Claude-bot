"""
Market data layer with explicit health status. Never silently substitutes a
stale price or another instrument's price — every fetch result is tagged
with a DataStatus and a UTC timestamp, and the caller decides what to do
with anything other than LIVE/DELAYED.

Yahoo Finance has tightened bot detection on its query endpoints, which
causes intermittent 401/429/empty-data responses via yfinance across ALL
instrument types (not one specific ticker). The fix the yfinance maintainers
recommend is impersonating a real browser TLS fingerprint via `curl_cffi`.
This module uses it when available and falls back gracefully if not
installed, so the app doesn't crash — it just loses the reliability boost.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import time
import math

try:
    from curl_cffi import requests as curl_requests
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    _CURL_CFFI_AVAILABLE = False


class DataStatus(str, Enum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class PriceQuote:
    ticker: str
    price: Optional[float]
    fetched_at_utc: Optional[datetime]  # when THIS fetch completed (UTC)
    bar_time_utc: Optional[datetime]    # timestamp of the underlying price bar, if known
    status: DataStatus
    provider: str
    error: Optional[str] = None
    attempts: int = 1
    interval: str = "1d"                # the bar interval actually returned
    bar_interval_seconds: int = 86400   # duration of one bar, for freshness math
    effective_age_seconds: Optional[float] = None  # age AFTER the bar should have closed


# Bar intervals we try, in order of preference, mapped to their duration.
INTERVAL_SECONDS = {
    "1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800,
    "60m": 3600, "1h": 3600, "90m": 5400, "1d": 86400,
}


# Instruments/exchanges where the underlying feed is inherently delayed
# (typical free-tier equity data), independent of freshness-by-age.
KNOWN_DELAYED_ASSET_CLASSES = {"stock"}


_session_cache = {"session": None}


def get_browser_session():
    """Return a cached curl_cffi session impersonating a real browser, or
    None if curl_cffi isn't installed (caller should fall back to plain
    yfinance behavior in that case)."""
    if not _CURL_CFFI_AVAILABLE:
        return None
    if _session_cache["session"] is None:
        _session_cache["session"] = curl_requests.Session(impersonate="chrome")
    return _session_cache["session"]


def curl_cffi_is_available() -> bool:
    return _CURL_CFFI_AVAILABLE


def effective_age(bar_time_utc: datetime, now_utc: datetime,
                   bar_interval_seconds: int) -> float:
    """Age of the data AFTER accounting for the bar's own duration.

    A bar is timestamped at its OPEN, so a bar that is still forming is not
    stale at all — a daily bar opened 6 hours ago is the current bar, not
    6-hour-old data. Effective age is therefore measured from when the bar
    SHOULD have closed (open + interval), floored at zero.
    """
    seconds_since_open = (now_utc - bar_time_utc).total_seconds()
    age = seconds_since_open - bar_interval_seconds
    return max(0.0, age)


def classify_freshness(
    bar_time_utc: Optional[datetime],
    now_utc: datetime,
    asset_class: str,
    live_threshold_seconds: float,
    stale_threshold_seconds: float,
    bar_interval_seconds: int = 0,
    is_daily_fallback: bool = False,
) -> DataStatus:
    """Decide LIVE/DELAYED/STALE, accounting for the bar interval.

    - bar_interval_seconds: duration of one bar. Pass 0 to compare against
      the raw timestamp (legacy behavior).
    - is_daily_fallback: True when we could only get daily candles (intraday
      unavailable). Daily data is never reported as LIVE, because a daily
      candle can't tell you what happened in the last few minutes.
    - Equities are never LIVE either: free equity feeds are exchange-delayed
      by nature, not merely "old".
    """
    if bar_time_utc is None:
        raise ValueError("classify_freshness requires a bar_time_utc")

    age_seconds = effective_age(bar_time_utc, now_utc, bar_interval_seconds)

    if is_daily_fallback:
        # A daily bar that's already closed and more than a day behind is
        # genuinely stale; otherwise it's usable but explicitly not live.
        return DataStatus.STALE if age_seconds > 86400 else DataStatus.DELAYED

    if age_seconds > stale_threshold_seconds:
        return DataStatus.STALE
    if asset_class in KNOWN_DELAYED_ASSET_CLASSES or age_seconds > live_threshold_seconds:
        return DataStatus.DELAYED
    return DataStatus.LIVE


def to_local_display(dt_utc: Optional[datetime], tz) -> str:
    """Format a UTC datetime for display in the given local tzinfo."""
    if dt_utc is None:
        return "—"
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    local = dt_utc.astimezone(tz)
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


def _ticker_obj(ticker: str, yf_module):
    """Build a yfinance Ticker, using the browser-impersonating session when
    curl_cffi is installed."""
    session = get_browser_session()
    if session is not None:
        try:
            return yf_module.Ticker(ticker, session=session)
        except TypeError:
            # Some yfinance versions/mocks don't accept a session kwarg.
            return yf_module.Ticker(ticker)
    return yf_module.Ticker(ticker)


def _clean_history(df):
    """Yahoo frequently returns trailing rows with NaN OHLC (a bar that has
    been allocated but not yet filled). Those must be dropped, not treated
    as a price — NaN fails every comparison, so a naive `price <= 0` check
    lets it straight through and the UI renders '$nan'."""
    if df is None or df.empty:
        return df
    if "Close" not in df.columns:
        return df.iloc[0:0]
    return df[df["Close"].notna()]


def _attempt_fetch(ticker: str, yf_module, use_browser_session: bool = True,
                    interval: str = "5m", period: str = "1d"):
    """One fetch attempt at a given interval. Returns (df_or_None, error_or_None)."""
    try:
        t = _ticker_obj(ticker, yf_module) if use_browser_session else yf_module.Ticker(ticker)
        try:
            hist = t.history(period=period, interval=interval)
        except TypeError:
            # Mocks/older signatures that only accept `period`.
            hist = t.history(period=period)
        return _clean_history(hist), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_quote(
    ticker: str,
    asset_class: str,
    yf_module,
    live_threshold_seconds: float = 60,
    stale_threshold_seconds: float = 900,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> PriceQuote:
    """Fetch a quote via the given yfinance-like module, with explicit
    status classification, browser-session impersonation (via curl_cffi,
    if installed), and a small retry loop for transient failures
    (rate limits, momentary auth hiccups). `yf_module` is injected so this
    function is unit-testable with a fake/mock instead of hitting the network.
    """
    now_utc = datetime.now(timezone.utc)

    if not ticker or not ticker.strip():
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=None, status=DataStatus.UNAVAILABLE,
                           provider="yfinance", error="Empty ticker")

    # Try intraday first so freshness is measured against a short bar. Fall
    # back to daily candles only if intraday returns nothing (market closed,
    # provider limits, or an instrument with no intraday history).
    ladder = [("5m", "1d"), ("15m", "5d"), ("1d", "5d")]

    last_error = None
    hist = None
    attempts = 0
    used_interval = "1d"

    for interval, period in ladder:
        for attempt in range(1, max_retries + 2):
            attempts += 1
            candidate, err = _attempt_fetch(ticker, yf_module, True, interval, period)
            if err is None and candidate is not None and not candidate.empty:
                hist = candidate
                used_interval = interval
                last_error = None
                break
            last_error = err or f"No rows returned at interval {interval}"
            if attempt <= max_retries:
                time.sleep(retry_backoff_seconds * attempt)  # linear backoff
        if hist is not None:
            break

    if hist is None or hist.empty:
        provider_note = "yfinance (curl_cffi)" if curl_cffi_is_available() else "yfinance"
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=None, status=DataStatus.UNAVAILABLE,
                           provider=provider_note, error=last_error, attempts=attempts)

    is_daily_fallback = used_interval == "1d"
    bar_seconds = INTERVAL_SECONDS.get(used_interval, 86400)

    try:
        price = float(hist["Close"].iloc[-1])
        bar_time = hist.index[-1].to_pydatetime()
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        else:
            bar_time = bar_time.astimezone(timezone.utc)
    except Exception as e:
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=None, status=DataStatus.UNAVAILABLE,
                           provider="yfinance", error=f"Malformed data: {type(e).__name__}: {e}",
                           attempts=attempts)

    if not math.isfinite(price) or price <= 0:
        return PriceQuote(ticker=ticker, price=None, fetched_at_utc=now_utc,
                           bar_time_utc=bar_time, status=DataStatus.UNAVAILABLE,
                           provider="yfinance",
                           error=f"Invalid price returned ({price}) — not a usable number.",
                           attempts=attempts, interval=used_interval,
                           bar_interval_seconds=bar_seconds)

    status = classify_freshness(bar_time, now_utc, asset_class,
                                 live_threshold_seconds, stale_threshold_seconds,
                                 bar_interval_seconds=bar_seconds,
                                 is_daily_fallback=is_daily_fallback)
    age = effective_age(bar_time, now_utc, bar_seconds)

    provider_note = "yfinance (curl_cffi)" if curl_cffi_is_available() else "yfinance"
    return PriceQuote(ticker=ticker, price=price, fetched_at_utc=now_utc,
                       bar_time_utc=bar_time, status=status, provider=provider_note,
                       attempts=attempts, interval=used_interval,
                       bar_interval_seconds=bar_seconds, effective_age_seconds=age)
