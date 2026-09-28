"""
Crypto Fear & Greed index (0 = extreme fear, 100 = extreme greed).

TWO SOURCES: CoinMarketCap publishes its own index but requires an API key, so
it is used only when COINMARKETCAP_API_KEY is configured. Otherwise the free
Alternative.me feed is used — the original and most widely quoted version of
this index, needing no key. Whichever answered is always named, because the two
can differ by several points on the same day.

WHAT IT IS AND ISN'T: a blend of volatility, momentum, volume, social activity
and BTC dominance, compressed into one number about the WHOLE market. It says
nothing about any individual coin or setup. It is widely used as a contrarian
gauge — extreme fear as a buying zone, extreme greed as a caution flag — but
that is folklore unless tested. This module therefore reports the number and
also records it with each setup, so the learner can find out whether setups
taken in greed or fear actually perform differently for this strategy.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
import requests

ALTERNATIVE_URL = "https://api.alternative.me/fng/"
CMC_URL = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"

_API_KEY = {"value": ""}


def set_cmc_api_key(key: str) -> None:
    _API_KEY["value"] = (key or "").strip()


def has_cmc_key() -> bool:
    return bool(_API_KEY["value"])


def classify(value: float) -> str:
    """Standard bands used by both publishers."""
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


def bucket(value: Optional[float]) -> Optional[str]:
    """Coarser grouping, for the learner — five bands would split the data too
    thinly to judge."""
    if value is None:
        return None
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


@dataclass
class FearGreed:
    value: int
    classification: str
    source: str
    updated_utc: Optional[datetime] = None
    previous: Optional[int] = None          # yesterday, for direction

    @property
    def direction(self) -> Optional[str]:
        if self.previous is None:
            return None
        if self.value > self.previous:
            return "rising"
        if self.value < self.previous:
            return "falling"
        return "flat"


def _from_cmc(timeout: int = 10) -> Tuple[Optional[FearGreed], Optional[str]]:
    try:
        resp = requests.get(CMC_URL, timeout=timeout,
                            headers={"X-CMC_PRO_API_KEY": _API_KEY["value"],
                                     "Accept": "application/json"})
        if resp.status_code in (401, 403):
            return None, "CoinMarketCap rejected the API key."
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or {}
        value = int(float(data["value"]))
        ts = data.get("update_time")
        updated = None
        if ts:
            try:
                updated = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                updated = None
        return FearGreed(value, data.get("value_classification") or classify(value),
                         "CoinMarketCap", updated), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _from_alternative(timeout: int = 10) -> Tuple[Optional[FearGreed], Optional[str]]:
    try:
        resp = requests.get(ALTERNATIVE_URL, params={"limit": 2}, timeout=timeout)
        resp.raise_for_status()
        rows = (resp.json() or {}).get("data") or []
        if not rows:
            return None, "No data returned."
        latest = rows[0]
        value = int(float(latest["value"]))
        prev = int(float(rows[1]["value"])) if len(rows) > 1 else None
        updated = None
        if latest.get("timestamp"):
            try:
                updated = datetime.fromtimestamp(int(latest["timestamp"]), tz=timezone.utc)
            except (ValueError, TypeError):
                updated = None
        return FearGreed(value, latest.get("value_classification") or classify(value),
                         "Alternative.me", updated, prev), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_fear_greed() -> Tuple[Optional[FearGreed], Optional[str]]:
    """CoinMarketCap's index when a key is configured, otherwise the free feed.
    Falls back to the free feed if CoinMarketCap fails, so a bad key doesn't
    leave the app with no reading at all."""
    if has_cmc_key():
        fg, err = _from_cmc()
        if fg is not None:
            return fg, None
        fg2, err2 = _from_alternative()
        if fg2 is not None:
            return fg2, f"CoinMarketCap failed ({err}); showing the free index instead."
        return None, f"CoinMarketCap: {err} | Alternative.me: {err2}"
    return _from_alternative()
