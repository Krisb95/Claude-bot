"""
Turning the universe list into the records this app works with.

WHY THIS IS ITS OWN MODULE: fetch_top_cryptos returns four values — two
dictionaries keyed by display label, a flag and a note — and the app wants a
list of records keyed by symbol. Doing that conversion inline in the page meant
it couldn't be tested, and a wrong unpacking crashed the app on first run
rather than in a test.
"""

from typing import Dict, List, Optional, Tuple

import exchanges
import universe


def top_coins(limit: int = 10, fetch=None) -> Tuple[List[Dict], str, bool]:
    """The top `limit` coins as records.

    Each record: {"label", "ticker", "id", "symbol"} — the display name, the
    Yahoo-style ticker, the CoinGecko id (None when unknown) and the bare
    symbol. Returns (coins, note, came_from_api).
    """
    fetch = fetch or universe.fetch_top_cryptos
    mapping, cg_ids, is_live, note = fetch(limit)
    coins = []
    for label in list(mapping)[:limit]:
        ticker = mapping[label]
        coins.append({"label": label, "ticker": ticker,
                      "id": cg_ids.get(label),
                      "symbol": exchanges.base_asset(ticker)})
    return coins, note, is_live


def reference_ids(coins) -> Tuple[str, ...]:
    """The CoinGecko ids worth asking for a price, de-duplicated and ordered."""
    return tuple(sorted({c["id"] for c in coins if c.get("id")}))


def price_for(coin: Dict, prices: Dict[str, float]) -> Optional[float]:
    """The reference price for one coin, or None when it has no id."""
    coin_id = coin.get("id")
    return prices.get(coin_id) if coin_id else None
