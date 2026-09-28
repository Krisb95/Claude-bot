"""
Turn a typed list of coin names into tradable markets.

People write watchlists loosely — "rndr", "immutable", "stacks", "pump" — and
some entries are genuinely ambiguous. Guessing silently is the dangerous
option: scanning the wrong coin looks exactly like scanning the right one. So
anything that can't be matched confidently is REPORTED rather than guessed,
and every match shows what it resolved to.

Matching is against the venue's own market list, so a name only resolves if
you can actually trade it there.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import difflib
import re

# The trader's list, as given.
DEFAULT_WATCHLIST = ("RSR, BTC, ETH, SOL, ZEC, HYPE, LIGHTER, TAO, NEAR, PONS, "
                     "FETCH.AI, SUI, UNI, RNDR, COMP, STACKS, IMMUTABLE, LINK, "
                     "XRP, AVAX, INJ, ONDO, WLD, GRT, AAVE, LDO, PYTH, TIA, ARB, ATOM")

# Memecoins are kept apart deliberately. They move on attention rather than
# anything measurable, so mixing them into the main list makes a scan's results
# harder to read — and worth judging separately once forward tracking has
# enough trades to compare the two.
MEME_WATCHLIST = ("FARTCOIN, PUMP, WIF, PEPE, BONK, DOGE, SHIB, POPCAT, MEW, SPX")

WATCHLISTS = {"Main": DEFAULT_WATCHLIST, "Memes": MEME_WATCHLIST}

# Common names and old tickers mapped to the symbol venues use.
ALIASES: Dict[str, str] = {
    "BITCOIN": "BTC", "ETHEREUM": "ETH", "SOLANA": "SOL", "ZCASH": "ZEC",
    "HYPERLIQUID": "HYPE", "BITTENSOR": "TAO", "RIPPLE": "XRP", "AVALANCHE": "AVAX",
    "INJECTIVE": "INJ", "UNISWAP": "UNI", "COMPOUND": "COMP", "CHAINLINK": "LINK",
    "STACKS": "STX", "IMMUTABLE": "IMX", "IMMUTABLEX": "IMX",
    "RNDR": "RENDER", "RENDERTOKEN": "RENDER",
    "RESERVERIGHTS": "RSR", "PUMPFUN": "PUMP", "DERIVE": "DRV", "NEARPROTOCOL": "NEAR",
    "POLKADOT": "DOT", "CARDANO": "ADA", "LITECOIN": "LTC", "DOGECOIN": "DOGE",
    # Confirmed by the trader. Tickers get reused across projects, so these are
    # their call, not a guess: LIT is the market they mean by "Lighter", and DRV
    # by "Derive"/"Derivative".
    "LIGHTER": "LIT", "FETCHAI": "FET", "FETCH": "FET",
}


@dataclass
class Resolved:
    typed: str            # what the user wrote
    symbol: str           # what it resolved to
    market: str           # the venue's market name (e.g. "kPEPE")
    via_alias: bool = False


def _normalise(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def parse_list(text: str) -> List[str]:
    """Split a typed watchlist into entries, keeping the original wording."""
    parts = re.split(r"[,\n;]+", text or "")
    return [p.strip() for p in parts if p.strip()]


def resolve(text: str, markets: Sequence[str],
            aliases: Optional[Dict[str, str]] = None
            ) -> Tuple[List[Resolved], List[str]]:
    """Match typed entries against available markets.

    markets: the venue's market names. Hyperliquid bundles some low-priced
    coins in thousands ("kPEPE" = 1,000 PEPE), so both the market name and its
    underlying symbol are matched.

    Returns (resolved, unmatched). An entry containing several words is tried
    as a whole and word by word, since "immutable xlink" is two coins typed as
    one. Nothing is guessed: anything unmatched comes back for you to correct.
    """
    aliases = {**ALIASES, **(aliases or {})}
    lookup: Dict[str, str] = {}
    for m in markets:
        lookup.setdefault(_normalise(m), m)
        base = m[1:] if len(m) > 1 and m[0] == "k" and m[1:].isupper() else m
        lookup.setdefault(_normalise(base), m)

    out: List[Resolved] = []
    unmatched: List[str] = []
    seen: Set[str] = set()

    def _match(cand: str) -> Optional[Resolved]:
        key = _normalise(cand)
        if not key:
            return None
        alias = aliases.get(key)
        market = lookup.get(key) or (lookup.get(_normalise(alias)) if alias else None)
        if not market:
            return None
        return Resolved(typed=cand, symbol=_normalise(alias or cand), market=market,
                        via_alias=bool(alias and _normalise(alias) != key))

    for entry in parse_list(text):
        whole = _match(entry)
        if whole:
            if whole.market not in seen:
                seen.add(whole.market)
                out.append(whole)
            continue
        if " " in entry:
            # "immutable xlink" is two coins typed as one. Each word is matched
            # separately, and words that DON'T match are still reported — a
            # partly-matched entry must not hide the part that failed.
            any_hit = False
            for word in entry.split():
                hit = _match(word)
                if hit:
                    any_hit = True
                    if hit.market not in seen:
                        seen.add(hit.market)
                        out.append(hit)
                elif _normalise(word):
                    unmatched.append(word)
            if not any_hit and entry not in unmatched:
                pass          # each word already reported above
            continue
        unmatched.append(entry)
    return out, unmatched


def suggest(entry: str, markets: Sequence[str], limit: int = 3) -> List[str]:
    """Near matches for an entry that didn't resolve, to help correct it."""
    key = _normalise(entry)
    if not key:
        return []
    by_norm = {_normalise(m): m for m in markets}
    # Substring matches first, then fuzzy ones, so typos like "LNK" -> LINK work.
    hits = [m for n, m in by_norm.items() if key in n or n in key]
    for n in difflib.get_close_matches(key, list(by_norm), n=limit, cutoff=0.6):
        if by_norm[n] not in hits:
            hits.append(by_norm[n])
    return hits[:limit]


def parse_aliases(text: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse user-defined mappings, one per line: 'lighter = LIT'.

    Lets the trader resolve names the built-in aliases can't, without anyone
    guessing on their behalf. Returns (aliases, bad_lines) so mistakes are
    shown rather than silently ignored.
    """
    aliases: Dict[str, str] = {}
    bad: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[=:>]+", line, maxsplit=1)
        if len(parts) != 2 or not _normalise(parts[0]) or not _normalise(parts[1]):
            bad.append(line)
            continue
        aliases[_normalise(parts[0])] = _normalise(parts[1])
    return aliases, bad


def search_markets(query: str, markets: Sequence[str], limit: int = 40) -> List[str]:
    """Markets whose name contains the query (empty query returns them all)."""
    key = _normalise(query)
    if not key:
        return sorted(markets)[:limit]
    return sorted([m for m in markets if key in _normalise(m)])[:limit]
