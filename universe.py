"""
Crypto universe: the top 100 coins by market cap, mapped to Yahoo Finance
'-USD' tickers.

The live list comes from CoinGecko's public API. That API is free and
rate-limited, so the result is cached and there is a static fallback list of
majors. The fallback is clearly reported as such rather than being passed off
as a live top-100 — an out-of-date list is fine, but silently pretending it is
current is not.

Not every CoinGecko symbol exists on Yahoo Finance, and a few collide with
stock tickers. Known bad mappings are excluded rather than left to fail at
fetch time with a confusing "no data" error.
"""

from typing import Dict, Tuple
import requests

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

# Symbols to skip: stablecoins (no meaningful setup), wrapped/staked
# derivatives that track another asset, and symbols Yahoo does not serve
# under the plain '<SYM>-USD' form.
EXCLUDED_SYMBOLS = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "BUSD",
    "WBTC", "WETH", "WEETH", "WSTETH", "STETH", "RETH", "CBBTC", "SOLVBTC",
    "WBETH", "METH", "EZETH", "RSETH", "LBTC", "BSC-USD", "USD0",
}

# Used when the live API is unavailable. Deliberately majors only — this is a
# fallback, not a pretend top-100.
FALLBACK_CRYPTO: Dict[str, str] = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "BNB (BNB)": "BNB-USD",
    "Solana (SOL)": "SOL-USD",
    "XRP (XRP)": "XRP-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Tron (TRX)": "TRX-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Sui (SUI)": "SUI-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Bitcoin Cash (BCH)": "BCH-USD",
    "Near (NEAR)": "NEAR-USD",
    "Aptos (APT)": "APT-USD",
    "Uniswap (UNI)": "UNI-USD",
    "Stellar (XLM)": "XLM-USD",
    "Hedera (HBAR)": "HBAR-USD",
    "Cosmos (ATOM)": "ATOM-USD",
    "Filecoin (FIL)": "FIL-USD",
    "Arbitrum (ARB)": "ARB-USD",
    "Optimism (OP)": "OP-USD",
    "Injective (INJ)": "INJ-USD",
    "Sei (SEI)": "SEI-USD",
    "Render (RENDER)": "RENDER-USD",
    "Immutable (IMX)": "IMX-USD",
    "Algorand (ALGO)": "ALGO-USD",
    "VeChain (VET)": "VET-USD",
    "Ethereum Classic (ETC)": "ETC-USD",
}

# CoinGecko ids for the fallback list, so the CoinGecko data source still works
# when the live universe fetch failed.
FALLBACK_CG_IDS: Dict[str, str] = {
    "Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum", "BNB (BNB)": "binancecoin",
    "Solana (SOL)": "solana", "XRP (XRP)": "ripple", "Cardano (ADA)": "cardano",
    "Dogecoin (DOGE)": "dogecoin", "Tron (TRX)": "tron",
    "Avalanche (AVAX)": "avalanche-2", "Chainlink (LINK)": "chainlink",
    "Polkadot (DOT)": "polkadot", "Sui (SUI)": "sui", "Litecoin (LTC)": "litecoin",
    "Bitcoin Cash (BCH)": "bitcoin-cash", "Near (NEAR)": "near",
    "Aptos (APT)": "aptos", "Uniswap (UNI)": "uniswap", "Stellar (XLM)": "stellar",
    "Hedera (HBAR)": "hedera-hashgraph", "Cosmos (ATOM)": "cosmos",
    "Filecoin (FIL)": "filecoin", "Arbitrum (ARB)": "arbitrum",
    "Optimism (OP)": "optimism", "Injective (INJ)": "injective-protocol",
    "Sei (SEI)": "sei-network", "Render (RENDER)": "render-token",
    "Immutable (IMX)": "immutable-x", "Algorand (ALGO)": "algorand",
    "VeChain (VET)": "vechain", "Ethereum Classic (ETC)": "ethereum-classic",
}


def fetch_top_cryptos(limit: int = 100, timeout: int = 10
                       ) -> Tuple[Dict[str, str], Dict[str, str], bool, str]:
    """Fetch the top `limit` coins by market cap.

    Returns (mapping, coingecko_ids, is_live, note):
      mapping        -> {"Bitcoin (BTC)": "BTC-USD", ...} Yahoo tickers
      coingecko_ids  -> {"Bitcoin (BTC)": "bitcoin", ...} used as a data
                        fallback for coins Yahoo does not list (e.g. HYPE)
      is_live        -> True if from the API, False if the static fallback
      note           -> human-readable explanation, always populated
    """
    try:
        response = requests.get(
            COINGECKO_MARKETS_URL,
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": min(limit, 250),
                "page": 1,
                "sparkline": "false",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        coins = response.json()
        if not isinstance(coins, list) or not coins:
            raise ValueError("Empty or malformed response")

        mapping: Dict[str, str] = {}
        cg_ids: Dict[str, str] = {}
        skipped = 0
        for coin in coins:
            symbol = str(coin.get("symbol", "")).upper().strip()
            name = str(coin.get("name", "")).strip()
            coin_id = str(coin.get("id", "")).strip()
            if not symbol or not name:
                continue
            if symbol in EXCLUDED_SYMBOLS:
                skipped += 1
                continue
            label = f"{name} ({symbol})"
            mapping[label] = f"{symbol}-USD"
            if coin_id:
                cg_ids[label] = coin_id

        if not mapping:
            raise ValueError("No usable symbols after filtering")

        note = (f"Live top {len(mapping)} by market cap from CoinGecko"
                + (f" ({skipped} stablecoins/wrapped assets excluded)." if skipped else "."))
        return mapping, cg_ids, True, note

    except Exception as e:
        note = (f"Could not reach CoinGecko ({type(e).__name__}) — showing a static list of "
                f"{len(FALLBACK_CRYPTO)} majors instead. This is NOT a live top-100.")
        return dict(FALLBACK_CRYPTO), dict(FALLBACK_CG_IDS), False, note
