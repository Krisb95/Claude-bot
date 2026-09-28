"""
What each coin actually does, in plain English.

WHY THIS IS WRITTEN OUT RATHER THAN FETCHED: descriptions come one API call per
coin, so covering a hundred would be slow and rate-limited. These are written
from general knowledge, kept to a sentence, and deliberately plain.

TREAT THEM AS A STARTING POINT, NOT A SOURCE OF TRUTH. Projects pivot, launch
new products and occasionally abandon their original purpose, and these
descriptions do not update themselves. Anything not covered here can be looked
up live from CoinGecko inside the app, and that is also worth doing for
anything you are about to put money into.

Nothing here is a view on whether a coin is a good investment — only what it
is for.
"""

from typing import Dict, Optional, Tuple

# category -> what that category means, in one line
CATEGORIES: Dict[str, str] = {
    "Layer 1": "A base blockchain that runs its own network and settles its own transactions.",
    "Layer 2": "A faster, cheaper network built on top of another chain, usually Ethereum.",
    "Exchange": "Issued by a trading venue; typically gives fee discounts or a share of revenue.",
    "DeFi": "Financial services — lending, trading, derivatives — run by code instead of a company.",
    "Stablecoin": "Designed to hold a steady value, usually one US dollar.",
    "Infrastructure": "Plumbing other crypto projects rely on: data, storage, computing, bridges.",
    "AI": "Aimed at machine learning — shared computing power, models or data.",
    "Meme": "Started as a joke or community token; value rests on attention rather than a product.",
    "Payments": "Built mainly for moving money between people or institutions.",
    "Gaming": "Built for games, virtual worlds or in-game items.",
    "Privacy": "Designed to keep transaction details confidential.",
    "RWA": "Brings real-world assets — bonds, property, credit — onto a blockchain.",
}

# symbol -> (category, one-line description)
COINS: Dict[str, Tuple[str, str]] = {
    "BTC": ("Layer 1", "The original cryptocurrency — a fixed-supply digital money and the "
                        "asset the rest of the market takes its cue from."),
    "ETH": ("Layer 1", "The main programmable blockchain; most DeFi, stablecoins and NFTs "
                        "run on it, and ETH pays the transaction fees."),
    "SOL": ("Layer 1", "A high-speed blockchain built for cheap, fast transactions; popular "
                        "for trading apps and memecoins."),
    "BNB": ("Exchange", "Binance's token — fee discounts on the exchange and gas on its own "
                         "BNB Chain."),
    "XRP": ("Payments", "Built for moving money between financial institutions quickly and "
                         "cheaply across borders."),
    "ADA": ("Layer 1", "A research-led blockchain that ships slowly and deliberately, with a "
                        "focus on formal correctness."),
    "DOGE": ("Meme", "The original joke coin, now a widely held meme asset with no roadmap "
                      "beyond being itself."),
    "TRX": ("Layer 1", "A chain used heavily for moving stablecoins cheaply, especially "
                        "USDT."),
    "AVAX": ("Layer 1", "A fast blockchain whose 'subnets' let institutions run their own "
                         "customised chains."),
    "LINK": ("Infrastructure", "Feeds outside data — prices, weather, events — into "
                                "blockchains, which cannot otherwise see the real world."),
    "DOT": ("Infrastructure", "Connects independent blockchains so they can share security "
                               "and pass messages."),
    "MATIC": ("Layer 2", "Ethereum scaling network offering far cheaper transactions."),
    "POL": ("Layer 2", "Polygon's token, used across its Ethereum scaling networks."),
    "LTC": ("Payments", "An early Bitcoin offshoot aimed at faster, cheaper everyday "
                         "payments."),
    "BCH": ("Payments", "A Bitcoin fork with bigger blocks, intended for cheaper payments."),
    "NEAR": ("Layer 1", "A blockchain built for ease of use — readable account names and "
                         "simple onboarding — now leaning into AI."),
    "APT": ("Layer 1", "A high-throughput chain from ex-Meta engineers using the Move "
                        "language."),
    "SUI": ("Layer 1", "A fast chain, also using Move, designed around owning objects "
                        "directly — aimed at games and consumer apps."),
    "UNI": ("DeFi", "Governs Uniswap, the largest decentralised exchange, where anyone can "
                     "swap tokens without an intermediary."),
    "AAVE": ("DeFi", "Lending and borrowing without a bank; rates set by supply and demand."),
    "COMP": ("DeFi", "An early lending protocol where deposits earn interest and loans are "
                      "over-collateralised."),
    "CRV": ("DeFi", "An exchange specialised in swapping similar assets, like one stablecoin "
                     "for another, with very low slippage."),
    "MKR": ("DeFi", "Governs the system behind the DAI stablecoin."),
    "INJ": ("DeFi", "A blockchain built specifically for trading — order books, derivatives "
                     "and prediction markets."),
    "ONDO": ("RWA", "Brings US Treasuries and similar traditional assets on-chain as "
                     "tokens."),
    "RENDER": ("Infrastructure", "A marketplace for spare GPU power, used for 3D rendering "
                                  "and increasingly AI work."),
    "TAO": ("AI", "A network that pays people for contributing useful machine-learning "
                   "models and computing power."),
    "FET": ("AI", "Autonomous software agents that negotiate and transact on your behalf."),
    "FIL": ("Infrastructure", "Pays people to store other people's files across a "
                               "distributed network."),
    "AR": ("Infrastructure", "Permanent file storage — pay once, stored indefinitely."),
    "ATOM": ("Infrastructure", "The hub of an ecosystem of interconnected app-specific "
                                "blockchains."),
    "STX": ("Layer 2", "Brings smart contracts to Bitcoin, settling on the Bitcoin chain."),
    "IMX": ("Gaming", "An Ethereum scaling network built specifically for games and NFTs."),
    "ZEC": ("Privacy", "Optional shielded transactions that hide sender, receiver and "
                        "amount."),
    "XMR": ("Privacy", "Private by default — all transaction details are obscured."),
    "HYPE": ("Exchange", "Powers Hyperliquid, an on-chain perpetuals exchange; holders "
                          "receive a share of trading fees."),
    "RSR": ("Stablecoin", "Backs a system for launching asset-backed stablecoins."),
    "PUMP": ("Meme", "The token of pump.fun, a platform where anyone can launch a memecoin "
                      "in seconds."),
    "WIF": ("Meme", "A Solana memecoin built around a dog-in-a-hat picture."),
    "PEPE": ("Meme", "An Ethereum memecoin based on the frog cartoon."),
    "SHIB": ("Meme", "A dog-themed memecoin that later added its own Ethereum layer 2."),
    "BONK": ("Meme", "Solana's best-known dog memecoin."),
    "OP": ("Layer 2", "An Ethereum scaling network, and the tech behind several other "
                       "chains."),
    "ARB": ("Layer 2", "The largest Ethereum scaling network by activity."),
    "SEI": ("Layer 1", "A chain optimised for trading, with fast finality."),
    "TIA": ("Infrastructure", "Provides data availability so other chains don't each need "
                               "their own."),
    "ALGO": ("Layer 1", "An academically designed chain focused on instant finality."),
    "VET": ("Infrastructure", "Supply-chain tracking for businesses."),
    "ETC": ("Layer 1", "The original Ethereum chain, preserved after the 2016 split."),
    "HBAR": ("Infrastructure", "An enterprise network governed by a council of large "
                                "companies."),
    "XLM": ("Payments", "Low-cost cross-border payments, often aimed at the unbanked."),
    "USDT": ("Stablecoin", "The most traded dollar-pegged token."),
    "USDC": ("Stablecoin", "A regulated dollar-pegged token backed by cash and Treasuries."),
    "LDO": ("DeFi", "Lets you stake ETH while keeping a tradable token representing it."),
    "JUP": ("DeFi", "Solana's main trade-routing exchange."),
    "ENA": ("Stablecoin", "Runs a synthetic dollar that earns yield from funding rates."),
    "DRV": ("DeFi", "An on-chain options and derivatives protocol."),
    "LIT": ("Infrastructure", "Identity and data-privacy tooling across chains."),
    "GRT": ("Infrastructure", "Indexes blockchain data so apps can query it quickly."),
    "SAND": ("Gaming", "A virtual world where land and items are tokens."),
    "MANA": ("Gaming", "A virtual world with user-owned land."),
    "AXS": ("Gaming", "The token behind Axie Infinity, an early play-to-earn game."),
    "CAKE": ("DeFi", "The main decentralised exchange on BNB Chain."),
    "SNX": ("DeFi", "Issues synthetic versions of real-world assets."),
    "DYDX": ("DeFi", "A decentralised perpetuals exchange."),
    "GMX": ("DeFi", "On-chain perpetuals trading with a shared liquidity pool."),
    "PENDLE": ("DeFi", "Lets traders buy and sell future yield separately from principal."),
    "ORDI": ("Meme", "An early token issued directly on Bitcoin via Ordinals."),
    "WLD": ("Infrastructure", "Proof-of-personhood via iris scanning, to prove you're human "
                               "online."),
    "TON": ("Layer 1", "A chain closely tied to Telegram and its mini-apps."),
    "ICP": ("Infrastructure", "Aims to host entire applications on-chain rather than on "
                               "cloud servers."),
    "KAS": ("Layer 1", "A proof-of-work chain using a block structure that allows fast "
                        "confirmation."),
    "PYTH": ("Infrastructure", "A price oracle fed directly by exchanges and trading firms, "
                                "used by most Solana apps and many other chains."),
    "HNT": ("Infrastructure", "Pays people to run wireless hotspots, building a real "
                               "physical network from consumer hardware."),
    "AKT": ("Infrastructure", "A marketplace for renting server capacity, an open "
                               "alternative to cloud providers."),
    "ENS": ("Infrastructure", "Readable names for wallet addresses, so you can send to "
                               "'name.eth' instead of a long string."),
    "ETHFI": ("DeFi", "Staking that keeps your ETH liquid, and one of the largest pools of "
                       "restaked collateral."),
    "EIGEN": ("Infrastructure", "Lets staked ETH secure other services besides Ethereum "
                                 "itself."),
    "SYRUP": ("RWA", "Institutional lending with a real loan book, on-chain."),
    "RAY": ("DeFi", "A long-running Solana exchange and liquidity venue."),
    "JTO": ("DeFi", "Solana staking that also captures block-ordering revenue."),
    "FARTCOIN": ("Meme", "A Solana memecoin with no product — it trades purely on "
                          "attention and community."),
    "POPCAT": ("Meme", "A Solana memecoin built around a cat meme."),
    "MEW": ("Meme", "A Solana cat-themed memecoin."),
    "SPX": ("Meme", "A memecoin riffing on stock-market culture."),
}


def describe(symbol: str) -> Optional[Tuple[str, str]]:
    """(category, description) for a symbol, or None if not covered."""
    if not symbol:
        return None
    return COINS.get(symbol.upper())


def category_of(symbol: str) -> Optional[str]:
    found = describe(symbol)
    return found[0] if found else None


def coverage(symbols) -> Tuple[int, int]:
    """(covered, total) for a list of symbols."""
    symbols = list(symbols)
    return sum(1 for s in symbols if describe(s)), len(symbols)
