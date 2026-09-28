"""
Portfolio-level risk aggregation across manually entered open positions.

HONEST SCOPE NOTE: the correlation check has two layers —
1. A REAL statistical correlation matrix, computed from actual historical
   return series if you supply them (see `correlation_matrix`).
2. A simple rule-based concentration/hedge-mismatch flag (e.g. multiple
   crypto longs alongside a BTC short) for when you don't have return
   series handy. This rule-based flag is a heuristic, not a statistical
   claim, and is labeled as such in its output. A hedge is never
   auto-classified as risk-free — mismatches are only ever flagged as
   "worth reviewing," never resolved automatically.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd


@dataclass
class OpenPosition:
    asset: str
    asset_class: str   # "crypto", "stock", "commodity", etc.
    direction: str      # "Long" or "Short"
    entry: float
    stop: float
    quantity: float
    contract_multiplier: float = 1.0


def risk_to_stop(position: OpenPosition) -> float:
    distance = abs(position.entry - position.stop)
    return distance * position.quantity * position.contract_multiplier


def notional(position: OpenPosition) -> float:
    return position.entry * position.quantity * position.contract_multiplier


@dataclass
class PortfolioRiskSummary:
    total_open_risk: float
    risk_by_asset_class: Dict[str, float]
    directional_exposure_by_class: Dict[str, float]  # net notional, +long / -short
    concentration_warnings: List[str]


def summarize_portfolio_risk(positions: List[OpenPosition]) -> PortfolioRiskSummary:
    total_risk = 0.0
    risk_by_class: Dict[str, float] = {}
    exposure_by_class: Dict[str, float] = {}
    warnings: List[str] = []

    long_crypto_assets = []
    short_crypto_assets = []

    for p in positions:
        r = risk_to_stop(p)
        total_risk += r
        risk_by_class[p.asset_class] = risk_by_class.get(p.asset_class, 0.0) + r

        signed_notional = notional(p) * (1 if p.direction.capitalize() == "Long" else -1)
        exposure_by_class[p.asset_class] = exposure_by_class.get(p.asset_class, 0.0) + signed_notional

        if p.asset_class == "crypto":
            if p.direction.capitalize() == "Long":
                long_crypto_assets.append(p.asset)
            else:
                short_crypto_assets.append(p.asset)

    # Rule-based flag mirroring the spec's own example: multiple crypto
    # longs combined with a BTC short is worth a second look, but is NOT
    # automatically resolved or classified as a "risk-free hedge" here.
    btc_shorts = [a for a in short_crypto_assets if a.upper().startswith("BTC")]
    if btc_shorts and len(long_crypto_assets) >= 2:
        warnings.append(
            f"You're short {', '.join(btc_shorts)} while holding {len(long_crypto_assets)} "
            f"other crypto long(s) ({', '.join(long_crypto_assets)}). Crypto majors often "
            f"move together — review whether this is an intentional hedge or unintended "
            f"directional overlap. This is a heuristic flag, not a statistical claim."
        )

    # Simple same-class, same-direction concentration flag.
    for asset_class, assets_in_class in _group_same_direction(positions).items():
        for direction, assets in assets_in_class.items():
            if len(assets) >= 3:
                warnings.append(
                    f"{len(assets)} separate {direction} positions in {asset_class} "
                    f"({', '.join(assets)}) — concentrated exposure to one asset class/direction."
                )

    return PortfolioRiskSummary(
        total_open_risk=total_risk,
        risk_by_asset_class=risk_by_class,
        directional_exposure_by_class=exposure_by_class,
        concentration_warnings=warnings,
    )


def _group_same_direction(positions: List[OpenPosition]) -> Dict[str, Dict[str, List[str]]]:
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for p in positions:
        d = p.direction.capitalize()
        grouped.setdefault(p.asset_class, {}).setdefault(d, []).append(p.asset)
    return grouped


def correlation_matrix(price_history: Dict[str, pd.Series]) -> pd.DataFrame:
    """Real statistical correlation from actual historical daily returns.
    price_history: {ticker: pd.Series of closes, indexed by date}.
    Requires at least 2 overlapping observations per pair or the result
    will contain NaN for that pair (pandas default — not hidden or faked).
    """
    if not price_history:
        return pd.DataFrame()
    returns = pd.DataFrame({t: s.pct_change() for t, s in price_history.items()})
    return returns.corr()
