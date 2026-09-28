"""
Plain-English trade plans.

Turns a setup's numbers into instructions a person can act on: what to do,
where to put the order, where the stop and target go, why, and when to cancel.
Two forms: a one-line summary for the results table, and a full plan card.

Wording is deliberately instructional but never predictive — it describes the
plan and its rules, not an expected outcome.
"""

from typing import List, Optional
from formatting import format_price as fp

AT_ENTRY = "At entry now"
WAIT_RETRACE = "Waiting for retrace"
WAIT_1H = "Waiting for 1H confirmation"
NO_SUPPORT = "No 5m support found"
NO_TREND = "No 4H trend"


def _pct(a: float, b: float) -> str:
    return f"{(a - b) / b * 100:+.2f}%"


def short_plan(direction: Optional[str], stage: Optional[str], entry: Optional[float],
               stop: Optional[float], target: Optional[float], rr: Optional[float],
               price: Optional[float]) -> str:
    """One line for the results table."""
    if not direction:
        return "No trade — the 4H candles aren't trending."
    buy = "buy" if direction == "Long" else "sell"
    rr_txt = f" ({rr:g}R)" if rr else ""

    if stage == NO_TREND:
        return "No trade — the 4H candles aren't trending."
    if entry is None or stop is None or target is None:
        if stage == NO_SUPPORT:
            side = "support below" if direction == "Long" else "resistance above"
            return f"{direction} trend confirmed, but no 5m {side} price yet — no entry."
        return f"{direction} setup incomplete — no entry yet."
    if stage == WAIT_1H:
        word = "bullish" if direction == "Long" else "bearish"
        return (f"{direction} forming — wait for a {word} 1H close, then limit {buy} "
                f"{fp(entry)}, stop {fp(stop)}, TP {fp(target)}{rr_txt}.")
    if stage == AT_ENTRY:
        return (f"{direction}: {buy} now near {fp(entry)}, stop {fp(stop)}, "
                f"TP {fp(target)}{rr_txt}.")
    gap = f" ({_pct(entry, price)})" if price else ""
    return (f"{direction}: limit {buy} {fp(entry)}{gap}, stop {fp(stop)}, "
            f"TP {fp(target)}{rr_txt}.")


def full_plan(ticker: str, direction: Optional[str], stage: Optional[str],
              entry: Optional[float], stop: Optional[float], target: Optional[float],
              rr: Optional[float], price: Optional[float],
              reasons: Optional[List[str]] = None,
              stop_method: str = "", target_method: str = "",
              expiry_text: str = "within 24 hours",
              reentry_steps: Optional[List[str]] = None) -> List[str]:
    """A full plan as markdown lines."""
    lines: List[str] = []
    if not direction or stage == NO_TREND:
        lines.append(f"**{ticker} — no trade.** The last 4H candles aren't making "
                     f"consistent higher highs and lows (or lower, for a short).")
        return lines

    long = direction == "Long"
    side = "BUY" if long else "SELL"
    lines.append(f"**{direction.upper()} {ticker} — {stage or 'setup'}**")

    if reasons:
        lines.append("**Why:** " + " ".join(r.rstrip(".") + "." for r in reasons if r))

    if entry is None or stop is None or target is None:
        lines.append("No complete plan yet — one of your three rules isn't met, so there's "
                     "no entry, stop or target to act on.")
        return lines

    risk = abs(entry - stop)
    reward = abs(target - entry)
    level = "support" if long else "resistance"

    if stage == AT_ENTRY:
        lines.append(f"**Entry:** price is at the {level} now — {side} near **{fp(entry)}**.")
    elif stage == WAIT_1H:
        word = "bullish" if long else "bearish"
        lines.append(f"**Entry:** wait for a {word} 1H candle to close first. Then place a "
                     f"limit {side} at **{fp(entry)}**, the previous 5m {level}.")
    else:
        gap = f", {_pct(entry, price)} from the live price of {fp(price)}" if price else ""
        lines.append(f"**Entry:** place a limit {side} at **{fp(entry)}** — the previous 5m "
                     f"{level}{gap}. Let the retrace come to you; don't {side.lower()} at "
                     f"market.")

    risk_pct = f" ({risk / entry * 100:.2f}% of entry)" if entry else ""
    beyond = "below" if long else "above"
    lines.append(f"**Stop loss:** **{fp(stop)}**"
                 + (f" — {stop_method}" if stop_method else f", {beyond} the {level}")
                 + f". Risk: {fp(risk)} per coin{risk_pct}.")
    lines.append(f"**Take profit:** **{fp(target)}**"
                 + (f" — {target_method}" if target_method else "")
                 + f". Reward: {fp(reward)} per coin"
                 + (f", {rr:g}× the risk." if rr else "."))
    if price is not None:
        short_of_live = (target < price) if long else (target > price)
        if short_of_live:
            where = "below" if long else "above"
            lines.append(
                f"**Note:** the take profit is {where} today's price of {fp(price)}. That's "
                f"intended — this plan takes profit on the bounce from the "
                f"{level}, before price gets back to where it is now. It is still a "
                f"{direction.lower()}.")
    lines.append(f"**Cancel the order if** it isn't filled {expiry_text}, or the 4H trend "
                 f"reverses before it fills.")
    lines.append("**Size it** on the 🧮 Risk tab so a stop-out costs only your chosen risk "
                 "per trade.")
    if reentry_steps:
        lines.append("**If you're stopped out:** " +
                     " ".join(f"({i}) {s.replace('**', '')}" for i, s in
                              enumerate(reentry_steps, 1)))
    return lines
