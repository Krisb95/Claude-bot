"""
Strategy Lab — a second Streamlit app for testing strategies against each other.

WHY SEPARATE: the trading app is for deciding what to do today, and every extra
control on it is one more thing to get wrong in a hurry. This is for the slower
question — do these rules actually make money — which wants different controls,
takes minutes to run, and shouldn't sit next to a live trade plan.

Deploy it as a second app on Streamlit Cloud from the same repository, with the
main file set to `compare_app.py`. It writes nothing, so the two apps can't
interfere with each other.

WHAT IT CANNOT TELL YOU: the 5-minute strategy. Only a couple of days of
5-minute history is reachable from a hosted server, which is far too short to
conclude anything. Run the app on your own machine for that.
"""

import pandas as pd
import streamlit as st

import coingecko
import coinlist
import exchanges
import frames as frame_check
import hyperliquid_data
import mean_reversion
import swing
import theme
import universe
from formatting import format_price

BUILD = "2026-09-28-lab2"

st.set_page_config(page_title="Strategy Lab", page_icon="⚖️", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)
st.title("⚖️ Strategy Lab")
st.caption(f"Build `{BUILD}` · nothing here is saved, and no trades are placed")


@st.cache_data(ttl=3600, show_spinner=False)
def _top_coins(limit):
    return coinlist.top_coins(limit)


@st.cache_data(ttl=300, show_spinner=False)
def _reference_prices(coin_ids):
    return coingecko.fetch_prices_bulk(list(coin_ids))


def _daily_candles(ticker, ref_price=None):
    """Daily candles from the first source that agrees with the reference price."""
    tried = []
    for name, fetch in (
            ("Binance", lambda: exchanges.fetch_binance_klines(ticker, "1d", limit=1000)),
            ("Coinbase", lambda: exchanges.fetch_coinbase_candles(ticker, "1d", limit=300)),
            ("Kraken", lambda: exchanges.fetch_kraken_ohlc(ticker, "1d")),
            ("Hyperliquid", lambda: hyperliquid_data.fetch_candles(
                f"HL:{exchanges.base_asset(ticker)}", "1d", 1000)),
    ):
        df, _err = fetch()
        if df is not None and not df.empty and ref_price:
            close = float(df["Close"].iloc[-1])
            if abs(close - ref_price) / ref_price * 100 > 8:
                continue          # a different market, or a different period
        tried.append((name, df))
    return frame_check.first_valid(tried, "1d", min_bars=150)


def _stats(name, results, target_rr):
    """Summary for one strategy, with the break-even rate beside the win rate."""
    rs = [r for r in results if r is not None]
    if not rs:
        return {"Strategy": name, "Trades": 0, "Win rate": "—",
                "Break-even needs": f"{1 / (1 + target_rr):.0%}", "Avg R": "—",
                "Total R": "—", "Luck range": "—", "_total": 0.0, "_luck": 0.0}
    wins = sum(1 for r in rs if r > 0)
    total = sum(rs)
    sd = float(pd.Series(rs).std(ddof=1)) if len(rs) > 1 else 0.0
    luck = 2 * sd * (len(rs) ** 0.5)
    return {"Strategy": name, "Trades": len(rs), "Win rate": f"{wins / len(rs):.0%}",
            "Break-even needs": f"{1 / (1 + target_rr):.0%}",
            "Avg R": f"{total / len(rs):+.3f}", "Total R": f"{total:+.1f}",
            "Luck range": f"±{luck:.0f}", "_total": total, "_luck": luck}


st.info(
    "Both strategies run on the **same coins, same daily candles, same period and same "
    "costs**, so the only difference is the rules. Read the win rate next to the "
    "break-even column: a small target wins often by construction, which is not the same "
    "as making money."
)

c1, c2, c3 = st.columns(3)
n_coins = c1.selectbox("Coins", [5, 10, 20, 30], index=1,
                       format_func=lambda n: f"Top {n} by market cap")
fee_r = c2.number_input("Costs per trade (R)", 0.0, 0.5, 0.05, 0.01,
                        help="0.05R is 12% of a 0.4R win but under 2% of a 3R win — "
                             "costs hurt small targets far more.")
min_rr = c3.number_input("Swing minimum reward:risk", 1.0, 6.0, 3.0, 0.5)

with st.expander("Mean Reversion settings"):
    m1, m2, m3 = st.columns(3)
    mr_target = m1.number_input("Target (R)", 0.1, 1.0, 0.4, 0.05,
                                help="At 0.4R the break-even win rate is 71%.")
    mr_stop = m2.number_input("Stop beyond the level (ATR)", 0.5, 4.0, 1.5, 0.25)
    mr_ext = m3.number_input("Minimum stretch (ATR)", 0.5, 4.0, 1.5, 0.25)
    mr_reject = st.checkbox("Require a rejection candle", value=False)

if st.button("▶ Run the comparison", use_container_width=True):
    coins, note, _live = _top_coins(n_coins)
    if not coins:
        st.error(f"Couldn't load the coin list. {note or ''}")
    else:
        refs, _err = _reference_prices(coinlist.reference_ids(coins))
        swing_r, mr_r, failures = [], [], []
        bar = st.progress(0.0, text="Starting…")
        for i, coin in enumerate(coins):
            sym = coin["symbol"].upper()
            bar.progress(i / max(len(coins), 1), text=f"{i + 1}/{len(coins)} · {sym}")
            ticker = coin["ticker"]
            ref = coinlist.price_for(coin, refs)
            df, src, why = _daily_candles(ticker, ref)
            if df is None:
                failures.append(f"{sym}: {why or 'no usable daily candles'}")
                continue
            swing_r += [t.r_result for t in swing.run_backtest(
                df, ticker, params=swing.SwingParams(min_rr=min_rr), fee_r=fee_r,
                require_confirmation=False)]
            mr_r += [t["r_result"] for t in mean_reversion.run_backtest(
                df, ticker, fee_r=fee_r,
                params=mean_reversion.MeanReversionParams(
                    target_r=mr_target, stop_atr=mr_stop,
                    min_extension_atr=mr_ext, require_rejection=mr_reject))]
        bar.empty()
        st.session_state.lab = (swing_r, mr_r, failures, min_rr, mr_target)

result = st.session_state.get("lab")
if result:
    swing_r, mr_r, failures, used_rr, used_target = result
    rows = [_stats(f"Swing Levels ({used_rr:g}:1)", swing_r, used_rr),
            _stats(f"Mean Reversion ({used_target:g}:1)", mr_r, used_target)]
    st.dataframe(pd.DataFrame(rows).drop(columns=["_total", "_luck"]),
                 use_container_width=True, hide_index=True)

    if all(r["Trades"] for r in rows):
        best = max(rows, key=lambda r: r["_total"])
        other = min(rows, key=lambda r: r["_total"])
        gap = best["_total"] - other["_total"]
        if abs(gap) < max(best["_luck"], other["_luck"]):
            st.info(
                f"**{best['Strategy']}** finished {gap:+.1f}R ahead, but that sits inside "
                f"the range luck alone produces over this many trades. Neither is shown to "
                f"be better — which is worth knowing before changing how you trade."
            )
        else:
            st.success(f"**{best['Strategy']}** beat {other['Strategy']} by {gap:+.1f}R, "
                       f"by more than luck explains over this sample.")
    else:
        st.warning("One strategy produced no trades. Try more coins, a lower minimum "
                   "reward:risk, or switch the rejection candle off.")

    for f in failures:
        st.caption(f"⚠️ {f}")
    st.caption(
        "A few hundred days on a handful of coins tells you which rules fitted this "
        "period. It does not tell you which will fit the next one, and neither result is "
        "a reason to trade real money without forward tracking first."
    )
else:
    st.caption("Press **Run the comparison** — it takes a minute or two.")
