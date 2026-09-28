"""
Bull Run Strategy V2 — trading analysis dashboard.

Discretionary trading support tool. It does NOT execute trades, connect to an
exchange, or place/modify orders — you enter everything manually on your own
platform. Setup scores measure checklist confluence only; they are not
win-probability estimates, edge claims, or profitability guarantees.
"""

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import zoneinfo

try:
    from data_layer import fetch_quote, DataStatus, to_local_display, curl_cffi_is_available
    from scoring import (score_setup, weakest_components, trade_policy_for_grade,
                          POSITIVE_COMPONENTS, NEGATIVE_COMPONENTS)
    from readiness import (determine_readiness, ReadinessInputs, Readiness, display_label,
                            ENTRY_SEQUENCE_STAGES, missing_entry_sequence_stages,
                            READINESS_DESCRIPTIONS)
    from risk_calc import (calculate_risk, InvalidRiskInputError,
                            stop_from_pct as calc_stop_from_pct,
                            liquidation_price, stop_is_beyond_liquidation)
    from backtest import (run_backtest, compute_metrics, split_in_out_sample,
                           example_sma_crossover_signals)
    from portfolio import OpenPosition, summarize_portfolio_risk
    from stops import StopManager, suggest_management_label
    from position_math import PositionSnapshot, analyse_scale_in, ScaleVerdict
    from scanner import fetch_multi_timeframe, analyze_candidate, scan_universe
    from strategy_backtest import run_strategy_backtest, stats_by_grade, verdict
    from universe import fetch_top_cryptos
    import coingecko
    import exchanges
    import venues as venues_mod
    import storage
    import tracking
    import trend_retrace
except ImportError as _e:
    # Almost always a part-finished upload: a new app.py alongside older module
    # files. Say so plainly instead of showing a bare ImportError.
    st.error(
        f"**Some files are out of date or missing.**\n\n`{_e}`\n\n"
        f"This happens when only some files were uploaded. Upload **every `.py` file** "
        f"from the latest download together — they're released as a set and expect each "
        f"other's newest versions."
    )
    st.stop()
import expectancy
import explain
import learning
import trade_log
import hyperliquid_data
import trade_review
import sentiment
import watchlist as watchlist_mod
import market_tools
import theme
import coin_info
import grid as grid_mod
import patterns
import uihelpers
import frames as frame_check
from formatting import format_price, format_rr


def _optional(name):
    """Import a module that adds a feature, or return None if it isn't there.

    These arrived in later builds, and a part-finished upload shouldn't take
    the whole app down — the feature that needs the file says what's missing,
    and everything else carries on working.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


stop_manager = _optional("stop_manager")
swing = _optional("swing")
mean_reversion = _optional("mean_reversion")
_MISSING_MODULES = [n for n, m in (("mean_reversion.py", mean_reversion),
                                    ("stop_manager.py", stop_manager),
                                    ("swing.py", swing)) if m is None]

_REQUIRED = {
    coingecko: ['build_frames_v2', 'fetch_description', 'fetch_ohlc', 'fetch_period_changes', 'fetch_prices_bulk', 'fetch_spot_price', 'has_api_key', 'search_coins', 'set_api_key'],
    hyperliquid_data: ['MarketContext', 'fetch_candles', 'fetch_candles_many', 'fetch_market_contexts', 'is_hl_ticker', 'rank_by_volume'],
    watchlist_mod: ['DEFAULT_WATCHLIST', 'MEME_WATCHLIST', '_normalise', 'parse_aliases', 'parse_list', 'resolve', 'search_markets', 'suggest'],
    sentiment: ['bucket', 'fetch_fear_greed', 'set_cmc_api_key'],
    theme: ['AMBER', 'BLUE', 'CSS', 'GRADE_COLOURS', 'GREEN', 'GREY', 'PURPLE', 'pill'],
    storage: ['add_journal_entry', 'clear_all', 'close_journal_entry', 'get_journal_df', 'get_signals_df', 'import_journal_csv', 'import_signals_csv', 'init_db', 'journal_to_csv_bytes', 'load_value', 'save_value', 'signals_to_csv_bytes', 'update_journal_entry', 'update_signal'],
    trade_log: ['OUTCOMES', 'complete_trade', 'default_exit_price', 'learning_trades', 'open_scanner_trades', 'pl_status_for', 'realized_r', 'take_trade'],
    trade_review: ['CLOSE', 'CLOSE_THESIS', 'HOLD', 'STILL_VALID', 'TIGHTEN', 'review', 'setup_still_viable'],
    learning: ['LearnedModel', 'MIN_TRADES', 'learn'],
    expectancy: ['EvidenceBook', 'PROVEN_NEGATIVE', 'PROVEN_POSITIVE', 'TOO_FEW', 'UNPROVEN', 'break_even_win_rate'],
    explain: ['AT_ENTRY', 'NO_TREND', 'WAIT_RETRACE', 'full_plan', 'short_plan'],
    trend_retrace: ['AT_ENTRY', 'STOP_BELOW_4H_CANDLE', 'STOP_BELOW_SUPPORT', 'StrategyParams', 'WAIT_RETRACE', 'analyze', 'atr', 'drop_forming', 'five_minute_levels', 'frames_from_5m', 'reentry_plan_text', 'scan_universe_tr'],
    tracking: ['record_from_ranked', 'record_one', 'tracked_stats', 'tracked_trades', 'update_all'],
    exchanges: ['base_asset', 'blocked_hosts', 'build_frames', 'fetch_binance_history', 'fetch_binance_klines', 'fetch_bybit_klines', 'fetch_bybit_tickers', 'fetch_coinbase_candles', 'fetch_kraken_ohlc', 'fetch_spot'],
    coin_info: ['CATEGORIES', 'COINS', 'coverage', 'describe'],
    grid_mod: ['MIN_LEVELS', 'WEIGHTINGS', 'build_grid', 'usable_levels'],
    patterns: ['bias_of', 'contradicts', 'detect', 'summarise'],
    uihelpers: ['scan_count'],
    swing: ['SwingParams', 'analyze', 'scan_universe'],
    stop_manager: ['suggest'],
    frame_check: ['check', 'first_valid'],
    mean_reversion: ['MeanReversionParams', 'analyze', 'scan_universe'],
}

# Features the app can run without. A missing one degrades that feature only,
# rather than blocking the entire app over a single lagging file.
_OPTIONAL = {(stop_manager, "trailing_plan")}
_REQUIRED = {m: attrs for m, attrs in _REQUIRED.items() if m is not None}

_stale = sorted({f"{m.__name__.split('.')[-1]}.py" for m, attrs in _REQUIRED.items()
                 for a in attrs if not hasattr(m, a) and (m, a) not in _OPTIONAL})
if _stale:
    # A module is present but older than app.py expects. This is always a
    # part-finished upload, and it surfaces as a confusing AttributeError deep
    # in the page, so it is caught here instead.
    _detail = "; ".join(
        f"`{m.__name__.split('.')[-1]}.py` is missing " +
        ", ".join(f"`{a}`" for a in attrs if not hasattr(m, a))
        for m, attrs in _REQUIRED.items()
        if any(not hasattr(m, a) and (m, a) not in _OPTIONAL for a in attrs))
    st.error(
        "**These files are out of date:** " + ", ".join(f"`{f}`" for f in _stale) +
        f"\n\n{_detail}.\n\nUpload those files again from the latest download. Check the "
        f"name has no space in it and that it sits beside `app.py` in the repo root — a "
        f"file saved as `stop manager.py` can't be imported and leaves the old one in use."
    )
    st.stop()

MIN_RR = 3.0   # reward:risk floor — nothing below this is shown, tracked or traded


def _methods(use_tr, params, direction):
    """How the stop and target were set, in words, for the plan text."""
    if not (use_tr and params):
        return "the nearest confirmed swing beyond entry", "the first level clearing your minimum R:R"
    beyond = "below" if direction == "Long" else "above"
    level = "support" if direction == "Long" else "resistance"
    if params.stop_mode == trend_retrace.STOP_BELOW_4H_CANDLE:
        stop_m = f"{beyond} the last closed 4H candle"
    else:
        stop_m = f"{params.stop_atr_mult:g}× the 5m ATR {beyond} the {level}"
    return stop_m, f"{params.target_r:g}× the risk"


def _plan_stage(r, use_tr):
    """The explain-module stage for a ranked row from either strategy."""
    if use_tr:
        return r.entry_status
    if r.direction is None:
        return explain.NO_TREND
    return explain.AT_ENTRY if r.entry_status == "AT_ZONE" else explain.WAIT_RETRACE


def _render_learned(model):
    """Show what the learner found, including what it rejected and why."""
    (st.success if model.rules else st.info)(model.summary)
    for r in model.rules:
        st.markdown(f"**Skip setups where:** {r.description}")
        st.caption(f"　Learning trades: {r.train_avg:+.2f}R each (n={r.train_n}) vs "
                   f"{r.train_rest_avg:+.2f}R for the rest · Unseen trades: "
                   f"{r.test_avg:+.2f}R each (n={r.test_n}) vs {r.test_rest_avg:+.2f}R")
    tested = [c for c in model.candidates if c.train_avg is not None
              and c.verdict.startswith(("Looked", "Confirmed, but"))]
    if tested:
        with st.expander(f"Lessons that looked promising but were rejected ({len(tested)})"):
            st.caption("These appeared in the learning trades but failed on unseen ones — "
                       "exactly the coincidences the hold-out check exists to catch.")
            for c in tested:
                ta = f"{c.test_avg:+.2f}R (n={c.test_n})" if c.test_avg is not None else "n/a"
                st.caption(f"**{c.description}** — learning {c.train_avg:+.2f}R "
                           f"(n={c.train_n}), unseen {ta}. {c.verdict}")


@st.cache_data(ttl=120, show_spinner=False)
def _daily_change(ticker: str, price: float):
    """(change in price, change %, previous close) over the last 24h, or Nones.

    Sourced to match wherever the price came from: Hyperliquid's own previous
    daily price for HL markets, otherwise daily candles from the exchange, and
    Yahoo for stocks and commodities.
    """
    if not price:
        return None, None, None
    prev = None
    if hyperliquid_data.is_hl_ticker(ticker):
        ctxs, _e = _cached_hl_contexts()
        row = next((c for c in ctxs if c["ticker"] == ticker), None)
        if row and row.get("change") is not None:
            prev = price / (1 + row["change"] / 100) if row["change"] != -100 else None
    if prev is None:
        df, _e = exchanges.fetch_binance_klines(ticker, "1d", limit=2)
        if df is None:
            df, _e = exchanges.fetch_kraken_ohlc(ticker, "1d")
        if df is not None and len(df) >= 2:
            prev = float(df["Close"].iloc[-2])
    if prev is None:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            hist = hist[hist["Close"].notna()]
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
        except Exception:
            prev = None
    if not prev:
        return None, None, None
    return price - prev, (price - prev) / prev * 100, prev


@st.cache_data(ttl=300, show_spinner=False)
def _yahoo_frames(ticker: str):
    """5m, 1H and 4H candles for a stock, commodity or FX pair.

    Yahoo has no 4H interval, so 4H is built from 1H bars. For anything that
    doesn't trade round the clock those 4H candles span session breaks, which
    is a real difference from crypto — see the warning shown beside the scan.
    """
    frames, problems = fetch_multi_timeframe(ticker, yf)
    out = {"4h": frames.get("4h", pd.DataFrame()),
           "1h": frames.get("1h", pd.DataFrame()),
           "5m": frames.get("5m", pd.DataFrame())}
    return out, problems


def _backtest_frames(ticker, days):
    """5m, 1H and 4H history for a backtest, from whichever source this server
    can actually reach. Bybit and Binance are blocked from US-hosted servers;
    Hyperliquid is not, but only serves its most recent 5,000 candles — about
    17 days of 5m data. Returns (m5, h1, h4, source, note)."""
    bars5 = days * 288
    m5, err = exchanges.fetch_bybit_klines(ticker, "5m", limit=1000)
    if m5 is not None and len(m5) >= 500:
        h1, _ = exchanges.fetch_bybit_klines(ticker, "1h", limit=1000)
        h4, _ = exchanges.fetch_bybit_klines(ticker, "4h", limit=1000)
        if h1 is not None and h4 is not None:
            return m5, h1, h4, "Bybit", None
    m5, err = exchanges.fetch_binance_history(ticker, "5m", bars5)
    if m5 is not None and not m5.empty:
        h1, _ = exchanges.fetch_binance_history(ticker, "1h", days * 24 + 48)
        h4, _ = exchanges.fetch_binance_history(ticker, "4h", days * 6 + 30)
        if h1 is not None and h4 is not None:
            return m5, h1, h4, "Binance", None
    base = ticker.upper().replace("-USD", "")
    m5, hl_err = hyperliquid_data.fetch_candles(f"HL:{base}", "5m", min(bars5, 5000))
    if m5 is not None and not m5.empty:
        h1, _ = hyperliquid_data.fetch_candles(f"HL:{base}", "1h", 2000)
        h4, _ = hyperliquid_data.fetch_candles(f"HL:{base}", "4h", 1000)
        if h1 is not None and h4 is not None:
            covered = len(m5) * 5 / 60 / 24
            note = (f"Hyperliquid data ({covered:.0f} days — it only keeps its most recent "
                    f"5,000 candles)") if covered < days - 1 else "Hyperliquid data"
            return m5, h1, h4, "Hyperliquid", note
    return None, None, None, None, (f"Bybit/Binance: {err} · Hyperliquid: {hl_err}")


def _size_plan(direction, entry, stop, target, ticker=None):
    """Position size for a plan, from the account settings in the sidebar.
    Returns None when the plan is incomplete or the numbers don't work."""
    if None in (direction, entry, stop) or not entry or not stop:
        return None
    try:
        return calculate_risk(
            account_equity=ACCOUNT_EQUITY, risk_pct=RISK_PCT, entry=float(entry),
            stop=float(stop), direction=direction,
            target=float(target) if target else None, leverage=LEVERAGE,
            ticker=ticker, fee_rate=FEE_RATE, slippage_pct=SLIPPAGE)
    except InvalidRiskInputError:
        return None


def _render_size(res, key=""):
    """Show the size beside a trade plan, so no tab-hopping is needed."""
    if res is None:
        return
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Buy / sell", f"{res.quantity:,.6g}".rstrip("0").rstrip(".") + " units")
    s2.metric("Position value", f"${res.position_notional:,.2f}")
    s3.metric("Risk if stopped", f"-${res.net_loss_at_stop:,.2f}")
    if res.net_profit_at_target is not None:
        s4.metric("Gain at target", f"${res.net_profit_at_target:,.2f}")
    bits = [f"Margin ${res.margin_required:,.2f}"]
    if LEVERAGE > 1:
        bits.append(f"{LEVERAGE:g}x leverage")
    bits.append(f"after ${res.fees_and_slippage_cost:,.2f} costs")
    st.caption(" · ".join(bits) + ". Sized from your sidebar settings: "
               f"${ACCOUNT_EQUITY:,.0f} equity, {RISK_PCT:g}% risk.")
    if res.exceeds_account_equity:
        st.error("Required margin exceeds your account equity.")
    for w in res.warnings:
        st.warning(w)
    if res.liquidation_warning and "could be liquidated" in res.liquidation_warning.lower():
        st.error(res.liquidation_warning)


def _render_grid(key, direction, price, levels, atr_value, target_r):
    """Offer a laddered entry, but only when the chart really shows more than
    one level. A single support means a single entry."""
    usable = grid_mod.usable_levels(levels, direction, price, atr_value or 0.0)
    if len(usable) < grid_mod.MIN_LEVELS:
        return
    if not st.toggle("Ladder the entry across several levels", key=f"gr_on_{key}",
                     help="Your strategy specifies one entry. This spreads it over the "
                          "supports below — optional, and untested by the backtest."):
        return

    c1, c2 = st.columns(2)
    n = c1.slider("Levels", min_value=2, max_value=len(usable), value=min(3, len(usable)),
                  key=f"gr_n_{key}")
    style = c2.selectbox("Sizing", grid_mod.WEIGHTINGS, key=f"gr_w_{key}")
    chosen = usable[:n]
    buffer = (atr_value or 0.0) * (TR_PARAMS.stop_atr_mult if TR_PARAMS else 3.0)
    stop = (min(chosen) - buffer) if direction == "Long" else (max(chosen) + buffer)
    plan = grid_mod.build_grid(direction, chosen, price, stop,
                               risk_amount=ACCOUNT_EQUITY * RISK_PCT / 100,
                               target_r=target_r or 3.0, weighting=style)
    if plan is None:
        st.caption("These levels don't form a valid ladder.")
        return

    st.dataframe(pd.DataFrame([{
        "#": l.index,
        "Limit price": format_price(l.price),
        "From live": f"{l.distance_pct:+.2f}%",
        "Share": f"{l.weight * 100:.0f}%",
        "Units": f"{l.units:,.6g}".rstrip("0").rstrip("."),
        "Value": f"${l.notional:,.2f}",
        "Risk if stopped here": f"${l.cumulative_risk:,.2f}",
    } for l in plan.levels]), use_container_width=True, hide_index=True)

    g1, g2, g3 = st.columns(3)
    g1.metric("Average entry", format_price(plan.average_entry),
              f"vs {format_price(plan.single_entry_price)} single")
    g2.metric("Stop (below all levels)", format_price(plan.stop))
    g3.metric("Target", format_price(plan.target),
              f"{format_rr(plan.reward_risk)}")
    st.caption(f"**All levels fill, then stopped: −${plan.total_risk:,.2f}** — the same as a "
               f"single entry, not multiplied. All fill, then target: "
               f"+${plan.reward_at_target:,.2f}.")
    st.caption(plan.partial_fill_note)
    for w in plan.warnings:
        st.caption(f"⚠️ {w}")
    st.caption("Place each line as its own limit order with the SAME stop. Log each fill "
               "separately in the journal. The backtest only models single entries, so "
               "there's no evidence yet that laddering helps this strategy.")


def _take_trade_widget(key, ticker, direction, entry, stop, target, features=None,
                       score=None, grade=None, reason="", suggested_qty=0.0):
    """'I took this trade' — one tap logs it to the journal.

    It used to be a toggle that merely revealed a Save button further down, and
    that button was easy to miss on a phone: people ticked the box and nothing
    appeared in the journal. Now the button saves immediately using the planned
    levels and your sidebar sizing, and anything that needs correcting — the
    actual fill, the quantity — is editable on the Journal tab afterwards.
    """
    if None in (direction, entry, stop, target):
        return
    saved_id = st.session_state.get(f"tk_saved_{key}")
    if saved_id:
        st.success(f"Logged as journal entry #{saved_id}. Correct the fill price or size "
                   f"on the 📓 Journal tab, and complete it there when it closes.")
        return

    if st.button(f"✅ I took this trade — log it", key=f"tk_go_{key}",
                 use_container_width=True):
        try:
            jid = trade_log.take_trade(
                ticker=ticker, direction=direction, planned_entry=float(entry),
                stop=float(stop), target=float(target),
                actual_entry=float(entry), quantity=float(suggested_qty or 0.0),
                leverage=float(LEVERAGE), features=features, followed_rules=True,
                strategy_config=_config_signature(USE_TR, TR_PARAMS),
                reason=reason, score=score, grade=grade)
            st.session_state[f"tk_saved_{key}"] = jid
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    st.caption(f"Logs it at {format_price(entry)} with "
               f"{(suggested_qty or 0):,.6g}".rstrip("0").rstrip(".")
               + " units from your sidebar risk settings — all editable afterwards.")


def _render_trailing(direction, entry, stop, atr, key=""):
    """The two numbers an exchange trailing stop needs, on the setup itself.

    Skipped quietly if the installed stop_manager.py predates it — one lagging
    file shouldn't stop the whole app from running, when everything else works.
    """
    _plan_fn = getattr(stop_manager, "trailing_plan", None)
    if _plan_fn is None or not TRAIL_ON or None in (direction, entry, stop) or not atr:
        return
    tp = _plan_fn(direction, float(entry), float(stop), float(atr),
                  activate_at_r=TRAIL_AT_R, trail_atr_mult=TRAIL_MULT)
    if tp is None:
        return
    st.markdown("**Trailing stop**")
    t1, t2, t3 = st.columns(3)
    t1.metric("Switch on at", format_price(tp.activate_at), f"{tp.activate_at_r:g}R")
    t2.metric("Trail distance", format_price(tp.distance), f"{tp.distance_pct:.2f}%")
    t3.metric("Locks in", f"{tp.locked_in_at_activation:+.2f}R")
    st.caption(tp.description)
    st.caption("Place the hard stop first — the trail is a change to it later, not a "
               "replacement. On Bybit these go in the trailing-stop fields as an "
               "activation price and a trail distance.")



def _fetch_checked(ticker, timeframe, ref_price=None, min_bars=200, hl_frame=None):
    """Candles for one instrument from the first source that passes.

    Pure fetching, no Streamlit calls, so it is safe to run in a worker thread.
    Returns (frame, source_name, note). Sources that a previous call found
    geo-blocked are skipped instantly by exchanges.py, so the ordering below
    costs one probe per host per session, not per coin.
    """
    gap_limit = 8.0 if timeframe == "1d" else 5.0
    tried = []
    if CRYPTO_SOURCE == "bybit":
        _d, _e = exchanges.fetch_bybit_klines(ticker, timeframe, limit=1000)
        tried.append(("Bybit", _d))
    _d, _e = exchanges.fetch_binance_klines(ticker, timeframe, limit=1000)
    tried.append(("Binance", _d))
    _d, _e = exchanges.fetch_coinbase_candles(ticker, timeframe, limit=300)
    tried.append(("Coinbase", _d))
    _d, _e = exchanges.fetch_kraken_ohlc(ticker, timeframe)
    tried.append(("Kraken", _d))
    if hl_frame is not None:
        tried.append(("Hyperliquid", hl_frame))
    else:
        _d, _e = hyperliquid_data.fetch_candles(
            f"HL:{exchanges.base_asset(ticker)}", timeframe, 1000)
        tried.append(("Hyperliquid", _d))

    checked, rejected = [], []
    for name, frame in tried:
        if frame is not None and not frame.empty and ref_price:
            close = float(frame["Close"].iloc[-1])
            gap = abs(close - ref_price) / ref_price * 100
            if gap > gap_limit:
                rejected.append(f"{name}: last close {close:,.6g} vs {ref_price:,.6g} "
                                f"reference ({gap:.0f}% out)")
                continue
        checked.append((name, frame))

    if not ref_price:
        # No reference to check against, so require two sources to agree.
        closes = [(n, float(f["Close"].iloc[-1]))
                  for n, f in checked if f is not None and not f.empty]
        agreed = None
        for i_, (n1, c1) in enumerate(closes):
            for n2, c2 in closes[i_ + 1:]:
                if c1 > 0 and abs(c1 - c2) / c1 * 100 <= 5:
                    agreed = {n1, n2}
                    break
            if agreed:
                break
        if closes and agreed is None:
            spread = ", ".join(f"{n}={c:,.6g}" for n, c in closes)
            return None, None, (f"No reference price, and the sources disagree ({spread}).")
        if agreed:
            checked = [(n, f) for n, f in checked if n in agreed]

    frame, src, why = frame_check.first_valid(checked, timeframe, min_bars=min_bars)
    return frame, src, "; ".join(rejected + ([why] if why else []))


def _prefetch_candles(tickers, timeframe, refs=None, min_bars=200, hl_frames=None,
                      progress=None, workers=6):
    """Fetch many instruments at once.

    Scanning was a coin at a time: each waiting on several HTTP round trips
    before the next even started. These are independent, so they run together
    and the whole scan takes about as long as its slowest coin rather than the
    sum of all of them. The progress callback runs on the calling thread.
    """
    refs = refs or {}
    hl_frames = hl_frames or {}
    out = {}
    if not tickers:
        return out

    def _one(tk):
        return tk, _fetch_checked(tk, timeframe, refs.get(tk), min_bars,
                                   hl_frames.get(exchanges.base_asset(tk)))

    with ThreadPoolExecutor(max_workers=min(workers, len(tickers))) as pool:
        futures = [pool.submit(_one, tk) for tk in tickers]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                tk, result = future.result()
                out[tk] = result
            except Exception as e:
                tk = None
            if progress:
                try:
                    progress(done, len(tickers), tk or "")
                except Exception:
                    pass
    return out


def _candles_for(ticker, timeframe, bars=400):
    """Candles for any instrument, from whichever source carries it.

    Shared by the chart, the stop manager and the reviews so they can't drift
    apart and show different prices for the same thing.
    """
    if "=" in ticker:                     # commodity or FX: Yahoo
        frames, _p = _yahoo_frames(ticker)
        df = frames.get(timeframe)
        if (df is None or df.empty) and timeframe == "1d":
            hist = yf.Ticker(ticker).history(period="2y", interval="1d")
            df = hist.rename(columns=str.title) if hist is not None else None
        return df
    base = ticker.upper().replace("HL:", "").replace("-USD", "")
    df, _e = hyperliquid_data.fetch_candles(f"HL:{base}", timeframe, bars)
    if df is None:
        df, _e = exchanges.fetch_bybit_klines(ticker, timeframe, limit=bars)
    if df is None:
        df, _e = exchanges.fetch_binance_klines(ticker, timeframe, limit=bars)
    return df


def _review_inputs(ticker, entry_time):
    """Closed candles and live price for reviewing an open trade, from the same
    source the trade came from (Hyperliquid for HL: tickers)."""
    now = pd.Timestamp.now(tz="UTC")
    hours = max(1.0, (now - entry_time).total_seconds() / 3600)
    frames = {}
    if hyperliquid_data.is_hl_ticker(ticker):
        n5 = int(min(5000, hours * 12 + 60))
        for tf, n in (("4h", 30), ("1h", 48), ("5m", n5)):
            df, _e = hyperliquid_data.fetch_candles(ticker, tf, n)
            frames[tf] = trend_retrace.drop_forming(df if df is not None else pd.DataFrame(),
                                                   tf, now)
        _cx, _e = hyperliquid_data.fetch_market_contexts()
        m = next((c for c in _cx if c.ticker == ticker), None)
        price = m.mark_price if m else None
    else:
        for tf, n in (("4h", 30), ("1h", 1000), ("5m", 1000)):
            df = None
            if CRYPTO_SOURCE == "bybit":
                df, _e = exchanges.fetch_bybit_klines(ticker, tf, limit=n)
            if df is None:
                df, _e = exchanges.fetch_binance_klines(ticker, tf, limit=n)
            if df is None:
                df, _e = exchanges.fetch_kraken_ohlc(ticker, tf)
            frames[tf] = trend_retrace.drop_forming(df if df is not None else pd.DataFrame(),
                                                   tf, now)
        price, _s, _e = exchanges.fetch_spot(ticker)
    m5 = frames["5m"]
    since = m5[m5.index >= entry_time] if not m5.empty else m5
    # Older than the 5m window? Fall back to 1H candles for the excursions.
    if not m5.empty and m5.index[0] > entry_time and not frames["1h"].empty:
        since = frames["1h"][frames["1h"].index >= entry_time]
    return frames, since, price, now


def _config_signature(use_tr, params):
    """The settings that backtest evidence depends on. Evidence measured under
    one stop/target/strategy says nothing about another."""
    if use_tr and params is not None:
        return (f"TR|n={params.trend_candles}|stop={params.stop_mode}|"
                f"atr={params.stop_atr_mult:g}|tp={params.target_r:g}")
    return "CONFLUENCE"

APP_BUILD = "2026-09-28-b81 (my strategy runs over your watchlist)"

st.set_page_config(page_title="Bull Run Strategy V2", page_icon="📈", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)

DB_READY = True
DB_ERROR = None
try:
    storage.init_db()
except Exception as _db_exc:  # pragma: no cover - environment dependent
    DB_READY = False
    DB_ERROR = f"{type(_db_exc).__name__}: {_db_exc}"

st.title("📈 Bull Run Strategy V2")
st.caption(f"Build `{APP_BUILD}` · data persisted to SQLite")
with st.expander("What this app does"):
    st.markdown(
        "**Scan** your coins for setups · **Review** whether a setup or an open trade "
        "still holds · **Journal** the trades you take · **Track** how setups actually "
        "resolve · **Coins** explains what each project does.\n\n"
        "**Swing Levels** applies the swing-trading principles Gareth Soloway's approach "
        "shares with mainstream technical trading: daily candles, levels the market has "
        "respected more than once, trend agreement, a confirming candle, a stop beyond "
        "the level and a target at the next one. It is **not** his proprietary system — "
        "those methods are taught in his courses and aren't reproduced here.\n\n"
        "**Trend Retrace** is your own 5-minute strategy: 4H higher highs and lows, a 1H "
        "confirmation candle, then a limit entry on the retrace to 5m support."
    )
if _MISSING_MODULES:
    st.warning(
        "**Some files are missing, so parts of the app are switched off:** "
        + ", ".join(f"`{n}`" for n in _MISSING_MODULES)
        + ". Everything else works. Upload those files to turn the features back on — "
          "`swing.py` is the Swing Levels strategy, `stop_manager.py` the "
          "trailing-stop numbers."
    )
if not DB_READY:
    st.error(
        f"Database could not be initialised, so the Journal and Positions tabs will "
        f"not save anything this session. Everything else still works.\n\n`{DB_ERROR}`"
    )
st.caption(
    "Discretionary trading support tool. Protect capital first — a no-trade decision "
    "is valid. Nothing here executes trades or connects to an exchange. Setup scores "
    "are a checklist quality measure, not a win probability or guarantee."
)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_crypto_universe():
    return fetch_top_cryptos(limit=100)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_exchange_frames(ticker: str):
    """Exchange candles, cached briefly. Exchanges tolerate far more traffic
    than CoinGecko, so this cache is about responsiveness, not rate limits."""
    r = exchanges.build_frames(ticker)
    return r.frames, r.source, r.problems


@st.cache_data(ttl=30, show_spinner=False)
def _cached_exchange_spot(ticker: str):
    return exchanges.fetch_spot(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_cg_frames(coin_id: str):
    """CoinGecko frames, cached for 5 minutes. Its free tier rate-limits hard
    (HTTP 429), and a scan costs three calls, so repeat scans of the same coin
    must not re-hit the API."""
    return coingecko.build_frames_v2(coin_id)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_cg_spot(coin_id: str):
    """CoinGecko spot price, cached for 60 seconds."""
    return coingecko.fetch_spot_price(coin_id)


CRYPTO_TICKERS_ALL, CRYPTO_CG_IDS, CRYPTO_IS_LIVE, CRYPTO_NOTE = _load_crypto_universe()

# CoinGecko ids keyed by SYMBOL as well as label. The watchlist and volume
# scans build their own labels, so a label-only lookup found nothing for them —
# which meant no reference price, and nothing to catch bad candles with.
CRYPTO_CG_BY_SYMBOL = {
    exchanges.base_asset(CRYPTO_TICKERS_ALL[lbl]): cg_id
    for lbl, cg_id in CRYPTO_CG_IDS.items()
    if lbl in CRYPTO_TICKERS_ALL and cg_id
}


@st.cache_data(ttl=86400, show_spinner=False)
def _cg_id_for_symbol(symbol: str):
    """CoinGecko id for a ticker symbol, searching if it isn't in the top 100."""
    sym = (symbol or "").upper()
    if sym in CRYPTO_CG_BY_SYMBOL:
        return CRYPTO_CG_BY_SYMBOL[sym]
    matches, _err = coingecko.search_coins(sym)
    return matches[0]["id"] if matches else None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_venue_listings():
    listings = venues_mod.fetch_listings()
    return listings.bybit, listings.hyperliquid, listings.problems

# Only commodities that exist as Bybit perpetuals, so every setup is tradable.
# The values are the Yahoo symbols used for CANDLES: Bybit's own API is blocked
# from US-hosted servers, and spot gold/silver track the XAU/XAG perps closely.
COMMODITY_TICKERS = {
    "Gold — XAUUSDT": "XAUUSD=X",
    "Silver — XAGUSDT": "XAGUSD=X",
    "Crude Oil — CLUSDT": "CL=F",
}

# What you'd actually place the order on, for each of the above.
COMMODITY_PERPS = {
    "XAUUSD=X": "XAUUSDT (Bybit, up to 75x)",
    "XAGUSD=X": "XAGUSDT (Bybit, up to 75x)",
    "CL=F": "CLUSDT (Bybit, up to 50x)",
}

FX_TICKERS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X", "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "AUD/JPY": "AUDJPY=X", "GBP/JPY": "GBPJPY=X", "USD/SGD": "SGD=X",
    "Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
}

# ---------------------------------------------------------------------
# Sidebar settings (shared across tabs)
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Strategy")
strategy_choice = st.sidebar.radio(
    "Scanner strategy",
    (["Mean Reversion (mine — high win rate)"] if mean_reversion else [])
    + (["Swing Levels (daily — Soloway-style)"] if swing else [])
    + ["Trend Retrace (your 5-minute strategy)"],
    key="strategy_choice",
    help="Trend Retrace: two 4H candles of HH/HL, a bullish 1H candle, then a 5m "
         "retrace to support (reverse for shorts). Confluence is the earlier 10-point "
         "checklist, kept for comparison.")
USE_TR = strategy_choice.startswith("Trend")
USE_SWING = strategy_choice.startswith("Swing") and swing is not None
USE_MR = strategy_choice.startswith("Mean") and mean_reversion is not None
if USE_MR:
    with st.sidebar.expander("Mean Reversion settings", expanded=False):
        st.caption("Small target, wide stop: it wins often by design. At a 0.4:1 target "
                   "a market with no edge still wins about 71% of the time, so the win "
                   "rate alone tells you nothing — one loss undoes two and a half wins.")
        _mr_target = st.number_input("Target (multiple of risk)", 0.1, 1.0, 0.4, 0.05,
                                     key="mr_target")
        st.caption(f"Break-even win rate at {_mr_target:g}:1 is "
                   f"**{1 / (1 + _mr_target):.0%}**.")
        _mr_stop = st.slider("Stop beyond the level (ATR)", 0.5, 4.0, 1.5, 0.25,
                             key="mr_stop")
        _mr_ext = st.slider("Minimum stretch from the average (ATR)", 0.5, 4.0, 1.5, 0.25,
                            key="mr_ext")
        _mr_reject = st.checkbox("Require a rejection candle", value=True, key="mr_reject",
                                 help="The trigger. Switching it off gives far more "
                                      "setups, most of them worse.")
    MR_PARAMS = mean_reversion.MeanReversionParams(
        target_r=_mr_target, stop_atr=_mr_stop, min_extension_atr=_mr_ext,
        require_rejection=_mr_reject)
else:
    MR_PARAMS = None
if USE_SWING:
    with st.sidebar.expander("Swing settings", expanded=False):
        st.caption("Daily candles: check once a day, hold for days or weeks. Built from "
                   "standard swing-trading principles — major levels, trend agreement and "
                   "a confirming candle — not anyone's proprietary system.")
        _sw_touches = st.slider("Minimum touches for a level", 2, 5, 2, key="sw_touches",
                                help="How many times price must have respected a level "
                                     "before it counts. Higher means far fewer setups.")
        _sw_rr = st.number_input("Minimum reward:risk", 1.0, 10.0, 3.0, 0.5, key="sw_rr")
        _sw_stop = st.slider("Stop beyond the level (daily ATR)", 0.25, 3.0, 0.75, 0.25,
                             key="sw_stop")
        _sw_confirm = st.checkbox("Require a confirming daily candle", value=True,
                                  key="sw_confirm",
                                  help="A hammer or engulfing candle at the level. Without "
                                       "it the level alone is the trigger.")
    SWING_PARAMS = swing.SwingParams(min_touches=int(_sw_touches), min_rr=_sw_rr,
                                      stop_atr=_sw_stop, require_pattern=_sw_confirm)
else:
    SWING_PARAMS = None

if USE_TR:
    with st.sidebar.expander("Stop & take profit — please confirm", expanded=False):
        st.caption("Your rules don't specify these, so these are adjustable defaults.")
        _stop_choice = st.radio("Stop loss", ["Below the 5m support (ATR buffer)",
                                              "Below the last closed 4H candle"],
                                key="tr_stop_mode")
        _stop_atr = st.number_input(
            "Buffer below support (x 5m ATR)", min_value=0.5, max_value=10.0, value=3.0,
            step=0.5, key="tr_stop_atr",
            disabled=_stop_choice.startswith("Below the last"),
            help="At 1x, trades finished in ~12 minutes — before any 1H candle could close, "
                 "so your 'add 50% after a 1H candle' rule could never trigger. 3x lets "
                 "trades run for hours.")
        _target_r = st.number_input(
            "Take profit (multiple of risk)", min_value=0.25, max_value=10.0, value=3.0,
            step=0.25, key="tr_target_r",
            help="Sets the trade-off between how OFTEN you win and how MUCH. A big target "
                 "wins rarely and large; a small one wins often and small.")
        _be = expectancy.break_even_win_rate(_target_r)
        st.caption(f"At **{_target_r:g}:1** you break even winning **{_be:.0%}** of trades "
                   f"(about {_be * 10:.0f} in 10), and profit above that.")
        _trend_n = st.number_input("4H candles required", min_value=1, max_value=5,
                                    value=2, step=1, key="tr_trend_n")
    with st.sidebar.expander("Setup quality filters", expanded=False):
        st.caption("**Off by default** — these are extra gates beyond your three rules, not "
                   "part of them. Each roughly halves the number of setups, so turn one on "
                   "only after the backtest shows it helps.")
        _q_body = st.slider("1H candle body, minimum share of its range", 0.0, 0.8, 0.0,
                            0.05, key="q_body",
                            help="A doji closes level — it confirms nothing.")
        _q_trend = st.slider("4H trend move, minimum ATR", 0.0, 3.0, 0.0, 0.1,
                             key="q_trend",
                             help="Marginally higher highs happen constantly in a range.")
        _q_ext = st.slider("Max distance from the 4H average (ATR)", 0.0, 8.0, 0.0, 0.5,
                           key="q_ext",
                           help="Stops you entering after price has already run. 0 = off.")
    TR_PARAMS = trend_retrace.StrategyParams(
        trend_candles=int(_trend_n),
        stop_mode=(trend_retrace.STOP_BELOW_4H_CANDLE if _stop_choice.startswith("Below the last")
                   else trend_retrace.STOP_BELOW_SUPPORT),
        stop_atr_mult=_stop_atr, target_r=_target_r,
        min_h1_body_ratio=_q_body, min_trend_move_atr=_q_trend, max_extension_atr=_q_ext)
else:
    TR_PARAMS = None

st.sidebar.subheader("Your account")
ACCOUNT_EQUITY = st.sidebar.number_input("Account equity ($)", min_value=0.0, value=10000.0,
                                          step=100.0, key="acct_equity")
RISK_PCT = st.sidebar.number_input("Risk per trade (%)", min_value=0.1, max_value=100.0,
                                    value=1.0, step=0.1, key="acct_risk",
                                    help="The most you'll lose if the stop is hit. Used "
                                         "everywhere — plans, positions and the calculator.")
LEVERAGE = st.sidebar.number_input("Leverage", min_value=1.0, max_value=50.0, value=1.0,
                                    step=0.5, key="acct_lev")
with st.sidebar.expander("Costs"):
    FEE_RATE = st.number_input("Fee rate (round trip)", min_value=0.0, value=0.0006,
                               step=0.0001, format="%.4f", key="acct_fee")
    SLIPPAGE = st.number_input("Slippage", min_value=0.0, value=0.0005, step=0.0001,
                               format="%.4f", key="acct_slip")
st.sidebar.caption(f"Risking **${ACCOUNT_EQUITY * RISK_PCT / 100:,.2f}** per trade.")

MAX_ENTRY_GAP = st.sidebar.slider(
    "Hide entries further than this from the live price (%)", 1.0, 30.0, 5.0, 0.5,
    key="max_entry_gap",
    help="A limit entry a long way from the current price is either a level you'd wait "
         "days for, or a data problem. Either way it isn't actionable now.")

st.sidebar.markdown("---")
with st.sidebar.expander("Trailing stop"):
    st.caption("Shown on every setup as two numbers you can type into Bybit. Whether "
               "trailing helps at all is measured in the 🔁 Backtest — leave it off if it "
               "doesn't.")
    TRAIL_ON = st.checkbox("Include a trailing stop in setups", value=True,
                           key="trail_on")
    TRAIL_AT_R = st.number_input("Switch it on at (R in profit)", min_value=0.25,
                                 max_value=5.0, value=1.0, step=0.25, key="trail_at_r",
                                 disabled=not TRAIL_ON)
    TRAIL_MULT = st.number_input("Trail distance (x ATR)", min_value=0.5, max_value=6.0,
                                 value=2.0, step=0.5, key="trail_mult",
                                 disabled=not TRAIL_ON)

with st.sidebar.expander("Advanced settings"):
    st.caption("Data-freshness rules and display timezone. The defaults rarely need "
               "changing — crypto prices come live from the exchange regardless.")
    live_threshold = st.number_input("LIVE threshold (seconds)", value=60, min_value=5)
    stale_threshold = st.number_input("STALE threshold (seconds)", value=900, min_value=60)
    tz_name = st.text_input("Timezone (IANA)", value="Australia/Sydney")
    try:
        local_tz = zoneinfo.ZoneInfo(tz_name)
        st.caption(f"Local now: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M %Z')}")
    except Exception:
        st.error(f"'{tz_name}' is not a valid IANA timezone — using UTC.")
        local_tz = timezone.utc

# A free CoinGecko demo key raises the rate limit a lot. Optional.
_cg_key = ""
try:
    _cg_key = st.secrets.get("COINGECKO_API_KEY", "")
except Exception:
    _cg_key = ""
if _cg_key:
    coingecko.set_api_key(_cg_key)

_cmc_key = ""
try:
    _cmc_key = st.secrets.get("COINMARKETCAP_API_KEY", "")
except Exception:
    _cmc_key = ""
if _cmc_key:
    sentiment.set_cmc_api_key(_cmc_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _coingecko_matches(symbol: str):
    return coingecko.search_coins(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def _exchange_has(symbol: str):
    """Is this symbol tradable on Binance/Kraken? A two-candle request is the
    cheapest reliable check. Used for watchlist names Hyperliquid doesn't list."""
    ticker = f"{symbol.upper()}-USD"
    df, _e = exchanges.fetch_binance_klines(ticker, "4h", limit=2)
    if df is not None and not df.empty:
        return ticker, "Binance"
    df, _e = exchanges.fetch_kraken_ohlc(ticker, "4h")
    if df is not None and not df.empty:
        return ticker, "Kraken"
    return None, None


@st.cache_data(ttl=120, show_spinner=False)
def _cached_cg_bulk(coin_ids):
    return coingecko.fetch_prices_bulk(list(coin_ids))


@st.cache_data(ttl=120, show_spinner=False)
def _cached_bybit_tickers():
    return exchanges.fetch_bybit_tickers()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_hl_contexts():
    ctxs, err = hyperliquid_data.fetch_market_contexts()
    return ([{"name": c.name, "ticker": c.ticker, "label": c.label,
              "volume": c.day_volume_usd, "mark": c.mark_price,
              "oi": c.open_interest_usd, "funding": c.funding_rate,
              "change": c.change_24h_pct} for c in ctxs], err)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_period_changes():
    return coingecko.fetch_period_changes("90d", limit=100)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_fear_greed():
    fg, err = sentiment.fetch_fear_greed()
    if fg is None:
        return None, err
    return {"value": fg.value, "classification": fg.classification, "source": fg.source,
            "previous": fg.previous, "direction": fg.direction}, err

_fg, _fg_err = _cached_fear_greed()
if _fg:
    st.sidebar.metric("Fear & Greed", f"{_fg['value']} · {_fg['classification']}",
                      (f"{_fg['direction']} from {_fg['previous']}"
                       if _fg.get("previous") is not None else None), delta_color="off")
    st.sidebar.caption(f"Source: {_fg['source']}. A whole-market mood gauge — it says "
                       f"nothing about any single setup.")
elif _fg_err:
    st.sidebar.caption(f"Fear & Greed unavailable: {_fg_err}")

st.sidebar.markdown("---")
st.sidebar.subheader("Tradable universe")
venue_mode = st.sidebar.radio(
    "Restrict coins to",
    ["Bybit only", "Hyperliquid only", "Bybit or Hyperliquid",
     "Bybit AND Hyperliquid", "All top-100"],
    key="venue_mode",
    help="Market-cap rank includes exchange tokens (WBT, OKB) and wrapped assets you "
         "cannot trade as perpetuals. Filtering to your venue keeps the scan actionable. "
         "Bybit blocks requests from the US server this app runs on, so its list may be "
         "unavailable.")
_VENUE_MODES = {"Bybit only": venues_mod.MODE_BYBIT,
                "Hyperliquid only": venues_mod.MODE_HYPERLIQUID,
                "Bybit or Hyperliquid": venues_mod.MODE_EITHER,
                "Bybit AND Hyperliquid": venues_mod.MODE_BOTH}

if venue_mode == "All top-100":
    CRYPTO_TICKERS = dict(CRYPTO_TICKERS_ALL)
    VENUE_TAGS = {}
    st.sidebar.caption(f"{len(CRYPTO_TICKERS)} coins (unfiltered).")
else:
    _by, _hl, _vprob = _load_venue_listings()
    _listings = venues_mod.VenueListings(bybit=_by, hyperliquid=_hl, problems=_vprob)
    CRYPTO_TICKERS, VENUE_TAGS, _vnotes = venues_mod.filter_universe(
        CRYPTO_TICKERS_ALL, _listings, mode=_VENUE_MODES[venue_mode])
    if venue_mode == "Hyperliquid only":
        st.sidebar.caption(f"Hyperliquid: {len(_hl)} perps")
    else:
        st.sidebar.caption(f"Bybit: {len(_by) or 'unavailable'} · "
                           f"Hyperliquid: {len(_hl) or 'unavailable'} perps")
    for _n in _vnotes:
        if "NOT applied" in _n or "missing from the list" in _n or "instead of an empty" in _n:
            st.sidebar.warning(_n)
        else:
            st.sidebar.caption(_n)

st.sidebar.markdown("---")
st.sidebar.subheader("Data source")
crypto_source = st.sidebar.radio(
    "Crypto prices from",
    ["Bybit", "Exchange (Binance/Kraken)", "CoinGecko", "Yahoo Finance"],
    key="crypto_source",
    help=("Exchange uses Binance, falling back to Kraken. It gives true OHLCV at every "
          "interval with no meaningful rate limit — the best option. CoinGecko covers "
          "more obscure coins but rate-limits hard and has no true daily OHLC. Yahoo "
          "misses newer listings. Sources fall back to each other automatically."),
)
CRYPTO_SOURCE = ("bybit" if crypto_source == "Bybit"
                 else "exchange" if crypto_source.startswith("Exchange")
                 else "coingecko" if crypto_source.startswith("CoinGecko")
                 else "yahoo")
CRYPTO_PREFERS_CG = CRYPTO_SOURCE == "coingecko"
st.sidebar.caption(
    "Stocks and commodities always use Yahoo Finance — CoinGecko has no equities "
    "or futures data."
)
BYBIT_TICKERS = {}
if CRYPTO_SOURCE == "bybit":
    BYBIT_TICKERS, _bybit_err = _cached_bybit_tickers()
    if BYBIT_TICKERS:
        st.sidebar.caption(f"Bybit: {len(BYBIT_TICKERS)} USDT perps.")
    else:
        st.sidebar.warning(
            "**Bybit can't be reached from this server** — it blocks US traffic and "
            "Streamlit is US-hosted. Binance is blocked too. Candles are coming from "
            "Hyperliquid instead; prices on majors are near-identical, so scans stay "
            "accurate and you place the orders on Bybit. Running this app on your own "
            "computer in Australia would use Bybit directly — see RUNNING_LOCALLY.md in "
            "the download.")

st.sidebar.markdown("---")
st.sidebar.caption(f"curl_cffi installed: **{curl_cffi_is_available()}**")
st.sidebar.caption(f"Scannable crypto: {len(CRYPTO_TICKERS)} symbols")
if coingecko.has_api_key():
    st.sidebar.caption("CoinGecko API key: **set** (higher rate limit)")
else:
    st.sidebar.caption(
        "CoinGecko API key: not set — free anonymous limits apply, so rapid "
        "scanning will hit 429s. A free demo key in Streamlit secrets "
        "(`COINGECKO_API_KEY`) raises this substantially.")
if not CRYPTO_IS_LIVE:
    st.sidebar.warning(CRYPTO_NOTE)
st.sidebar.warning(
    "Streamlit Cloud wipes the database whenever the app redeploys — including every time "
    "you upload new files. Export your journal and tracking CSVs before updating, then "
    "import them back afterwards."
)

(tab_scan, tab_review, tab_journal, tab_track, tab_learn) = st.tabs(
    ["🎯 Scan", "🔍 Review", "📓 Journal", "📈 Tracking", "📚 Coins"]
)

# ---------------------------------------------------------------------
# TAB: Market & data health
# ---------------------------------------------------------------------
with tab_learn:
    st.subheader("What each coin actually does")
    st.caption(
        "One line per coin, in plain English. These are written from general knowledge and "
        "don't update themselves — projects pivot and change direction, so look anything up "
        "properly before putting money into it. Nothing here is a view on whether a coin is "
        "worth owning, only what it's for."
    )

    _g1, _g2 = st.columns([2, 1])
    _q = _g1.text_input("Search", key="gloss_q", placeholder="e.g. sol, oracle, privacy")
    _cats = ["All"] + sorted(coin_info.CATEGORIES)
    _cat = _g2.selectbox("Category", _cats, key="gloss_cat")

    _order = [t.replace("-USD", "") for t in CRYPTO_TICKERS_ALL.values()]
    _ranked = [s for s in _order if coin_info.describe(s)]
    _rest = sorted(s for s in coin_info.COINS if s not in _ranked)
    _symbols = _ranked + _rest

    def _matches(sym):
        cat, text = coin_info.describe(sym)
        if _cat != "All" and cat != _cat:
            return False
        if not _q:
            return True
        needle = _q.strip().lower()
        return needle in sym.lower() or needle in text.lower() or needle in cat.lower()

    _shown = [s for s in _symbols if _matches(s)]
    _covered, _total = coin_info.coverage(_order)
    st.caption(f"{len(_shown)} shown · {len(coin_info.COINS)} coins described, covering "
               f"{_covered} of the top {_total} by market cap.")

    _colours = {"Layer 1": theme.GREEN, "Layer 2": theme.BLUE, "DeFi": theme.PURPLE,
                "Meme": theme.AMBER, "Exchange": theme.BLUE, "AI": theme.PURPLE,
                "Infrastructure": theme.GREY, "Privacy": theme.GREY,
                "Payments": theme.GREEN, "Gaming": theme.AMBER,
                "Stablecoin": theme.GREY, "RWA": theme.BLUE}
    for sym in _shown:
        cat, text = coin_info.describe(sym)
        rank = _order.index(sym) + 1 if sym in _order else None
        with st.container(border=True):
            st.markdown(
                f"**{sym}**" + (f"  ·  #{rank} by market cap" if rank else "")
                + "  " + theme.pill(cat, _colours.get(cat, theme.GREY)),
                unsafe_allow_html=True)
            st.write(text)

    if not _shown:
        st.info("Nothing matches. Try a different word, or look the coin up below.")

    st.markdown("##### Look up any other coin")
    st.caption("Fetches the project's own description from CoinGecko — one call, so it's "
               "done on request rather than for a whole list.")
    _lu = st.text_input("Ticker", key="gloss_lookup", placeholder="e.g. CARDS")
    if st.button("Look it up", key="gloss_go") and _lu.strip():
        with st.spinner("Searching…"):
            _cands, _err = _coingecko_matches(_lu.strip().upper())
        if not _cands:
            st.warning(_err or "No coin found with that ticker.")
        else:
            for c in _cands[:3]:
                with st.container(border=True):
                    st.markdown(f"**{c['name']}** ({c['symbol']})"
                                + (f"  ·  #{c['rank']} by market cap" if c["rank"] else ""))
                    _txt, _home, _derr = coingecko.fetch_description(c["id"])
                    if _txt:
                        st.write(_txt[:700] + ("…" if len(_txt) > 700 else ""))
                        if _home:
                            st.caption(f"Project site: {_home}")
                        st.caption("This is the project's own description of itself, so it "
                                   "reads as marketing. Useful for what it does, not for "
                                   "whether it's any good.")
                    else:
                        st.caption(_derr or "No description available.")

# ---------------------------------------------------------------------
# TAB: Scanner
# ---------------------------------------------------------------------
with tab_scan:
    st.subheader("Candidate scanner")
    st.caption(
        "Derives the checklist from real 1D / 4H / 1H structure. Every item is "
        "overridable — ❔ means it could not be evaluated and scores zero rather than guessing."
    )

    _modes = (["Rank the universe", "Auto-scan one instrument"]
              if (USE_TR or USE_SWING or USE_MR)
              else ["Rank the universe", "Auto-scan one instrument", "Manual checklist"])
    mode = st.radio("Mode", _modes, key="scan_mode",
                    help="Rank the universe checks many coins and shortlists the best; "
                         "single-instrument mode shows every rule for one coin in detail.")

    score_evidence, seq_evidence = {}, {}

    if mode == "Rank the universe":
        market = st.radio("Market", ["Crypto", "Commodities", "FX"],
                          horizontal=True, key="uni_market")
        NON_CRYPTO = market != "Crypto"
        if NON_CRYPTO:
            _lists = {"Commodities": COMMODITY_TICKERS, "FX": FX_TICKERS}
            _list = _lists[market]
            st.caption(
                f"Your three rules applied to {market.lower()}, using Yahoo Finance data. "
                f"Yahoo has no 4H interval, so 4H candles are built from 1H bars.")
            if market == "FX":
                st.caption(
                    "FX trades 24 hours, five days a week, so it fits the strategy better "
                    "than stocks — but weekend gaps can still open past a stop, and Yahoo's "
                    "FX prices are indicative rather than a broker's dealable quotes.")
            else:
                st.caption(
                    "Only commodities Bybit lists as perpetuals — gold, silver and crude — "
                    "so every setup is one you can actually place. They trade 24/7 with "
                    "funding every four hours, which suits the strategy better than stocks.")
                st.caption(
                    "Candles come from Yahoo (spot gold and silver, and the crude futures "
                    "contract) because Bybit's API is blocked from this server. Those track "
                    "the perps closely but aren't identical — check the price on Bybit "
                    "before placing an order. Crude futures also pause briefly each day and "
                    "close at weekends, so crude may read STALE while CLUSDT is still "
                    "trading.")
            st.caption("None of this has been backtested — the backtest only covers crypto.")

        coin_source = st.radio(
            "Which coins",
            ["Top by market cap", "High volume, outside the top 20 (Hyperliquid)",
             "My watchlist"],
            key="uni_coin_source",
            help="The second option ranks Hyperliquid perps by their 24-hour volume on "
                 "Hyperliquid — the liquidity you'd actually trade into — and skips the "
                 "largest coins by market cap.")
        WATCHLIST_MODE = (coin_source == "My watchlist") and not NON_CRYPTO
        HL_MODE = ((coin_source.startswith("High volume") or WATCHLIST_MODE)
                   and not NON_CRYPTO)
        if WATCHLIST_MODE:
            _wl_which = st.radio("Watchlist", ["Main", "Memes", "Both"], horizontal=True,
                                 key="wl_which",
                                 help="Memes are kept separate because they move on "
                                      "attention rather than anything measurable — worth "
                                      "judging on their own once you have enough trades.")
            _wl_saved_main = storage.load_value("wl_text_main")[0]
            _wl_saved_meme = storage.load_value("wl_text_memes")[0]
            if _wl_which in ("Main", "Both"):
                _main_text = st.text_area(
                    "Main list", value=_wl_saved_main or watchlist_mod.DEFAULT_WATCHLIST,
                    height=90, key="wl_text_main",
                    help="Separate with commas. Full names work too (stacks, immutable).")
            else:
                _main_text = ""
            if _wl_which in ("Memes", "Both"):
                _meme_text = st.text_area(
                    "Meme list", value=_wl_saved_meme or watchlist_mod.MEME_WATCHLIST,
                    height=70, key="wl_text_memes")
            else:
                _meme_text = ""
            if st.button("Save lists", key="wl_save"):
                if _wl_which in ("Main", "Both"):
                    storage.save_value("wl_text_main", _main_text)
                if _wl_which in ("Memes", "Both"):
                    storage.save_value("wl_text_memes", _meme_text)
                st.success("Saved. These survive until the database is wiped.")
            wl_text = ", ".join(t for t in (_main_text, _meme_text) if t.strip())
            _wl_markets = []
            _wl_ctxs, _wl_err = _cached_hl_contexts()
            if _wl_ctxs:
                _wl_markets = [c["name"] for c in _wl_ctxs]
            _alias_text, _ = storage.load_value("wl_aliases")
            _alias_text = _alias_text or ""
            _aliases, _bad_alias = watchlist_mod.parse_aliases(_alias_text)
            _wl_found, _wl_missing = watchlist_mod.resolve(wl_text, _wl_markets,
                                                            aliases=_aliases)
            _elsewhere = []      # set below; defined here so a failed market
            _via_cg = []         # list can't crash the rest of the page
            if _wl_markets:
                st.caption(f"Matched {len(_wl_found)} of "
                           f"{len(watchlist_mod.parse_list(wl_text))} entries on Hyperliquid: "
                           + ", ".join(sorted({r.market for r in _wl_found})))
                # Anything Hyperliquid doesn't list may still trade elsewhere.
                _elsewhere = []
                _still_missing = []
                for u in _wl_missing:
                    _sym = watchlist_mod._normalise(_aliases.get(
                        watchlist_mod._normalise(u), u))
                    _tk, _venue = _exchange_has(_sym)
                    if _tk:
                        _elsewhere.append((u, _sym, _tk, _venue))
                    else:
                        _still_missing.append(u)
                # Last resort: CoinGecko covers coins no exchange here lists.
                _via_cg = []
                _truly_missing = []
                _cg_choice = storage.load_value("wl_cg_ids")[0] or {}
                for u in _still_missing:
                    _sym = watchlist_mod._normalise(_aliases.get(
                        watchlist_mod._normalise(u), u))
                    _cands, _cg_err = _coingecko_matches(_sym)
                    if not _cands:
                        _truly_missing.append((u, _cg_err))
                        continue
                    _chosen = _cg_choice.get(_sym)
                    if len(_cands) == 1:
                        _chosen = _chosen or _cands[0]["id"]
                    if _chosen:
                        _pick = next((c for c in _cands if c["id"] == _chosen), _cands[0])
                        _via_cg.append((u, _sym, _pick))
                    else:
                        # Several projects share this ticker — the trader picks.
                        st.warning(f"**{u}** matches {len(_cands)} different projects with "
                                   f"the ticker {_sym}. Choose which one you mean:")
                        _opt = st.selectbox(
                            f"Which {_sym}?",
                            [f"{c['name']} ({c['id']})"
                             + (f" · rank {c['rank']}" if c["rank"] else " · unranked")
                             for c in _cands],
                            key=f"cgpick_{_sym}")
                        if st.button(f"Use this for {_sym}", key=f"cgset_{_sym}"):
                            _cg_choice[_sym] = _cands[
                                [f"{c['name']} ({c['id']})"
                                 + (f" · rank {c['rank']}" if c["rank"] else " · unranked")
                                 for c in _cands].index(_opt)]["id"]
                            storage.save_value("wl_cg_ids", _cg_choice)
                            st.rerun()

                if _via_cg:
                    st.info("Not on any exchange here, so CoinGecko's candles will be used: "
                            + "; ".join(f"**{u}** → {p['name']}" for u, _s, p in _via_cg)
                            + ". CoinGecko aggregates across exchanges and has no true daily "
                              "OHLC, so these scans are coarser — and you still need a venue "
                              "that actually lists the coin to trade it.")
                if _truly_missing:
                    _hints = []
                    for u, why in _truly_missing:
                        s_ = watchlist_mod.suggest(u, _wl_markets)
                        _hints.append(f"**{u}**" + (f" → did you mean {', '.join(s_)}?"
                                                    if s_ else f" ({why})"))
                    st.warning("Couldn't find anywhere: " + "; ".join(_hints)
                               + ". Use the tools below to find the right name, or map it "
                                 "yourself — nothing is guessed for you, because scanning "
                                 "the wrong coin looks exactly like scanning the right one.")

                with st.expander(f"🔎 Browse all {len(_wl_markets)} Hyperliquid markets"):
                    _q = st.text_input("Search", key="wl_search",
                                       placeholder="e.g. light, card, drv")
                    _hits = watchlist_mod.search_markets(_q, _wl_markets, limit=60)
                    if _hits:
                        st.write(" · ".join(f"`{m}`" for m in _hits))
                        st.caption("Copy the exact name into your list above. If a coin isn't "
                                   "here, Hyperliquid doesn't list it as a perpetual and it "
                                   "can't be scanned or traded there.")
                    else:
                        st.caption("No market contains that text.")

                with st.expander("🔗 My name mappings"):
                    st.caption("One per line, e.g. `lighter = LIT`. Use this when you know "
                               "which market a name refers to. Check the project on "
                               "Hyperliquid first — tickers are reused across projects, and "
                               "mapping to the wrong one would silently scan the wrong coin.")
                    _new_alias = st.text_area("Mappings", value=_alias_text, height=90,
                                              key="wl_alias_text")
                    if st.button("Save mappings", key="wl_alias_save"):
                        _parsed, _badlines = watchlist_mod.parse_aliases(_new_alias)
                        _unknown = [f"{k} → {v}" for k, v in _parsed.items()
                                    if v not in {watchlist_mod._normalise(m)
                                                 for m in _wl_markets}]
                        storage.save_value("wl_aliases", _new_alias)
                        if _badlines:
                            st.warning("Couldn't read: " + "; ".join(_badlines))
                        if _unknown:
                            st.warning("Saved, but these point at markets Hyperliquid doesn't "
                                       "list: " + "; ".join(_unknown))
                        st.rerun()
                    if _bad_alias:
                        st.warning("Existing lines that couldn't be read: "
                                   + "; ".join(_bad_alias))
            else:
                st.error(f"Couldn't load Hyperliquid's market list: {_wl_err}")
        elif HL_MODE:
            v1, v2, v3 = st.columns(3)
            hl_exclude_n = v1.number_input("Skip the top N by market cap", min_value=0,
                                           max_value=100, value=20, step=5, key="hl_excl")
            hl_min_vol = v2.number_input("Min 24h volume ($M)", min_value=0.0, value=20.0,
                                         step=5.0, key="hl_minvol",
                                         help="Volume on Hyperliquid only. Thinner markets "
                                              "mean wider spreads and more slippage.")
            hl_count = v3.selectbox("How many coins", [10, 20, 30, 50, 100], index=1,
                                    key="hl_count",
                                    format_func=lambda n: f"Top {n} by volume")
            st.caption(
                "⚠️ High volume outside the top coins often means a sharp move, news or a "
                "pump. These markets are usually more volatile, and your strategy hasn't "
                "been backtested on them — treat results with extra caution.")
        if USE_MR:
            st.caption(
                "**My strategy.** It looks for a market stretched away from its average "
                "that has arrived at a price it respected before, and takes a small "
                "profit on the snap back. Daily candles, so one look a day."
            )
            st.caption(
                f"It wins often on purpose — and that is not the same as making money. "
                f"At {MR_PARAMS.target_r if MR_PARAMS else 0.4:g}:1 a market with no edge "
                f"still wins about "
                f"{1 / (1 + (MR_PARAMS.target_r if MR_PARAMS else 0.4)):.0%} of the time, "
                f"so it only pays above that. One loss undoes "
                f"{1 / (MR_PARAMS.target_r if MR_PARAMS else 0.4):.1f} wins, which is why "
                f"the stop matters more here, not less."
            )
            # Respect whichever coin source is selected — watchlist, high
            # volume or top by market cap — rather than forcing the top list.
            if NON_CRYPTO:
                _lo, _hi, _default_n = uihelpers.scan_count(len(_list))
                universe_size = (_default_n if _lo is None else
                                 st.slider(f"How many {market.lower()} to scan",
                                           _lo, _hi, _default_n, key="uni_noncrypto_n"))
            elif WATCHLIST_MODE:
                universe_size = len(_wl_found) + len(_elsewhere) + len(_via_cg)
                st.caption(f"Running it over your **{_wl_which.lower()}** watchlist — "
                           f"{universe_size} instruments.")
            elif HL_MODE:
                universe_size = hl_count
            else:
                universe_size = st.selectbox(
                    "Scan the top…", [10, 20, 30, 50, 100], index=1, key="uni_size",
                    format_func=lambda n: f"Top {n} by market cap")
            uni_direction = "Auto"
            uni_min_rr = MR_PARAMS.target_r if MR_PARAMS else 0.4
        elif USE_SWING:
            st.caption(
                "Applies the swing rules to each instrument's **daily** chart: a clear "
                "daily trend, price at a level the market has respected before, and a "
                "confirming daily candle. Most instruments will show nothing on most "
                "days — that is the point of a higher-timeframe strategy."
            )
            universe_size = st.selectbox(
                "Scan the top…", [10, 20, 30, 50, 100], index=1, key="uni_size",
                format_func=lambda n: f"Top {n} by market cap")
            uni_direction, uni_min_rr = "Auto", SWING_PARAMS.min_rr if SWING_PARAMS else 3.0
        elif USE_TR:
            st.caption(
                "Applies your three rules to each coin: **two 4H candles of higher highs and "
                "higher lows**, a **bullish 1H candle**, and a **5m retrace to previous "
                "support** (reversed for shorts). Only closed candles count. A setup meeting "
                "all three rules scores 10/10; the score is a checklist count, not a "
                "probability of profit."
            )
            if NON_CRYPTO:
                _lo, _hi, _default_n = uihelpers.scan_count(len(_list))
                if _lo is None:
                    universe_size = _default_n
                    st.caption(f"Scanning all {universe_size} — there are only a few.")
                else:
                    universe_size = st.slider(f"How many {market.lower()} to scan",
                                               _lo, _hi, _default_n, key="uni_noncrypto_n")
            elif WATCHLIST_MODE:
                universe_size = len(_wl_found) + len(_elsewhere) + len(_via_cg)
            elif HL_MODE:
                universe_size = hl_count
            else:
                universe_size = st.selectbox(
                    "Scan the top…", [10, 20, 30, 50, 100], index=1, key="uni_size",
                    format_func=lambda n: f"Top {n} by market cap",
                    help="More coins means more API calls and a longer wait.")
            uni_direction, uni_min_rr = "Auto", 3.0
        else:
            st.caption(
                "Scores many instruments on the original 10-point confluence checklist "
                "using daily, 4H and 1H candles. Treat results as a shortlist, then run a "
                "full single-instrument scan on anything promising."
            )
            u1, u2, u3 = st.columns(3)
            if NON_CRYPTO:
                _lo, _hi, _default_n = uihelpers.scan_count(len(_list))
                if _lo is None:
                    universe_size = _default_n
                    st.caption(f"Scanning all {universe_size} — there are only a few.")
                else:
                    universe_size = st.slider(f"How many {market.lower()} to scan",
                                               _lo, _hi, _default_n, key="uni_noncrypto_n")
            elif WATCHLIST_MODE:
                universe_size = len(_wl_found) + len(_elsewhere) + len(_via_cg)
            elif HL_MODE:
                universe_size = hl_count
            else:
                universe_size = u1.selectbox(
                    "Scan the top…", [10, 20, 30, 50, 100], index=1, key="uni_size",
                    format_func=lambda n: f"Top {n} by market cap",
                    help="More coins means more API calls and a longer wait.")
            uni_direction = u2.selectbox("Direction", ["Auto", "Long", "Short"], key="uni_dir")
            uni_min_rr = u3.number_input("Min R:R", min_value=3.0, value=3.0,
                                          step=0.5, key="uni_rr")

        if NON_CRYPTO:
            st.caption(f"Roughly {universe_size * 3:.0f}s for {universe_size} instruments — "
                       f"Yahoo is slower than an exchange API.")
        elif HL_MODE:
            st.caption(f"Usually under {max(8, universe_size * 0.4):.0f}s for "
                       f"{universe_size} coins — one request each, sent in parallel within "
                       f"Hyperliquid's per-minute budget, with 4H and 1H built from the 5m "
                       f"candles.")
            if WATCHLIST_MODE and not _wl_found:
                st.info("Nothing to scan yet — fix the unmatched entries above.")
        else:
            per_coin = 0.6 if CRYPTO_SOURCE == "exchange" else 1.4   # 4 calls/coin
            est = universe_size * per_coin
            st.caption(
                f"Roughly {est:.0f}s for {universe_size} coins using "
                f"{'exchange data (fast — no meaningful rate limit)' if CRYPTO_SOURCE == 'exchange' else 'CoinGecko (calls spaced to avoid 429s)'}."
            )

        st.checkbox("Save these setups so I can re-check them later",
                    value=True, key="track_signals",
                    help="Every setup with a complete plan is saved with its levels frozen. "
                         "Review them any time on the 📈 Tracking tab to see whether they're "
                         "still worth leaving pending.")

        if st.button("🔎 Scan universe", use_container_width=True):
            hl_marks = {}
            if NON_CRYPTO:
                _names = list(_list)[:universe_size]
                instruments = [(n, _list[n], (_list[n], None)) for n in _names]
                st.session_state.hl_info = {}
            elif WATCHLIST_MODE:
                _all, _e = _cached_hl_contexts()
                _by_name = {c["name"]: c for c in _all}
                _sel = [_by_name[r.market] for r in _wl_found if r.market in _by_name]
                instruments = [(c["label"], c["ticker"], (c["ticker"], None)) for c in _sel]
                # Watchlist coins that live on another venue.
                instruments += [(f"{sym} ({venue})", tk, (tk, None))
                                for _u, sym, tk, venue in _elsewhere]
                instruments += [(f"{sym} (CoinGecko)", f"{sym}-USD", (f"{sym}-USD", p["id"]))
                                for _u, sym, p in _via_cg]
                hl_marks = {c["ticker"]: c["mark"] for c in _sel}
                st.session_state.hl_info = {
                    c["ticker"]: hyperliquid_data.MarketContext(
                        c["name"], c["volume"], c["mark"], c["oi"], c["funding"],
                        c.get("change"), None)
                    for c in _sel}
            elif HL_MODE:
                _ctxs, _hl_err = hyperliquid_data.fetch_market_contexts()
                if not _ctxs:
                    st.error(f"Couldn't load Hyperliquid volumes: {_hl_err}")
                _top = {v.upper().replace("-USD", "")
                        for v in list(CRYPTO_TICKERS_ALL.values())[:int(hl_exclude_n)]}
                _picks = hyperliquid_data.rank_by_volume(
                    _ctxs, exclude_bases=_top, min_volume_usd=hl_min_vol * 1e6,
                    limit=universe_size)
                if _ctxs and not _picks:
                    st.warning("No Hyperliquid coins met the volume minimum after skipping "
                               "the top coins. Lower the minimum and try again.")
                instruments = [(c.label, c.ticker, (c.ticker, None)) for c in _picks]
                hl_marks = {c.ticker: c.mark_price for c in _picks}
                st.session_state.hl_info = {c.ticker: c for c in _picks}
            else:
                labels = list(CRYPTO_TICKERS)[:universe_size]
                instruments = [(lbl, CRYPTO_TICKERS[lbl],
                                 (CRYPTO_TICKERS[lbl], CRYPTO_CG_IDS.get(lbl)))
                               for lbl in labels]
                st.session_state.hl_info = {}

            bar = st.progress(0.0, text="Starting…")

            def _loader(payload):
                ticker, coin_id = payload
                if hyperliquid_data.is_hl_ticker(ticker):
                    out = {}
                    for tf, n in (("4h", 300), ("1d", 200), ("1h", 200)):
                        df, _e = hyperliquid_data.fetch_candles(ticker, tf, n)
                        out[tf] = df if df is not None else pd.DataFrame()
                    return out, []
                if CRYPTO_SOURCE == "exchange":
                    df, err = exchanges.fetch_binance_klines(ticker, "4h")
                    fetch_1d = exchanges.fetch_binance_klines
                    if df is None:
                        df, err = exchanges.fetch_kraken_ohlc(ticker, "4h")
                        fetch_1d = exchanges.fetch_kraken_ohlc
                    if df is not None and not df.empty:
                        # Daily candles let 1D/4H alignment be judged from two
                        # genuinely independent timeframes. Without them the
                        # alignment points are withheld rather than faked.
                        d1, _e = fetch_1d(ticker, "1d")
                        # 1H lets the entry-confirmation component be judged, so
                        # the full 10 points are reachable in a universe scan.
                        h1, _e = fetch_1d(ticker, "1h")
                        return {"4h": df, "1d": d1 if d1 is not None else pd.DataFrame(),
                                "1h": h1 if h1 is not None else pd.DataFrame()}, []
                if coin_id:
                    df, _label, err = coingecko.fetch_ohlc(coin_id, days=30)
                    if df is not None and not df.empty:
                        return {"4h": df, "1d": pd.DataFrame(), "1h": pd.DataFrame()}, []
                return {}, [err or "no data"]

            def _spot(payload):
                """Live price for the coin — from the SAME venue as its candles.

                Mixing sources is what produced plans priced nowhere near the
                market: the levels were read from one venue's candles while the
                "live price" came from another's. If the candles came from
                Hyperliquid, the price must be Hyperliquid's too.
                """
                ticker, coin_id = payload
                if NON_CRYPTO:
                    q = fetch_quote(ticker, "commodity",
                                     yf, live_threshold_seconds=live_threshold,
                                     stale_threshold_seconds=stale_threshold, max_retries=1)
                    return q.price if q and q.price else None

                base = ticker.upper().replace("HL:", "").replace("-USD", "")
                _ref = _REF_PRICES.get(ticker)
                if _ref:
                    return _ref            # CoinGecko: the price you'd verify against
                _from = (_loader_notes.get(ticker) or (None, ""))[0] or ""

                if hyperliquid_data.is_hl_ticker(ticker) or _from.startswith("Hyperliquid"):
                    px = hl_marks.get(ticker) or _HL_MARKS.get(base)
                    if px:
                        return px
                if _from.startswith("Bybit"):
                    _b = BYBIT_TICKERS.get(base)
                    if _b:
                        return _b["last"]
                if _from:
                    # Candles came from a venue with no quick quote — use their
                    # own last close rather than another venue's price.
                    _c = _last_closes.get(ticker)
                    if _c:
                        return _c
                if CRYPTO_SOURCE == "bybit":
                    _b = BYBIT_TICKERS.get(base)
                    if _b:
                        return _b["last"]
                px, _src, _err = exchanges.fetch_spot(ticker)
                if px is None and coin_id:
                    px, _upd, _e = coingecko.fetch_spot_price(coin_id)
                return px

            def _progress(i, total, label):
                bar.progress(min(i / max(total, 1), 1.0), text=f"{i}/{total} · {label}")

            _loader_notes = {}
            _last_closes = {}
            _PREFETCHED = {}
            _HL_MARKS = {}
            _REF_PRICES = {}          # ticker -> CoinGecko price, the reference
            if not NON_CRYPTO:
                _ctx, _cerr = _cached_hl_contexts()
                _HL_MARKS = {c["name"].upper(): c["mark"] for c in (_ctx or [])
                             if c.get("mark")}
                # CoinGecko aggregates across exchanges and is reachable from
                # anywhere, so it is the reference every candle source gets
                # checked against. A source whose last close disagrees with it
                # is describing a different market — or a different week.
                # Bundled markets (Hyperliquid's kPEPE, kBONK…) are quoted per
                # 1,000 coins, so the reference price has to be scaled to match
                # or every one of them looks 1,000x wrong.
                _ids, _mults = {}, {}
                for l, t, _k in instruments:
                    sym = exchanges.base_asset(t)
                    mult = 1.0
                    if (len(sym) > 1 and sym.startswith("K")
                            and sym[1:] in CRYPTO_CG_BY_SYMBOL):
                        sym, mult = sym[1:], 1000.0
                    _ids[t] = CRYPTO_CG_IDS.get(l) or _cg_id_for_symbol(sym)
                    _mults[t] = mult
                _cg_prices, _cg_err = _cached_cg_bulk(
                    tuple(sorted({v for v in _ids.values() if v})))
                _REF_PRICES = {t: _cg_prices[cid] * _mults.get(t, 1.0)
                               for t, cid in _ids.items()
                               if cid and _cg_prices.get(cid)}

            def _tr_loader(payload):
                """4H, 1H and 5m candles for the Trend Retrace rules."""
                ticker, _coin_id = payload
                if NON_CRYPTO:
                    frames, _p = _yahoo_frames(ticker)
                    return frames
                out = {}
                base = ticker.upper().replace("HL:", "").replace("-USD", "")

                # Ask every source that might have this coin, then use the
                # first frame that really is 5m data and really is current.
                # Trusting a source's label is what produced plans priced
                # nowhere near the market.
                def _five_minute():
                    pre = _PREFETCHED.get(ticker)
                    if pre is not None:
                        return pre

                    tried = []
                    if CRYPTO_SOURCE == "bybit":
                        bb, _e = exchanges.fetch_bybit_klines(ticker, "5m", limit=1000)
                        tried.append(("Bybit", bb))
                    bi, _e = exchanges.fetch_binance_klines(ticker, "5m", limit=1000)
                    tried.append(("Binance", bi))
                    cb, _e = exchanges.fetch_coinbase_candles(ticker, "5m", limit=300)
                    tried.append(("Coinbase", cb))
                    kr, _e = exchanges.fetch_kraken_ohlc(ticker, "5m")
                    tried.append(("Kraken", kr))
                    hl, _e = hyperliquid_data.fetch_candles(f"HL:{base}", "5m", 1000)
                    tried.append(("Hyperliquid", hl))

                    # Drop any source whose last close disagrees with the
                    # reference price — that is the check that was missing.
                    ref = _REF_PRICES.get(ticker)
                    checked, rejected = [], []
                    for name, frame in tried:
                        if frame is not None and not frame.empty and ref:
                            close = float(frame["Close"].iloc[-1])
                            gap = abs(close - ref) / ref * 100
                            if gap > 5:
                                rejected.append(
                                    f"{name}: last close {close:,.6g} vs {ref:,.6g} on "
                                    f"CoinGecko ({gap:.0f}% out)")
                                continue
                        checked.append((name, frame))
                    if not ref:
                        # No reference price available. Rather than trust a lone
                        # source, require two to agree — one wrong feed can't
                        # then set the levels on its own.
                        closes = [(n, float(f["Close"].iloc[-1]))
                                  for n, f in checked if f is not None and not f.empty]
                        agreed = None
                        for i_, (n1, c1) in enumerate(closes):
                            for n2, c2 in closes[i_ + 1:]:
                                if c1 > 0 and abs(c1 - c2) / c1 * 100 <= 5:
                                    agreed = {n1, n2}
                                    break
                            if agreed:
                                break
                        if closes and agreed is None:
                            spread = ", ".join(f"{n}={c:,.6g}" for n, c in closes)
                            return None, None, (
                                f"No CoinGecko price to check against, and the sources "
                                f"disagree ({spread}) — so none can be trusted.")
                        if agreed:
                            checked = [(n, f) for n, f in checked if n in agreed]

                    df_, src_, why_ = frame_check.first_valid(checked, "5m", min_bars=200)
                    return df_, src_, "; ".join(rejected + ([why_] if why_ else []))

                df, src, why = _five_minute()
                if df is not None:
                    _loader_notes[ticker] = (src, why)
                    _last_closes[ticker] = float(df["Close"].iloc[-1])
                    return trend_retrace.frames_from_5m(df)
                _loader_notes[ticker] = (None, why)
                return {"4h": pd.DataFrame(), "1h": pd.DataFrame(), "5m": pd.DataFrame()}

            if USE_MR:
                _mr_cache = {}

                def _mr_loader(payload):
                    tk = payload[0] if isinstance(payload, tuple) else payload
                    if tk in _mr_cache:
                        return _mr_cache[tk]
                    pre = _PREFETCHED.get(tk)
                    if pre is not None:
                        df, src, why = pre
                        _loader_notes[tk] = (src, why)
                        if df is not None:
                            _last_closes[tk] = float(df["Close"].iloc[-1])
                        _mr_cache[tk] = df
                        return df
                    df, src, why = _fetch_checked(tk, "1d", _REF_PRICES.get(tk),
                                                   min_bars=120)
                    _loader_notes[tk] = (src, why)
                    _mr_cache[tk] = df
                    return df

                _tickers = [t for _l, t, _k in instruments]
                bar.progress(0.0, text=f"Fetching {len(_tickers)} daily charts…")
                _PREFETCHED.update(_prefetch_candles(
                    _tickers, "1d", refs=_REF_PRICES, min_bars=120,
                    progress=lambda i, n, nm: bar.progress(
                        min(i / max(n, 1), 1.0), text=f"{i}/{n} · {nm}")))
                with st.spinner("Looking for stretched markets at a level…"):
                    ranked = mean_reversion.scan_universe(
                        instruments, _mr_loader, spot_loader=_spot,
                        params=MR_PARAMS, progress=_progress)

            elif USE_SWING:
                _daily_cache = {}
                # Hyperliquid last, not first — it is the source that was
                # returning prices from another period entirely.

                def _daily_loader(payload):
                    """Daily candles, checked the same way as the 5m ones.

                    This path previously trusted whatever Hyperliquid returned,
                    which is how a swing scan could price NEAR at 1.57 while it
                    traded at 5.49. Every source is now compared against the
                    CoinGecko reference before use.
                    """
                    tk = payload[0] if isinstance(payload, tuple) else payload
                    if tk in _daily_cache:
                        return _daily_cache[tk]
                    if NON_CRYPTO:
                        _fr, _p = _yahoo_frames(tk)
                        df = _fr.get("1d")
                        if df is None or df.empty:
                            _h = yf.Ticker(tk).history(period="3y", interval="1d")
                            df = _h.rename(columns=str.title) if _h is not None else None
                        _daily_cache[tk] = df
                        return df

                    pre = _PREFETCHED.get(tk)
                    if pre is not None:
                        df, src, why = pre
                        _loader_notes[tk] = (src, why)
                        if df is not None:
                            _last_closes[tk] = float(df["Close"].iloc[-1])
                        _daily_cache[tk] = df
                        return df

                    base = exchanges.base_asset(tk)
                    tried = []
                    if CRYPTO_SOURCE == "bybit":
                        _d, _e = exchanges.fetch_bybit_klines(tk, "1d", limit=1000)
                        tried.append(("Bybit", _d))
                    _d, _e = exchanges.fetch_binance_klines(tk, "1d", limit=1000)
                    tried.append(("Binance", _d))
                    _d, _e = exchanges.fetch_coinbase_candles(tk, "1d", limit=300)
                    tried.append(("Coinbase", _d))
                    _d, _e = exchanges.fetch_kraken_ohlc(tk, "1d")
                    tried.append(("Kraken", _d))
                    tried.append(("Hyperliquid", _bulk_daily.get(base)))

                    ref = _REF_PRICES.get(tk)
                    checked, rejected = [], []
                    for name, frame in tried:
                        if frame is not None and not frame.empty and ref:
                            close = float(frame["Close"].iloc[-1])
                            gap = abs(close - ref) / ref * 100
                            if gap > 8:      # daily closes move more than 5m ones
                                rejected.append(f"{name}: last close {close:,.6g} vs "
                                                f"{ref:,.6g} reference ({gap:.0f}% out)")
                                continue
                        checked.append((name, frame))
                    df, src, why = frame_check.first_valid(checked, "1d", min_bars=120)
                    _loader_notes[tk] = (src, "; ".join(rejected + ([why] if why else [])))
                    if df is not None:
                        _last_closes[tk] = float(df["Close"].iloc[-1])
                    _daily_cache[tk] = df
                    return df

                # One daily request per coin, in parallel within the budget.
                _bulk_daily = {}
                if not NON_CRYPTO:
                    _names = [f"HL:{t.upper().replace('HL:', '').replace('-USD', '')}"
                              for _l, t, _k in instruments]
                    bar.progress(0.0, text=f"Fetching {len(_names)} daily charts…")
                    _got = hyperliquid_data.fetch_candles_many(
                        _names, "1d", 1000,
                        progress=lambda i, n, nm: bar.progress(
                            min(i / max(n, 1), 1.0), text=f"{i}/{n} · {nm}"))
                    _bulk_daily = {k.replace("HL:", ""): v for k, v in _got.items()}

                if not NON_CRYPTO:
                    _tickers = [t for _l, t, _k in instruments]
                    bar.progress(0.0, text=f"Fetching {len(_tickers)} daily charts…")
                    _PREFETCHED.update(_prefetch_candles(
                        _tickers, "1d", refs=_REF_PRICES, min_bars=120,
                        hl_frames=_bulk_daily,
                        progress=lambda i, n, nm: bar.progress(
                            min(i / max(n, 1), 1.0), text=f"{i}/{n} · {nm}")))

                with st.spinner("Applying the swing rules…"):
                    ranked = swing.scan_universe(
                        instruments, _daily_loader, spot_loader=_spot,
                        params=SWING_PARAMS, progress=_progress)
            elif USE_TR:
                with st.spinner("Scanning…"):
                    _extra = {"sentiment": sentiment.bucket(_fg["value"])} if _fg else None
                    _level_cache = {}

                    # Fetch every Hyperliquid coin's candles at once. The venue
                    # limits total weight per minute, not the gap between
                    # calls, so 20 coins fit comfortably inside one window and
                    # finish in seconds instead of one request at a time.
                    # Bulk-fetch EVERY crypto ticker, not just HL:-prefixed ones.
                    # In market-cap mode the tickers look like BTC-USD, so the
                    # old check never matched and every coin fell through to
                    # the slow per-timeframe path.
                    _bulk, _bulk_key = {}, {}
                    if not NON_CRYPTO:
                        for _l, _t, _k in instruments:
                            _bulk_key[_t] = (_t if hyperliquid_data.is_hl_ticker(_t)
                                             else f"HL:{_t.upper().replace('-USD', '')}")
                        bar.progress(0.0, text=f"Fetching {len(_bulk_key)} coins…")
                        _fetched = hyperliquid_data.fetch_candles_many(
                            list(_bulk_key.values()), "5m", 1000,
                            progress=lambda i, n, nm: bar.progress(
                                min(i / max(n, 1), 1.0), text=f"{i}/{n} · {nm}"))
                        _bulk = {tk: _fetched.get(key) for tk, key in _bulk_key.items()}

                    _diag = {}

                    def _tr_loader_capture(payload):
                        _tk = payload[0]
                        _pre = _bulk.get(_tk)
                        if _pre is not None:
                            _chk = frame_check.check(_pre, "5m", min_bars=300)
                            if not _chk.ok:
                                _loader_notes[_tk] = ("Hyperliquid (rejected)", _chk.reason)
                                _pre = None      # fall through to the other sources
                        if _pre is not None and len(_pre) >= 720:
                            frames = trend_retrace.frames_from_5m(_pre)
                            _src = "Hyperliquid 5m (bulk)"
                            _loader_notes[_tk] = ("Hyperliquid", "")
                            _last_closes[_tk] = float(_pre["Close"].iloc[-1])
                        else:
                            frames = _tr_loader(payload)
                            _src = "per-timeframe fallback"
                        _f5 = frames.get("5m")
                        if _f5 is not None and not _f5.empty:
                            _diag[_tk] = {
                                "source": _src, "bars": len(_f5),
                                "first": _f5.index[0], "last": _f5.index[-1],
                                "last_close": float(_f5["Close"].iloc[-1]),
                                "low": float(_f5["Low"].min()),
                                "high": float(_f5["High"].max()),
                            }
                        # Keep the 5m structure so a ladder can be offered later
                        # without re-fetching everything.
                        _level_cache[payload[0]] = {
                            "5m": frames.get("5m"),
                            "atr": trend_retrace.atr(frames.get("5m"))}
                        return frames

                    _tickers = [t for _l, t, _k in instruments]
                    bar.progress(0.0, text=f"Fetching {len(_tickers)} coins…")
                    _PREFETCHED.update(_prefetch_candles(
                        _tickers, "5m", refs=_REF_PRICES, min_bars=200,
                        hl_frames={exchanges.base_asset(k): v
                                   for k, v in (_bulk or {}).items() if v is not None},
                        progress=lambda i, n, nm: bar.progress(
                            min(i / max(n, 1), 1.0), text=f"{i}/{n} · {nm}")))
                    ranked = trend_retrace.scan_universe_tr(
                        instruments, _tr_loader_capture, spot_loader=_spot,
                        params=TR_PARAMS, progress=_progress, extra_features=_extra)
                    st.session_state.scan_diag = _diag
                    st.session_state.uni_levels = {
                        r.ticker: {
                            "levels": trend_retrace.five_minute_levels(
                                _level_cache.get(r.ticker, {}).get("5m"),
                                r.direction, r.price or 0, TR_PARAMS),
                            "atr": _level_cache.get(r.ticker, {}).get("atr"),
                        }
                        for r in ranked if r.direction and r.price
                        and _level_cache.get(r.ticker, {}).get("5m") is not None}
            else:
                with st.spinner("Scanning…"):
                    ranked = scan_universe(
                        instruments, _loader, min_rr=uni_min_rr,
                        direction_override=None if uni_direction == "Auto" else uni_direction,
                        progress_callback=_progress, spot_loader=_spot)
            bar.empty()
            if _loader_notes:
                _bad = {t: n for t, n in _loader_notes.items() if n[1]}
                _used = {t: n[0] for t, n in _loader_notes.items() if n[0]}
                with st.expander(
                        f"🔧 Data check — sources used"
                        + (f" · {len(_bad)} source(s) rejected" if _bad else "")):
                    st.caption("Every source is compared against the CoinGecko price before "
                               "its candles are used. Anything more than a few percent out "
                               "is a different market or a different week, and is refused.")
                    if _used:
                        st.dataframe(pd.DataFrame(
                            [{"Ticker": t, "Candles from": s,
                              "Reference price": format_price(_REF_PRICES.get(t))
                                                 if _REF_PRICES.get(t) else "—",
                              "Last close": format_price(_last_closes.get(t))
                                            if _last_closes.get(t) else "—"}
                             for t, s in _used.items()]),
                            use_container_width=True, hide_index=True)
                    for t, (s, why) in _bad.items():
                        st.caption(f"**{t}** — {why}")

            _d = st.session_state.get("scan_diag") or {}
            if _d:
                _now = pd.Timestamp.now(tz="UTC")
                _rows_d = []
                for _tk, _info in list(_d.items())[:60]:
                    _age = (_now - _info["last"]).total_seconds() / 3600
                    _rows_d.append({
                        "Ticker": _tk, "Source": _info["source"], "Bars": _info["bars"],
                        "Newest candle": _info["last"].strftime("%d %b %H:%M"),
                        "Age (h)": f"{_age:.1f}",
                        "Last close": format_price(_info["last_close"]),
                        "Range": f"{format_price(_info['low'])}–"
                                 f"{format_price(_info['high'])}",
                    })
                _stale_rows = [r for r in _rows_d if float(r["Age (h)"]) > 2]
                with st.expander(
                        f"🔧 Data check — where each price came from"
                        + (f" ({len(_stale_rows)} look stale)" if _stale_rows else "")):
                    st.caption("If a plan's levels look wrong, check this first: the newest "
                               "candle should be minutes old and the last close should match "
                               "the live price.")
                    st.dataframe(pd.DataFrame(_rows_d), use_container_width=True,
                                 hide_index=True)

            _blocked = exchanges.blocked_hosts()
            if _blocked:
                st.caption("Unreachable from this server, so skipped: "
                           + ", ".join(sorted(_blocked)) + ". Candles came from "
                           "Hyperliquid instead.")
            st.session_state.ranked = ranked
            if st.session_state.get("track_signals", True):
                added = tracking.record_from_ranked(
                    ranked, VENUE_TAGS, grades=None,
                    min_rr=(min(MIN_RR, TR_PARAMS.target_r) if USE_TR and TR_PARAMS
                            else MIN_RR))
                if added:
                    st.toast(f"Recorded {added} new signal(s) for forward tracking.")

        ranked = st.session_state.get("ranked")
        if ranked:
            scored_all = [r for r in ranked if r.error is None]

            # Anything priced a long way from the market is dropped outright,
            # whatever its score. An entry 71% below the live price is never
            # something to act on, and the reason hardly matters.
            _too_far = []
            _near_enough = []
            for r in scored_all:
                if r.entry and r.price:
                    _gap = abs(r.entry - r.price) / r.price * 100
                    if _gap > MAX_ENTRY_GAP:
                        _too_far.append((r, _gap))
                        continue
                _near_enough.append(r)
            scored_all = _near_enough
            # Never hide setups that meet the target the trader chose.
            if USE_MR and MR_PARAMS:
                _rr_floor = MR_PARAMS.target_r      # 0.4:1 is the design, not a fault
            elif USE_TR and TR_PARAMS:
                _rr_floor = min(MIN_RR, TR_PARAMS.target_r)
            else:
                _rr_floor = max(MIN_RR, uni_min_rr)
            failed = [r for r in ranked if r.error is not None]

            _ev_raw, _ev_when = storage.load_value("evidence")
            EVIDENCE = None
            if _ev_raw:
                if _ev_raw.get("config") == _config_signature(USE_TR, TR_PARAMS):
                    EVIDENCE = expectancy.EvidenceBook.from_dict(_ev_raw)
                else:
                    st.warning(
                        "Your backtest evidence was measured with different strategy "
                        "settings (stop, target or strategy have changed since). It no "
                        "longer applies, so it isn't used. Re-run the 🔁 Backtest.")

            f1, f2 = st.columns(2)
            min_score = f1.number_input(
                "Show scores of at least", min_value=0.0, max_value=10.0, value=0.0,
                step=0.5, key="uni_min_score",
                help="Scores are whole numbers, so 8.5 behaves the same as 9. "
                     "A 9 or 10 appears in only a few percent of market states.")
            a_plus_only = f2.checkbox("A+ only", key="uni_aplus_only")
            proven_only = st.checkbox(
                "Only show setups with proven positive expectancy",
                key="uni_proven_only", disabled=EVIDENCE is None,
                help="Hides anything the backtest hasn't shown to make money over many "
                     "trades. Run the backtest first to build the evidence.")
            if EVIDENCE is None:
                st.caption("ℹ️ No evidence yet — run the 🔁 Backtest to label setups with "
                           "whether they've made money historically.")
            else:
                st.caption(f"Evidence from: {EVIDENCE.source}.")

            def _evidence(r):
                if EVIDENCE is None:
                    return None, ""
                return EVIDENCE.evidence_for(r.ticker, r.direction)

            _lm_raw, _ = storage.load_value("learned_model")
            LEARNED = None
            if _lm_raw and _lm_raw.get("config") == _config_signature(USE_TR, TR_PARAMS):
                LEARNED = learning.LearnedModel.from_dict(_lm_raw)
            _has_lessons = LEARNED is not None and bool(LEARNED.rules)
            apply_lessons = st.checkbox(
                "Skip setups the system has learned to avoid",
                key="uni_apply_lessons", disabled=not _has_lessons,
                help="Hides setups matching a lesson confirmed on unseen trades.")
            if _has_lessons:
                st.caption("Lessons in use (from " + LEARNED.source + "): "
                           + "; ".join(f"skip when {r_.description}" for r_ in LEARNED.rules))
            elif LEARNED is not None:
                st.caption("The system has checked its trades and found no lesson worth "
                           "acting on yet — so it isn't filtering anything.")

            def _lesson(r):
                if not _has_lessons:
                    return True, []
                return LEARNED.check(r.features)

            ok = [r for r in scored_all
                  if r.score >= min_score and (not a_plus_only or r.grade == "A+")
                  and r.reward_risk is not None and r.reward_risk >= _rr_floor - 1e-9
                  and (not proven_only
                       or _evidence(r)[0] == expectancy.PROVEN_POSITIVE)
                  and (not apply_lessons or _lesson(r)[0])]
            hidden = len(scored_all) - len(ok)

            st.markdown(f"##### Results · {len(ok)} shown"
                        + (f", {hidden} hidden by filter" if hidden else "")
                        + (f", {len(failed)} failed" if failed else ""))
            if _too_far:
                with st.expander(f"{len(_too_far)} setup(s) too far from the live price"):
                    st.caption(f"Dropped because the entry is more than {MAX_ENTRY_GAP:g}% "
                               f"from the current price. Adjust the limit in the sidebar.")
                    for r, gap in sorted(_too_far, key=lambda x: -x[1]):
                        st.caption(f"• **{r.label}** — entry {format_price(r.entry)} vs "
                                   f"live {format_price(r.price)} ({gap:.0f}% away)")

            if hidden:
                _why_hidden = []
                for r in scored_all:
                    if r in ok:
                        continue
                    if r.reward_risk is None:
                        _why_hidden.append(f"{r.label}: no target far enough away to give "
                                           f"{_rr_floor:g}:1, so no reward:risk")
                    elif r.reward_risk < _rr_floor - 1e-9:
                        _why_hidden.append(f"{r.label}: {format_rr(r.reward_risk)} is below "
                                           f"your {_rr_floor:g}:1 minimum")
                    elif r.score < min_score:
                        _why_hidden.append(f"{r.label}: scored {r.score:.0f}, below your "
                                           f"{min_score:g} minimum")
                    elif a_plus_only and r.grade != "A+":
                        _why_hidden.append(f"{r.label}: grade {r.grade}, and A+ only is on")
                    elif proven_only:
                        _why_hidden.append(f"{r.label}: no proven positive expectancy yet")
                    elif apply_lessons:
                        _why_hidden.append(f"{r.label}: matches a lesson you're filtering out")
                with st.expander(f"Why {hidden} setup(s) are hidden"):
                    for line in _why_hidden:
                        st.caption(f"• {line}")
            st.caption(f"Setups with reward:risk below {_rr_floor:g}:1 are never shown.")
            if EVIDENCE is not None and ok:
                with st.expander("Why each setup has the evidence label it does"):
                    for r in ok:
                        v, why = _evidence(r)
                        st.caption(f"**{r.label} {r.direction or ''}** — {v}. {why}")
            if scored_all and not ok:
                best = max(scored_all, key=lambda r: r.score)
                st.info(
                    f"Nothing meets the filter right now. The best setup scanned was "
                    f"**{best.label} at {best.score:.0f}/10 ({best.grade})**. An empty list is "
                    f"a legitimate answer — your rulebook treats no-trade as a valid decision.")
            if ok:
                def _gap(r):
                    if not (r.entry and r.price):
                        return "—"
                    g = (r.entry - r.price) / r.price * 100
                    return "at price" if abs(g) < 0.05 else f"{g:+.2f}%"

                _hlinfo = st.session_state.get("hl_info") or {}
                table = pd.DataFrame([{
                    "Instrument": r.label,
                    "Trade plan": explain.short_plan(
                        r.direction, _plan_stage(r, USE_TR), r.entry, r.stop, r.target,
                        r.reward_risk, r.price),
                    "Learned": ("—" if not _has_lessons else
                                ("✅ OK" if _lesson(r)[0] else "⚠️ Avoid")),
                    "Evidence": {expectancy.PROVEN_POSITIVE: "✅ Proven +",
                                 expectancy.PROVEN_NEGATIVE: "❌ Proven −",
                                 expectancy.UNPROVEN: "❔ Unproven",
                                 expectancy.TOO_FEW: "— Too few"}.get(_evidence(r)[0], "—"),
                    "Venue": ("—" if NON_CRYPTO else
                              "HL" if hyperliquid_data.is_hl_ticker(r.ticker)
                              else "/".join(VENUE_TAGS.get(r.label, [])) or "—"),
                    **({"24h volume": f"${_hlinfo[r.ticker].day_volume_usd / 1e6:,.0f}M",
                        "Open interest": f"${_hlinfo[r.ticker].open_interest_usd / 1e6:,.0f}M",
                        "Funding (1h)": f"{_hlinfo[r.ticker].funding_rate * 100:+.4f}%"}
                       if r.ticker in _hlinfo else {}),
                    "Score": f"{r.score:.1f}",
                    "Grade": r.grade,
                    "Dir": r.direction or "—",
                    "Live price": format_price(r.price) + ("" if r.price_is_live else " *"),
                    "Entry": format_price(r.entry),
                    "Entry vs live": _gap(r),
                    "Stop": format_price(r.stop),
                    "Target": format_price(r.target),
                    "R:R": format_rr(r.reward_risk),
                    ("Status" if USE_TR else "Regime"):
                        (r.entry_status or "—") if USE_TR else r.regime.title(),
                } for r in ok])
                _detail = st.toggle("Show every column", value=False, key="uni_detail",
                                    help="Off keeps the essentials only — easier to read on "
                                         "a phone.")
                _essential = ["Instrument", "Dir", "Live price", "Entry", "Entry vs live",
                              "Stop", "Target", "R:R", "Status", "Grade"]
                _view = table if _detail else table[[c for c in _essential
                                                     if c in table.columns]]
                st.dataframe(_view, use_container_width=True, hide_index=True,
                             column_config={"Trade plan": st.column_config.TextColumn(
                                 "Trade plan", width="large")})
                if not _detail:
                    st.caption("Showing the essentials. Turn on **Show every column** for "
                               "venue, volume, funding, evidence, the learned check and the "
                               "one-line plan.")

                _complete = [r for r in ok if None not in (r.entry, r.stop, r.target)]
                if _complete:
                    with st.expander(f"📋 Trade plans ({len(_complete)})",
                                     expanded=len(_complete) <= 5):
                        for r in _complete:
                            _sm, _tm = _methods(USE_TR, TR_PARAMS, r.direction)
                            _reasons = [x.strip() for x in (r.note or "").split(";") if x.strip()]
                            with st.container(border=True):
                                _ok_l, _why_l = _lesson(r)
                                if not _ok_l:
                                    st.warning("⚠️ The system has learned to avoid setups like "
                                               "this: " + "; ".join(_why_l) + ".")
                                for line in explain.full_plan(
                                        r.label, r.direction, _plan_stage(r, USE_TR),
                                        r.entry, r.stop, r.target, r.reward_risk, r.price,
                                        reasons=_reasons if USE_TR else None,
                                        stop_method=_sm, target_method=_tm,
                                        reentry_steps=(trend_retrace.reentry_plan_text(
                                            r.direction) if USE_TR else None)):
                                    st.markdown(line)
                                _sz = _size_plan(r.direction, r.entry, r.stop, r.target,
                                                  r.ticker)
                                _render_size(_sz)
                                _render_trailing(
                                    r.direction, r.entry, r.stop,
                                    (r.features or {}).get("atr")
                                    or (abs(r.entry - r.stop) / (TR_PARAMS.stop_atr_mult
                                        if (USE_TR and TR_PARAMS) else 0.75)
                                        if r.entry and r.stop else None),
                                    key=f"u_{r.ticker}")
                                if r.ticker in COMMODITY_PERPS:
                                    st.caption(f"Place this on **{COMMODITY_PERPS[r.ticker]}**.")
                                _lv = st.session_state.get("uni_levels", {}).get(r.ticker)
                                if _lv:
                                    _render_grid(f"u_{r.ticker}", r.direction, r.price,
                                                 _lv.get("levels", []), _lv.get("atr"),
                                                 r.reward_risk)
                                _take_trade_widget(
                                    f"u_{r.ticker}_{r.direction}", r.ticker, r.direction,
                                    r.entry, r.stop, r.target, features=r.features,
                                    score=r.score, grade=r.grade,
                                    reason=f"Scanner: {r.entry_status or ''}",
                                    suggested_qty=_sz.quantity if _sz else 0.0)
                st.caption(
                    "**Entry** is the planned price to place your order at — a limit order "
                    "at a confluence zone, not the market price. **Entry vs live** shows how "
                    "far price must travel to fill it. Stop, target and R:R are all measured "
                    "from that entry. A price marked * is the last 4H close because the live "
                    "spot fetch failed.")

                _counts = {g: sum(1 for r in ok if r.grade == g) for g in ("A+", "B", "C")}
                st.markdown(
                    " ".join(theme.pill(f"{g} · {n}", theme.GRADE_COLOURS[g])
                             for g, n in _counts.items() if n),
                    unsafe_allow_html=True)

                tradeable = [r for r in ok if r.grade in ("A+", "B")]
                if tradeable:
                    st.success(
                        f"{len(tradeable)} setup(s) at grade B or better: "
                        + ", ".join(f"{r.label} ({r.grade}, {r.score:.1f})" for r in tradeable[:5])
                    )
                else:
                    st.info(
                        "Nothing reached grade B. Per the rulebook, C setups are not traded — "
                        "a no-trade decision is valid and common."
                    )
            if failed:
                with st.expander(f"⚠️ {len(failed)} instrument(s) could not be scored"):
                    for r in failed:
                        st.caption(f"**{r.label}** — {r.error}")

        score_evidence, seq_evidence = {}, {}

    elif mode == "Auto-scan one instrument" and USE_MR:
        mr1, mr2 = st.columns([3, 1])
        _mr_market = mr1.selectbox("Market", ["Crypto", "Commodities", "FX"],
                                   key="mrs_market")
        _mr_lists = {"Crypto": CRYPTO_TICKERS, "Commodities": COMMODITY_TICKERS,
                     "FX": FX_TICKERS}
        _mr_name = mr2.selectbox("Instrument", list(_mr_lists[_mr_market]), key="mrs_name")
        _mr_ticker = _mr_lists[_mr_market][_mr_name]

        st.caption("Checks whether this market is stretched from its average and sitting "
                   "at a price it has respected before, with a candle rejecting that level.")

        if st.button("🔍 Check this market", use_container_width=True, key="mrs_go"):
            with st.spinner("Loading daily candles…"):
                _ref = None
                _cid = _cg_id_for_symbol(exchanges.base_asset(_mr_ticker))
                if _cid:
                    _p, _e = _cached_cg_bulk((_cid,))
                    _ref = _p.get(_cid)
                _df, _src, _why = _fetch_checked(_mr_ticker, "1d", _ref, min_bars=120)
            if _df is None:
                st.error(f"No usable daily candles. {_why}")
            else:
                st.session_state.mr_plan = mean_reversion.analyze(
                    _mr_ticker, _df, price=_ref, params=MR_PARAMS)
                st.session_state.mr_src = _src

        _mrp = st.session_state.get("mr_plan")
        if _mrp is None:
            st.info("Pick a market and press **Check this market**.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Instrument", _mrp.ticker)
            h2.metric("Direction", _mrp.direction or "—")
            h3.metric("Score", f"{_mrp.score:.0f}/10", _mrp.grade)
            h4.metric("Price", format_price(_mrp.price))
            st.markdown(f"##### Status: **{_mrp.stage}**")
            for r_ in _mrp.reasons:
                st.caption(f"• {r_}")
            if _mrp.entry and _mrp.stop and _mrp.target:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Entry", format_price(_mrp.entry))
                e2.metric("Stop", format_price(_mrp.stop))
                e3.metric("Target", format_price(_mrp.target))
                e4.metric("R:R", format_rr(_mrp.reward_risk))
                _szm = _size_plan(_mrp.direction, _mrp.entry, _mrp.stop, _mrp.target,
                                   _mrp.ticker)
                _render_size(_szm)
                if st.button("💾 Save this setup to re-check later",
                             key=f"savemr_{_mrp.ticker}", use_container_width=True):
                    _sid = tracking.record_one(_mrp, source="mean-reversion")
                    st.success(f"Saved as #{_sid}." if _sid else "Already saved.")
                _take_trade_widget(f"mr_{_mrp.ticker}", _mrp.ticker, _mrp.direction,
                                   _mrp.entry, _mrp.stop, _mrp.target,
                                   features=_mrp.features, score=_mrp.score,
                                   grade=_mrp.grade,
                                   reason=f"Mean Reversion: {_mrp.stage}",
                                   suggested_qty=_szm.quantity if _szm else 0.0)
            st.caption(f"Candles from {st.session_state.get('mr_src', 'unknown')}.")

        score_evidence, seq_evidence = {}, {}

    elif mode == "Auto-scan one instrument" and USE_SWING:
        sw1, sw2 = st.columns([3, 1])
        _sw_market = sw1.selectbox("Market", ["Crypto", "Commodities", "FX"],
                                   key="sw_market")
        _sw_lists = {"Crypto": CRYPTO_TICKERS, "Commodities": COMMODITY_TICKERS,
                     "FX": FX_TICKERS}
        _sw_name = sw2.selectbox("Instrument", list(_sw_lists[_sw_market]), key="sw_name")
        _sw_ticker = _sw_lists[_sw_market][_sw_name]

        st.caption("Checks daily candles: is there a clear daily trend, is price at a level "
                   "the market has respected before, and has a confirming candle printed. "
                   "One look a day is enough.")

        if st.button("🔍 Check the daily chart", use_container_width=True):
            with st.spinner("Loading daily candles…"):
                _d = None
                if _sw_market == "Crypto":
                    _d, _e = hyperliquid_data.fetch_candles(
                        f"HL:{_sw_ticker.upper().replace('-USD', '')}", "1d", 1000)
                    if _d is None:
                        _d, _e = exchanges.fetch_bybit_klines(_sw_ticker, "1d", limit=1000)
                    if _d is None:
                        _d, _e = exchanges.fetch_binance_klines(_sw_ticker, "1d", limit=1000)
                else:
                    _frames, _p = _yahoo_frames(_sw_ticker)
                    _d = _frames.get("1d", pd.DataFrame())
                    if _d is None or _d.empty:
                        _hist = yf.Ticker(_sw_ticker).history(period="3y", interval="1d")
                        _d = _hist.rename(columns=str.title) if _hist is not None else None
                if _d is None or _d.empty:
                    st.error("Couldn't load daily candles for this instrument.")
                else:
                    st.session_state.sw_plan = swing.analyze(_sw_ticker, _d,
                                                             params=SWING_PARAMS)
                    st.session_state.sw_bars = len(_d)

        _swp = st.session_state.get("sw_plan")
        if _swp is None:
            st.info("Pick an instrument and press **Check the daily chart**.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Instrument", _swp.ticker)
            h2.metric("Direction", _swp.direction or "—")
            h3.metric("Score", f"{_swp.score:.0f}/10", _swp.grade)
            h4.metric("Price", format_price(_swp.price))
            st.markdown(f"##### Status: **{_swp.stage}**")
            for r_ in _swp.reasons:
                st.caption(f"• {r_}")
            _sw_gap_ok = True
            if _swp.entry and _swp.price:
                _g = abs(_swp.entry - _swp.price) / _swp.price * 100
                if _g > MAX_ENTRY_GAP:
                    _sw_gap_ok = False
                    st.error(
                        f"**Not actionable** — the entry ({format_price(_swp.entry)}) is "
                        f"{_g:.0f}% from the live price ({format_price(_swp.price)}), past "
                        f"your {MAX_ENTRY_GAP:g}% limit.")
            if _swp.entry and _swp.stop and _swp.target and _sw_gap_ok:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Entry (limit)", format_price(_swp.entry))
                e2.metric("Stop", format_price(_swp.stop))
                e3.metric("Target", format_price(_swp.target))
                e4.metric("R:R", format_rr(_swp.reward_risk))
                _szs = _size_plan(_swp.direction, _swp.entry, _swp.stop, _swp.target,
                                   _swp.ticker)
                _render_size(_szs)
                _render_trailing(_swp.direction, _swp.entry, _swp.stop,
                                  (abs(_swp.entry - _swp.stop)
                                   / (SWING_PARAMS.stop_atr if SWING_PARAMS else 0.75))
                                  if _swp.entry and _swp.stop else None,
                                  key=f"sw_{_swp.ticker}")
                if st.button("💾 Save this setup to re-check later",
                             key=f"savesw_{_swp.ticker}", use_container_width=True):
                    _sid = tracking.record_one(_swp, source="swing-check")
                    st.success(f"Saved as #{_sid}. Re-check it on the 📈 Tracking tab."
                               if _sid else "Already saved — see the 📈 Tracking tab.")
                _take_trade_widget(f"sw_{_swp.ticker}", _swp.ticker, _swp.direction,
                                   _swp.entry, _swp.stop, _swp.target,
                                   features=_swp.features, score=_swp.score,
                                   grade=_swp.grade,
                                   reason=f"Swing Levels: {_swp.stage}",
                                   suggested_qty=_szs.quantity if _szs else 0.0)
            st.caption(f"Read from {st.session_state.get('sw_bars', 0)} daily candles. "
                       f"Trades typically last days to weeks — check once a day.")

        score_evidence, seq_evidence = {}, {}

    elif mode == "Auto-scan one instrument" and USE_TR:
        tr1, tr2 = st.columns([3, 1])
        _tl = tr1.selectbox("Coin", list(CRYPTO_TICKERS), key="tr_single")
        tr_ticker = CRYPTO_TICKERS[_tl]
        tr_dir = tr2.selectbox("Direction", ["Auto", "Long", "Short"], key="tr_single_dir",
                               help="Auto takes the direction from the 4H candles.")
        stopped_out = st.checkbox("I was just stopped out on this coin — show re-entry plan",
                                  key="tr_stopped")

        if st.button("🔍 Check the rules", use_container_width=True):
            with st.spinner(f"Checking {tr_ticker}…"):
                frames = {}
                for tf, lim in (("4h", 30), ("1h", 48), ("5m", 400)):
                    df = None
                    if CRYPTO_SOURCE == "bybit":
                        df, _e = exchanges.fetch_bybit_klines(tr_ticker, tf, limit=lim)
                    if df is None:
                        df, _e = exchanges.fetch_binance_klines(tr_ticker, tf, limit=lim)
                    if df is None:
                        df, _e = exchanges.fetch_kraken_ohlc(tr_ticker, tf)
                    if df is None:
                        df, _e = hyperliquid_data.fetch_candles(
                            f"HL:{tr_ticker.upper().replace('-USD', '')}", tf, lim)
                    frames[tf] = trend_retrace.drop_forming(
                        df if df is not None else pd.DataFrame(), tf)
                live, _src, _err = _cached_exchange_spot(tr_ticker)
                st.session_state.tr_plan = trend_retrace.analyze(
                    tr_ticker, frames["4h"], frames["1h"], frames["5m"], live_price=live,
                    params=TR_PARAMS,
                    direction_override=None if tr_dir == "Auto" else tr_dir)
                _p = st.session_state.tr_plan
                _pats = patterns.detect(frames["5m"]) + patterns.detect(frames["1h"])
                st.session_state.tr_patterns = _pats
                if _p.features is not None and _pats:
                    _p.features["pattern"] = patterns.summarise(_pats)
                    _p.features["pattern_bias"] = patterns.bias_of(_pats)
                st.session_state.tr_levels = {
                    "levels": (trend_retrace.five_minute_levels(
                        frames["5m"], _p.direction, _p.current_price or 0, TR_PARAMS)
                        if _p.direction and _p.current_price else []),
                    "atr": trend_retrace.atr(frames["5m"]),
                }

        plan = st.session_state.get("tr_plan")
        if plan is None:
            st.info("Pick a coin and press **Check the rules**.")
        else:
            h1c, h2c, h3c, h4c = st.columns(4)
            h1c.metric("Coin", plan.ticker)
            h2c.metric("Direction", plan.direction or "—")
            h3c.metric("Score", f"{plan.score:.0f}/10", plan.grade)
            h4c.metric("Live price", format_price(plan.current_price))

            st.markdown(f"##### Status: **{plan.stage}**")
            labels = {"4h": "1 · 4H trend — two candles of HH/HL (LH/LL for shorts)",
                      "1h": "2 · 1H confirmation candle",
                      "5m": "3 · 5m retrace to previous support (resistance for shorts)"}
            for key_, text in labels.items():
                rule = plan.rules.get(key_)
                if rule is None:
                    st.markdown(f"⬜ **{text}** — not checked (no direction yet)")
                else:
                    st.markdown(f"{'✅' if rule.passed else '❌'} **{text}**")
                    st.caption(f"　↳ {rule.reason}")

            if plan.entry is not None:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Entry (limit)", format_price(plan.entry))
                e2.metric("Stop", format_price(plan.stop))
                e3.metric("Target", format_price(plan.target))
                e4.metric("R:R", format_rr(plan.reward_risk))
                if plan.current_price:
                    gap = (plan.entry - plan.current_price) / plan.current_price * 100
                    st.caption(f"Entry is {gap:+.2f}% from the live price — a limit order "
                               f"waiting for the retrace, not a market buy.")
            _gap_ok = True
            if plan.entry and plan.current_price:
                _g = abs(plan.entry - plan.current_price) / plan.current_price * 100
                if _g > MAX_ENTRY_GAP:
                    _gap_ok = False
                    st.error(
                        f"**Not actionable** — the entry ({format_price(plan.entry)}) is "
                        f"{_g:.0f}% from the live price ({format_price(plan.current_price)}), "
                        f"past your {MAX_ENTRY_GAP:g}% limit. Either the level is days away "
                        f"or the candles disagree with the market. No plan is shown.")
            for p_ in plan.problems:
                st.warning(p_)
            _pats = st.session_state.get("tr_patterns") or []
            if _pats:
                st.markdown("##### Candlestick patterns right now")
                for _pt in _pats:
                    _icon = {"bullish": "🟢", "bearish": "🔴"}.get(_pt.bias, "⚪")
                    st.markdown(f"{_icon} **{_pt.name}** — {_pt.description}")
                    st.caption(f"　{_pt.convention}")
                if plan.direction and patterns.contradicts(_pats, plan.direction):
                    st.warning(f"At least one pattern points against this "
                               f"{plan.direction.lower()}.")
                st.caption("Patterns describe what just happened; they don't predict what "
                           "comes next. They're recorded with the setup so the learner can "
                           "test whether they make any difference to your results.")

            if plan.quality_fails:
                st.warning("**Weak setup — your three rules pass, but:**")
                for q_ in plan.quality_fails:
                    st.caption(f"• {q_}")

            if plan.entry is not None:
                _sm, _tm = _methods(True, TR_PARAMS, plan.direction)
                with st.container(border=True):
                    st.markdown("##### Trade plan")
                    for line in explain.full_plan(
                            plan.ticker, plan.direction, plan.stage, plan.entry, plan.stop,
                            plan.target, plan.reward_risk, plan.current_price,
                            reasons=[r_.reason for r_ in plan.rules.values()],
                            stop_method=_sm, target_method=_tm):
                        st.markdown(line)
                    _sz = _size_plan(plan.direction, plan.entry, plan.stop, plan.target,
                                      plan.ticker)
                    _render_size(_sz)
                    _render_trailing(plan.direction, plan.entry, plan.stop,
                                      (st.session_state.get("tr_levels") or {}).get("atr"),
                                      key=f"s_{plan.ticker}")
                    _tr_levels = st.session_state.get("tr_levels") or {}
                    _render_grid(f"s_{plan.ticker}", plan.direction, plan.current_price,
                                 _tr_levels.get("levels", []), _tr_levels.get("atr"),
                                 plan.reward_risk)
                    if st.button("💾 Save this setup to re-check later",
                                 key=f"save_{plan.ticker}", use_container_width=True):
                        _sid = tracking.record_one(plan, source="single-check")
                        st.success(f"Saved as #{_sid}. Re-check it on the 📈 Tracking tab."
                                   if _sid else
                                   "Already saved — it's on the 📈 Tracking tab.")
                    _take_trade_widget(
                        f"s_{plan.ticker}_{plan.direction}", plan.ticker, plan.direction,
                        plan.entry, plan.stop, plan.target, features=plan.features,
                        score=plan.score, grade=plan.grade,
                        reason=f"Single-coin check: {plan.stage}",
                        suggested_qty=_sz.quantity if _sz else 0.0)

            if plan.stage == trend_retrace.AT_ENTRY:
                st.success("All three rules are met and price is at the entry level.")
            elif plan.stage == trend_retrace.WAIT_RETRACE:
                st.info("All three rules are met. Place the limit order at the entry and let "
                        "the retrace come to you.")

            if stopped_out and plan.direction:
                st.markdown("##### Re-entry plan (after being stopped out)")
                for i_, step_ in enumerate(trend_retrace.reentry_plan_text(plan.direction), 1):
                    st.markdown(f"{i_}. {step_}")
                st.caption("Each re-entry half should risk half of your normal amount, so the "
                           "full re-entry risks the same as one normal trade.")

        score_evidence, seq_evidence = {}, {}

    elif mode == "Auto-scan one instrument":
        s1, s2, s3 = st.columns([2, 1, 1])
        with s1:
            stype = st.radio("Asset type", ["Crypto", "Stock", "Commodity"],
                              horizontal=True, key="scan_type")
            if stype == "Crypto":
                _slabel = st.selectbox("Symbol", list(CRYPTO_TICKERS), key="scan_c")
                scan_ticker = CRYPTO_TICKERS[_slabel]
                scan_cg_id = CRYPTO_CG_IDS.get(_slabel)
            elif stype == "Commodity":
                scan_ticker = COMMODITY_TICKERS[st.selectbox("Symbol", list(COMMODITY_TICKERS), key="scan_o")]
                scan_cg_id = None
            else:
                scan_ticker = st.text_input("Stock ticker", value="AAPL", key="scan_s").upper().strip()
                scan_cg_id = None
        with s2:
            dir_choice = st.selectbox("Direction", ["Auto", "Long", "Short"], key="scan_dir")
        with s3:
            min_rr = st.number_input("Min R:R", min_value=3.0, value=3.0, step=0.5, key="scan_rr")

        if st.button("🔍 Scan", use_container_width=True):
            with st.spinner(f"Analysing {scan_ticker} across 1D / 4H / 1H…"):
                use_exchange_first = (stype == "Crypto" and CRYPTO_SOURCE == "exchange")
                use_cg_first = (stype == "Crypto" and CRYPTO_PREFERS_CG and scan_cg_id)
                frames, problems = {}, []

                if use_exchange_first:
                    frames, ex_source, problems = _cached_exchange_frames(scan_ticker)
                    if ex_source:
                        problems = ([f"True OHLCV candles from {ex_source} "
                                      f"(all timeframes native, nothing resampled)."]
                                    + list(problems))
                    else:
                        # Neither exchange lists it — fall back to CoinGecko.
                        if scan_cg_id:
                            frames, cg_problems = _cached_cg_frames(scan_cg_id)
                            problems = ([f"{scan_ticker} is not listed on Binance or Kraken "
                                          f"— using CoinGecko candles instead."] + cg_problems)
                        if not frames or all(f is None or f.empty for f in frames.values()):
                            frames, y_problems = fetch_multi_timeframe(scan_ticker, yf)
                            problems = ["Exchanges and CoinGecko both failed — "
                                        "fell back to Yahoo Finance."] + y_problems
                elif use_cg_first:
                    frames, problems = _cached_cg_frames(scan_cg_id)
                    problems = ["Candles from CoinGecko (4H is native, not resampled)."] + problems
                    if all(f is None or f.empty for f in frames.values()):
                        frames, problems = fetch_multi_timeframe(scan_ticker, yf)
                        problems = ["CoinGecko returned nothing — fell back to Yahoo Finance."] + problems
                    else:
                        # Partial failure (commonly a rate-limited daily call):
                        # fill only the missing frames from Yahoo instead of
                        # discarding the good CoinGecko candles we already have.
                        missing = [k for k, f in frames.items() if f is None or f.empty]
                        if missing:
                            y_frames, _y_problems = fetch_multi_timeframe(scan_ticker, yf)
                            filled = []
                            for k in missing:
                                yf_frame = y_frames.get(k)
                                if yf_frame is not None and not yf_frame.empty:
                                    frames[k] = yf_frame
                                    filled.append(k)
                            if filled:
                                problems.append(
                                    f"Gap-filled {', '.join(filled)} from Yahoo Finance after "
                                    f"CoinGecko failed on those timeframes — this scan mixes "
                                    f"two data sources.")
                else:
                    frames, problems = fetch_multi_timeframe(scan_ticker, yf)
                    if all(f is None or f.empty for f in frames.values()) and scan_cg_id:
                        cg_frames, cg_problems = _cached_cg_frames(scan_cg_id)
                        if any(f is not None and not f.empty for f in cg_frames.values()):
                            frames = cg_frames
                            problems = ([f"Yahoo Finance has no data for {scan_ticker}; "
                                          f"using CoinGecko candles instead."] + cg_problems)

                _live = None
                if stype == "Crypto":
                    _live, _src, _err = _cached_exchange_spot(scan_ticker)
                st.session_state.analysis = analyze_candidate(
                    scan_ticker, frames,
                    direction_override=None if dir_choice == "Auto" else dir_choice,
                    min_rr=min_rr, data_problems=problems, live_price=_live)
                # Bump the scan id so the evidence checkboxes below get FRESH
                # widget keys. Streamlit ignores `value=` for a key that already
                # exists in session state, so reusing keys would freeze the
                # checkboxes (and therefore the score) at the first scan's result.
                st.session_state.scan_id = st.session_state.get("scan_id", 0) + 1

        analysis = st.session_state.get("analysis")
        if analysis is None:
            st.info("Pick an instrument and press Scan.")
        else:
            for p in analysis.data_problems:
                st.warning(f"Data gap: {p}")

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Instrument", analysis.ticker)
            h2.metric("1D regime", analysis.regime_1d.title())
            h3.metric("Direction", analysis.direction or "—")
            h4.metric("Price", format_price(analysis.current_price))
            if analysis.price_source:
                st.caption(
                    f"Entry price taken from the latest **{analysis.price_source}** close. "
                    f"If this differs from the Market tab, one of them is a slightly older bar — "
                    f"always confirm against your broker before entering."
                )

            plan = analysis.entry_plan
            if plan is not None:
                badge = {"AT_ZONE": "🟢 AT ZONE", "APPROACHING": "🟡 APPROACHING",
                         "FAR": "⚪ TOO FAR", "MISSED": "🔴 MISSED"}[plan.status]
                st.markdown(f"##### Entry plan · {badge} · order type: **{plan.order_type}**")
                z1, z2, z3 = st.columns(3)
                z1.metric("Entry zone",
                          f"{format_price(plan.zone_low)} – {format_price(plan.zone_high)}")
                z2.metric("Distance to zone", f"{plan.distance_pct:.2f}%",
                          f"{plan.distance_atr:.1f} ATR")
                z3.metric("Confluence", f"{plan.confluence} source(s)")
                st.caption(f"{plan.rationale}  \nSources: {', '.join(plan.sources)}")
                if plan.confluence == 1:
                    st.caption("⚠️ Only one source supports this level — weaker than a "
                               "zone where a swing, a Fib level and equal highs/lows agree.")
                if analysis.entry_is_planned and analysis.current_price:
                    gap = (analysis.current_price - analysis.entry) / analysis.current_price * 100
                    st.info(
                        f"Planned entry **{format_price(analysis.entry)}** vs live "
                        f"**{format_price(analysis.current_price)}** "
                        f"({abs(gap):.1f}% {'better' if (gap > 0) == (analysis.direction == 'Long') else 'worse'}). "
                        f"Stop, target and reward:risk below are all measured from the "
                        f"planned entry, not the live price.")
                if analysis.alternative_zones:
                    with st.expander(f"Other candidate zones ({len(analysis.alternative_zones)})"):
                        for z in analysis.alternative_zones:
                            st.caption(
                                f"**{format_price(z.zone_low)} – {format_price(z.zone_high)}** · "
                                f"{z.status} · {z.distance_pct:.2f}% away · "
                                f"{z.confluence} source(s): {', '.join(z.sources)}")
            st.markdown("---")

            if analysis.stop is not None and analysis.target is not None:
                l1, l2, l3 = st.columns(3)
                l1.metric("Entry", format_price(analysis.entry))
                l2.metric("Structural stop", format_price(analysis.stop))
                l3.metric("Structural target", format_price(analysis.target))
                if analysis.reward_risk is not None:
                    ok = analysis.reward_risk >= min_rr
                    (st.success if ok else st.warning)(
                        f"Reward:risk **{analysis.reward_risk:.2f}:1** — {analysis.level_reason}")
                if st.button("📋 Send these levels to the Risk tab"):
                    st.session_state.prefill = {
                        "entry": float(analysis.entry), "stop": float(analysis.stop),
                        "target": float(analysis.target),
                        "direction": analysis.direction or "Long",
                        "ticker": analysis.ticker,
                    }
                    st.success("Levels copied — open the 🧮 Risk tab.")
            else:
                st.warning("Could not derive a full entry/stop/target from confirmed structure.")

            sid = st.session_state.get("scan_id", 0)
            auto_result = score_setup(analysis.score_dict())
            st.markdown(
                f"##### Auto-derived evidence · scanner says "
                f"**{auto_result.normalized_score:.1f}/10 ({auto_result.label})**"
            )
            st.caption(
                "Ticking or unticking below overrides the scanner; the score at the "
                "bottom reflects your edits, this one does not."
            )
            for key, points, label in POSITIVE_COMPONENTS + NEGATIVE_COMPONENTS:
                item = analysis.score_evidence.get(key)
                val = item.value if item else None
                icon = {True: "✅", False: "❌", None: "❔"}[val]
                score_evidence[key] = st.checkbox(
                    f"{icon} {label} ({points:+d})", value=bool(val),
                    key=f"as_{sid}_{key}")
                if item:
                    st.caption(f"　↳ {item.reason}")

            with st.expander("Entry sequence detail"):
                for key, desc in ENTRY_SEQUENCE_STAGES:
                    item = analysis.sequence_evidence.get(key)
                    val = item.value if item else None
                    icon = {True: "✅", False: "❌", None: "❔"}[val]
                    seq_evidence[key] = st.checkbox(
                        f"{icon} {desc}", value=bool(val), key=f"aq_{sid}_{key}")
                    if item:
                        st.caption(f"　↳ {item.reason}")
    else:
        st.markdown("##### Entry sequence")
        cols = st.columns(2)
        for i, (key, desc) in enumerate(ENTRY_SEQUENCE_STAGES):
            with cols[i % 2]:
                seq_evidence[key] = st.checkbox(desc, key=f"ms_{key}")

        st.markdown("##### Confluence evidence")
        m1, m2 = st.columns(2)
        with m1:
            for key, points, label in POSITIVE_COMPONENTS:
                score_evidence[key] = st.checkbox(f"{label} (+{points})", key=f"mp_{key}")
        with m2:
            for key, points, label in NEGATIVE_COMPONENTS:
                score_evidence[key] = st.checkbox(f"{label} ({points})", key=f"mn_{key}")

    # The confluence score/readiness panel only applies to the original
    # strategy's single-coin and manual modes. Under Trend Retrace, or after a
    # universe scan, it would show a meaningless score built from no evidence.
    if not USE_TR and not USE_SWING and not USE_MR and mode != "Rank the universe":
        st.markdown("---")
        result = score_setup(score_evidence)
        r1, r2 = st.columns([1, 2])
        r1.metric("Setup score", f"{result.normalized_score:.1f}/10", result.label)
        r2.info(trade_policy_for_grade(result.label))

        if result.label != "A+":
            missing = weakest_components(result)
            if missing:
                st.caption("Weakest: " + "; ".join(
                    f"{li.label} (+{li.points_possible} unclaimed)" for li in missing))

        with st.expander("📊 Score breakdown — where every point went"):
            earned = sum(li.points_awarded for li in result.breakdown if li.points_awarded > 0)
            available = sum(li.points_possible for li in result.breakdown if li.points_possible > 0)
            st.caption(f"Earned **{earned}** of **{available}** positive points.")
            rows = []
            for li in result.breakdown:
                rows.append({
                    "Component": li.label,
                    "Worth": f"{li.points_possible:+d}",
                    "Earned": f"{li.points_awarded:+d}",
                    "Evidence": {True: "yes", False: "no", None: "could not evaluate"}[li.evidence],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "The two-point items (regime alignment, liquidity sweep + reclaim) are the "
                "difference between a C and an A+. If they never fire, 4–5/10 is the "
                "structural ceiling — that is the scoring working as designed, not a fault, "
                "but it is worth reading their reasons above to see whether you agree."
            )

        st.markdown("##### Readiness")
        q = st.session_state.get("quote")
        data_ok = q is not None and q.status in (DataStatus.LIVE, DataStatus.DELAYED)
        risk_ok = st.checkbox("Risk checks pass (sized, within budget)", key="rdy_risk")
        invalidated = st.checkbox("Structural invalidation has occurred", key="rdy_inval")
        entered = st.checkbox("I have manually entered this trade", key="rdy_active")

        readiness = determine_readiness(ReadinessInputs(
            data_is_valid=data_ok, invalidation_hit=invalidated, user_marked_active=entered,
            near_actionable_location=seq_evidence.get("location", False),
            confirmation_triggered=seq_evidence.get("confirmation", False),
            invalidation_defined=seq_evidence.get("structural_invalidation", False),
            risk_checks_pass=risk_ok))

        st.metric("Readiness", f"{display_label(readiness)} ({readiness.value})")
        st.caption(READINESS_DESCRIPTIONS[readiness])

        if not data_ok:
            st.info("Readiness shows DATA ERROR because no usable quote is loaded. "
                    "Fetch the instrument on the 🌍 Market tab first.")

        if readiness in (Readiness.CONDITIONAL, Readiness.NOT_READY):
            gaps = missing_entry_sequence_stages(seq_evidence)
            if gaps:
                st.warning("What would confirm this:\n" + "\n".join(f"- {g}" for g in gaps))

with tab_review:
    st.subheader("Review")
    st.caption(
        "Two things live here: setups the scanner saved earlier, so you can decide whether "
        "they're still worth leaving pending, and a one-off review of any setup you type in."
    )
    _rv_tab1, _rv_tab2, _rv_tab3 = st.tabs(["Saved setups", "Review any setup",
                                             "Where's my stop?"])

    with _rv_tab3:
      if stop_manager is None:
        st.warning("This needs `stop_manager.py`, which isn't in your repo yet.")
      else:
        st.caption(
            "Tell it where you actually got filled and it works out the stop from the "
            "chart's structure — then updates it as the trade moves. It only ever moves "
            "the stop closer to price, never further away."
        )
        sm1, sm2 = st.columns([3, 1])
        _sm_lists = {"Crypto": CRYPTO_TICKERS, "Commodities": COMMODITY_TICKERS,
                     "FX": FX_TICKERS}
        _sm_market = sm1.selectbox("Market", list(_sm_lists), key="sm_market")
        _sm_name = sm1.selectbox("Instrument", list(_sm_lists[_sm_market]), key="sm_name")
        _sm_ticker = _sm_lists[_sm_market][_sm_name]
        _sm_dir = sm2.selectbox("Direction", ["Long", "Short"], key="sm_dir")

        n1, n2 = st.columns(2)
        _sm_entry = n1.number_input("I entered at", min_value=0.0, format="%.8f",
                                    key="sm_entry")
        _sm_stop = n2.number_input("My stop now (0 if none yet)", min_value=0.0,
                                   format="%.8f", key="sm_stop")
        _sm_tf = st.selectbox("Read structure from", ["5m", "1h", "4h", "1d"],
                              index=3 if USE_SWING else 0, key="sm_tf",
                              help="Use the timeframe you're trading: 5m for intraday, "
                                   "daily for swing trades.")

        if st.button("📏 Work out my stop", use_container_width=True, key="sm_go"):
            if not _sm_entry:
                st.error("Enter the price you got filled at.")
            else:
                with st.spinner("Reading the chart…"):
                    _df = _candles_for(_sm_ticker, _sm_tf, 400)
                if _df is None or _df.empty:
                    st.error("Couldn't load candles for that instrument and timeframe.")
                else:
                    _live = float(_df["Close"].iloc[-1])
                    _atr = trend_retrace.atr(_df)
                    try:
                        st.session_state.sm_result = (stop_manager.suggest(
                            _sm_dir, entry=_sm_entry, price=_live, candles=_df, atr=_atr,
                            current_stop=_sm_stop or None), _live)
                    except ValueError as e:
                        st.error(str(e))

        _smr = st.session_state.get("sm_result")
        if _smr:
            _s, _live = _smr
            with st.container(border=True):
                if _s.price is None:
                    st.warning(_s.reason)
                else:
                    (st.success if _s.is_tighter else st.info)(
                        f"**{_s.basis}** → {format_price(_s.price)}")
                    st.write(_s.reason)
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Suggested stop", format_price(_s.price))
                    k2.metric("Price now", format_price(_live))
                    if _s.current_r is not None:
                        k3.metric("Trade is at", f"{_s.current_r:+.2f}R")
                    if _s.locked_in is not None:
                        _word = ("locks in" if _s.locked_in > 0 else "risks")
                        st.caption(f"If it triggers, that {_word} "
                                   f"{format_price(abs(_s.locked_in))} per unit "
                                   f"(risk from entry: {format_price(_s.risk_per_unit)}).")
                    if _s.alternatives:
                        st.caption("Also possible: " + " · ".join(_s.alternatives))
                    if not _s.is_tighter:
                        st.caption("This isn't tighter than your current stop, so there's "
                                   "nothing to change.")
                st.caption("Advice only — move the stop on the exchange yourself. Never "
                           "widen it: if the reason for the trade has gone, close it "
                           "instead.")



    with _rv_tab1:
        st.caption("Every setup the scanner produced, with its levels frozen at the time. A "
                   "plan made yesterday describes a chart that no longer exists, so this "
                   "re-checks each one against the current trend and price, and lets you "
                   "cancel the ones that have gone stale.")
        _sig_df = storage.get_signals_df()
        _pending = (_sig_df[_sig_df["status"] == "PENDING"] if not _sig_df.empty else _sig_df)
        if _pending.empty:
            st.caption("No setups waiting to fill.")
        elif st.button(f"Re-check {len(_pending)} waiting setup(s)", use_container_width=True):
            _out = []
            bar = st.progress(0.0, text="Checking…")
            for i_, (_, sg) in enumerate(_pending.iterrows()):
                bar.progress(i_ / max(len(_pending), 1), text=sg["ticker"])
                _created = pd.Timestamp(sg["created_utc"])
                _created = _created.tz_localize("UTC") if _created.tzinfo is None else _created
                try:
                    _fr, _since, _px, _now = _review_inputs(sg["ticker"], _created)
                    if _px is None:
                        _out.append((sg, None, "No current price available."))
                        continue
                    _v = trade_review.setup_still_viable(
                        sg["direction"], float(sg["entry"]), float(sg["stop"]), _created,
                        float(_px), _now, _fr["4h"],
                        trend_candles=(TR_PARAMS.trend_candles if TR_PARAMS else 2))
                    _out.append((sg, _v, None))
                except Exception as e:
                    _out.append((sg, None, f"{type(e).__name__}: {e}"))
            bar.empty()
            st.session_state.viability = _out

        for sg, _v, _err in st.session_state.get("viability", []):
            with st.container(border=True):
                _age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(sg["created_utc"]).tz_localize(
                    "UTC") if pd.Timestamp(sg["created_utc"]).tzinfo is None
                    else pd.Timestamp.now(tz="UTC") - pd.Timestamp(sg["created_utc"]))
                st.markdown(f"**{sg['direction']} {sg['label'] or sg['ticker']}** "
                            f"· {sg.get('grade') or '—'} · saved "
                            f"{_age.total_seconds() / 3600:.0f}h ago")
                st.caption(f"Entry {format_price(sg['entry'])} · stop {format_price(sg['stop'])} "
                           f"· target {format_price(sg['target'])}")
                if _err or _v is None:
                    st.caption(f"Couldn't check: {_err}")
                    continue
                box = {trade_review.STILL_VALID: st.success}.get(_v.verdict, st.warning)
                box(f"**{_v.verdict}** · now {format_price(_v.price)}")
                for why in _v.reasons:
                    st.caption(f"• {why}")
                if _v.verdict != trade_review.STILL_VALID:
                    if st.button("Cancel this setup", key=f"vcancel_{int(sg['id'])}"):
                        storage.update_signal(int(sg["id"]), status="EXPIRED",
                                              exit_utc=datetime.now(timezone.utc).isoformat())
                        st.session_state.viability = [
                            x for x in st.session_state.viability if x[0]["id"] != sg["id"]]
                        st.rerun()


    with _rv_tab2:
        st.caption(
            "A second opinion on any setup — one you're already in, or one you're thinking "
            "about. Enter the levels and it re-checks them against the current chart: is "
            "the trend still there, has price passed the entry, and what would closing now "
            "cost against letting the stop do its job."
        )

        rv1, rv2 = st.columns([3, 1])
        _rv_market = rv1.selectbox("Market", ["Crypto", "Commodities", "FX"], key="rv_market")
        _rv_lists = {"Crypto": CRYPTO_TICKERS, "Commodities": COMMODITY_TICKERS,
                     "FX": FX_TICKERS}
        _rv_name = rv1.selectbox("Instrument", list(_rv_lists[_rv_market]), key="rv_name")
        _rv_ticker = _rv_lists[_rv_market][_rv_name]
        _rv_dir = rv2.selectbox("Direction", ["Long", "Short"], key="rv_dir")

        # Prefill from the last single-coin check, if it was this instrument.
        _last = st.session_state.get("tr_plan") or st.session_state.get("sw_plan")
        _pre = {}
        if _last is not None and getattr(_last, "ticker", None) == _rv_ticker:
            _pre = {"entry": _last.entry, "stop": _last.stop, "target": _last.target}
            st.caption("Levels pre-filled from your last check of this instrument.")

        q1, q2, q3 = st.columns(3)
        _rv_entry = q1.number_input("Entry", min_value=0.0, format="%.8f",
                                    value=float(_pre.get("entry") or 0.0), key="rv_entry")
        _rv_stop = q2.number_input("Stop", min_value=0.0, format="%.8f",
                                   value=float(_pre.get("stop") or 0.0), key="rv_stop")
        _rv_target = q3.number_input("Target", min_value=0.0, format="%.8f",
                                     value=float(_pre.get("target") or 0.0), key="rv_target")
        _rv_in = st.checkbox("I'm already in this trade", key="rv_in")
        _rv_hours = st.number_input("Hours since I entered", min_value=0.0, value=6.0,
                                    step=1.0, key="rv_hours",
                                    disabled=not _rv_in) if _rv_in else 0.0

        if st.button("🔍 Review it", use_container_width=True, key="rv_run"):
            if not (_rv_entry and _rv_stop):
                st.error("Enter at least an entry and a stop.")
            else:
                with st.spinner("Loading the current chart…"):
                    _since_when = (pd.Timestamp.now(tz="UTC")
                                   - pd.Timedelta(hours=float(_rv_hours or 1)))
                    _fr, _bars_since, _px, _now = _review_inputs(_rv_ticker, _since_when)
                if _px is None:
                    st.error("Couldn't fetch a current price for this instrument.")
                elif _rv_in:
                    st.session_state.rv_result = ("open", trade_review.review(
                        direction=_rv_dir, entry=_rv_entry, stop=_rv_stop,
                        target=_rv_target or _rv_entry, entry_time=_since_when,
                        price=float(_px), now=_now, df_4h=_fr["4h"], df_1h=_fr["1h"],
                        candles_since_entry=_bars_since,
                        structure_candles=_fr["5m"].tail(200),
                        atr_value=trend_retrace.atr(_fr["5m"]),
                        trend_candles=(TR_PARAMS.trend_candles if TR_PARAMS else 2),
                        stale_hours=st.session_state.get("trb_stale", 12.0)), float(_px))
                else:
                    st.session_state.rv_result = ("planned", trade_review.setup_still_viable(
                        direction=_rv_dir, entry=_rv_entry, stop=_rv_stop,
                        created=_since_when, price=float(_px), now=_now, df_4h=_fr["4h"],
                        trend_candles=(TR_PARAMS.trend_candles if TR_PARAMS else 2)),
                        float(_px))

        _res = st.session_state.get("rv_result")
        if _res:
            _kind, _r, _price_now = _res
            with st.container(border=True):
                if _kind == "open":
                    box = {trade_review.HOLD: st.success, trade_review.TIGHTEN: st.info,
                           trade_review.CLOSE: st.warning,
                           trade_review.CLOSE_THESIS: st.error}[_r.verdict]
                    box(f"**{_r.verdict}** · price now {format_price(_price_now)}")
                    for why in _r.reasons:
                        st.caption(f"• {why}")
                    for act in _r.actions:
                        st.markdown(act)
                    if _r.current_r is not None:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Close now", f"{_r.close_now_r:+.2f}R")
                        m2.metric("If stop hits", f"{_r.stop_r:+.2f}R")
                        m3.metric("If target hits", f"{_r.target_r:+.2f}R"
                                  if _r.target_r else "—")
                    if _r.suggested_stop is not None:
                        st.caption(f"Suggested tighter stop: "
                                   f"**{format_price(_r.suggested_stop)}** — never wider.")
                else:
                    ok = _r.verdict == trade_review.STILL_VALID
                    (st.success if ok else st.warning)(
                        f"**{_r.verdict}** · price now {format_price(_price_now)}")
                    for why in _r.reasons:
                        st.caption(f"• {why}")
                st.caption("A second opinion based on your rules, not an instruction. "
                           "Nothing is changed on the exchange.")




# ---------------------------------------------------------------------
# TAB: Positions
# ---------------------------------------------------------------------
with tab_journal:
    st.subheader("Trade journal")

    # --- trades taken from the scanner, waiting to be completed ----------
    _open = trade_log.open_scanner_trades()
    st.markdown(f"##### Trades to complete ({len(_open)})")
    if _open.empty:
        st.caption("Trades you mark **I took this trade** on the 🎯 Scanner appear here. "
                   "When one closes, complete it so the system can learn from it.")
    for _, jr in _open.iterrows():
        jid = int(jr["id"])
        with st.container(border=True):
            st.markdown(f"**#{jid} · {jr['direction']} {jr['asset']}** — entry "
                        f"{format_price(jr['entry'])}, stop {format_price(jr['sl'])}, "
                        f"target {format_price(jr['tp'])}")
            outcome = st.radio("How did it end?", trade_log.OUTCOMES, horizontal=True,
                               key=f"cm_out_{jid}")
            _default = trade_log.default_exit_price(outcome, float(jr["sl"]), float(jr["tp"]),
                                                    manual=float(jr["entry"]))
            cc1, cc2 = st.columns(2)
            exit_px = cc1.number_input("Exit price achieved", value=float(_default),
                                       format="%.8f", key=f"cm_exit_{jid}_{outcome}")
            fill_px = cc2.number_input("Actual entry (correct if needed)",
                                       value=float(jr["entry"]), format="%.8f",
                                       key=f"cm_fill_{jid}")
            cc3, cc4 = st.columns(2)
            fin_sl = cc3.number_input("Final stop loss (if you moved it)",
                                      value=float(jr["sl"]), format="%.8f",
                                      key=f"cm_sl_{jid}")
            fin_tp = cc4.number_input("Final take profit (if you moved it)",
                                      value=float(jr["tp"]), format="%.8f",
                                      key=f"cm_tp_{jid}")
            fees = st.number_input("Fees paid ($, optional)", min_value=0.0, value=0.0,
                                   key=f"cm_fee_{jid}")
            _init = jr.get("initial_stop")
            _init = float(jr["sl"]) if _init is None or pd.isna(_init) else float(_init)
            _r = trade_log.realized_r(jr["direction"], fill_px, _init, exit_px)
            if _r is not None:
                st.caption(f"Result: **{_r:+.2f}R** — {trade_log.pl_status_for(_r).lower()}, "
                           f"measured against your original stop of {format_price(_init)}.")
            if st.button("🔍 Review this trade", key=f"rv_go_{jid}", use_container_width=True):
                with st.spinner("Reviewing…"):
                    _et = pd.Timestamp(jr["timestamp_utc"])
                    _et = _et.tz_localize("UTC") if _et.tzinfo is None else _et
                    _fr, _since, _px, _now = _review_inputs(jr["asset"], _et)
                    if _px is None:
                        st.error("Couldn't fetch a current price for this coin.")
                    else:
                        st.session_state[f"rv_{jid}"] = (trade_review.review(
                            direction=jr["direction"], entry=float(jr["entry"]),
                            stop=float(jr["sl"]), target=float(jr["tp"]), entry_time=_et,
                            price=float(_px), now=_now, df_4h=_fr["4h"], df_1h=_fr["1h"],
                            candles_since_entry=_since,
                            structure_candles=_fr["5m"].tail(200),
                            atr_value=trend_retrace.atr(_fr["5m"]),
                            trend_candles=(TR_PARAMS.trend_candles if TR_PARAMS else 2),
                            stale_hours=st.session_state.get("trb_stale", 12.0)), float(_px))

            _rv = st.session_state.get(f"rv_{jid}")
            if _rv is not None:
                rv, rv_px = _rv
                box = {trade_review.HOLD: st.success, trade_review.TIGHTEN: st.info,
                       trade_review.CLOSE: st.warning,
                       trade_review.CLOSE_THESIS: st.error}[rv.verdict]
                box(f"**Review: {rv.verdict}**")
                for why in rv.reasons:
                    st.caption(f"• {why}")
                for act in rv.actions:
                    st.markdown(act)
                if rv.current_r is not None:
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Close now", f"{rv.close_now_r:+.2f}R")
                    rc2.metric("If stop hits", f"{rv.stop_r:+.2f}R")
                    rc3.metric("If target hits", f"{rv.target_r:+.2f}R")
                if rv.suggested_stop is not None:
                    widening = (rv.suggested_stop < float(jr["sl"]) if jr["direction"] == "Long"
                                else rv.suggested_stop > float(jr["sl"]))
                    if not widening and st.button(
                            f"Move stop to {format_price(rv.suggested_stop)}",
                            key=f"rv_sl_{jid}"):
                        storage.update_journal_entry(jid, sl=float(rv.suggested_stop),
                                                     updated_sl=float(rv.suggested_stop))
                        st.success("Stop updated in the journal — remember to move it on "
                                   "Hyperliquid too. Results still count against your "
                                   "original stop.")
                        st.session_state.pop(f"rv_{jid}", None)
                        st.rerun()
                if rv.verdict in (trade_review.CLOSE, trade_review.CLOSE_THESIS):
                    if st.button(f"Close now at {format_price(rv_px)} ({rv.close_now_r:+.2f}R)",
                                 key=f"rv_close_{jid}"):
                        trade_log.complete_trade(jid, exit_price=rv_px,
                                                 exit_reason=f"Closed early after review: "
                                                             f"{rv.verdict}")
                        st.success("Completed in the journal — remember to close it on "
                                   "Hyperliquid too.")
                        st.session_state.pop(f"rv_{jid}", None)
                        st.rerun()
                st.caption("The review is a second opinion based on your rules, not an "
                           "instruction. Nothing is changed on the exchange.")

            if st.button("✅ Complete trade", key=f"cm_go_{jid}", use_container_width=True):
                trade_log.complete_trade(jid, exit_price=exit_px, actual_entry=fill_px,
                                         final_stop=fin_sl, final_target=fin_tp, fees=fees,
                                         exit_reason=outcome)
                st.success(f"#{jid} completed at {_r:+.2f}R.")
                st.rerun()

    # --- learning from the trades you've actually taken -------------------
    st.markdown("##### 🧠 Learn from your trades")
    _include_disc = st.checkbox(
        "Include trades where I didn't follow the rules", key="lt_disc",
        help="Off by default: those trades test your judgement, not the strategy.")
    _mine = trade_log.learning_trades(config=_config_signature(USE_TR, TR_PARAMS),
                                      followed_only=not _include_disc)
    if _mine:
        _wins = sum(t.status == "WIN" for t in _mine)
        _avg = sum(t.r_result for t in _mine) / len(_mine)
        m1, m2, m3 = st.columns(3)
        m1.metric("Completed trades", len(_mine))
        m2.metric("Win rate", f"{_wins / len(_mine):.0%}")
        m3.metric("Average result", f"{_avg:+.2f}R")
    st.caption(
        f"{len(_mine)} completed trade(s) under your current settings; about "
        f"{learning.MIN_TRADES} are needed before lessons are trustworthy. Your real trades "
        f"are the most valuable evidence there is — they include your actual fills and exits.")
    if st.button("🧠 Learn from my trades", use_container_width=True,
                 disabled=len(_mine) < learning.MIN_TRADES, key="lt_go"):
        _m = learning.learn(_mine, source=f"your real trades, {len(_mine)} trades")
        st.session_state.mine_model = _m
        _md = _m.to_dict()
        _md["config"] = _config_signature(USE_TR, TR_PARAMS)
        storage.save_value("learned_model", _md)
    if st.session_state.get("mine_model") is not None:
        _render_learned(st.session_state.mine_model)

    st.markdown("---")
    st.markdown("##### Add a trade manually")

    with st.form("journal_add"):
        j1, j2, j3, j4 = st.columns(4)
        ja = j1.text_input("Asset")
        jd = j2.selectbox("Direction", ["Long", "Short"])
        jl = j3.number_input("Leverage", min_value=1.0, value=1.0)
        je = j4.number_input("Entry", min_value=0.0)
        j5, j6, j7 = st.columns(3)
        jtp = j5.number_input("TP", min_value=0.0)
        jsl = j6.number_input("SL", min_value=0.0)
        jsz = j7.number_input("Size (notional USD)", min_value=0.0)
        jr = st.text_area("Entry reason / score breakdown")
        if st.form_submit_button("Add entry") and ja:
            storage.add_journal_entry({
                "asset": ja, "direction": jd, "leverage": jl, "entry": je,
                "tp": jtp, "sl": jsl, "size_notional_usd": jsz, "entry_reason": jr})
            st.success("Added.")
            st.rerun()

    _jup = st.file_uploader("Restore journal from an exported CSV", type=["csv"],
                            key="journal_upload")
    if _jup is not None and st.button("Import journal", key="journal_import"):
        try:
            _n = storage.import_journal_csv(_jup.getvalue())
            st.success(f"Restored {_n} journal entr{'y' if _n == 1 else 'ies'}.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    df = storage.get_journal_df()
    if df.empty:
        st.info("No journal entries yet.")
    else:
        _hide = ["features", "strategy_config", "margin", "current_pl",
                 "potential_profit_at_tp", "potential_loss_at_sl", "management_notes"]
        _view = df.drop(columns=[c for c in _hide if c in df.columns])
        _first = [c for c in ("id", "status", "asset", "direction", "pl_status",
                              "realized_r", "entry", "sl", "tp", "exit_price",
                              "realized_pnl") if c in _view.columns]
        _view = _view[_first + [c for c in _view.columns if c not in _first]]
        st.dataframe(_view, use_container_width=True, hide_index=True,
                     column_config={"realized_r": st.column_config.NumberColumn(
                         "Result (R)", format="%+.2f")})
        st.caption("The full export below includes every column, including the setup "
                   "snapshots used for learning.")
        st.warning("**Export your journal before every app update.** Streamlit wipes the "
                   "database on redeploy — your trades, and everything the system has "
                   "learned from them, would otherwise be lost. Restore with the import "
                   "box below.")
        st.download_button("⬇️ Export CSV", data=storage.journal_to_csv_bytes(),
                            file_name=f"trade_journal_{datetime.now().date()}.csv",
                            mime="text/csv", use_container_width=True)

        open_rows = df[df["status"] == "Open"]
        if not open_rows.empty:
            st.markdown("##### Close a trade")
            cid = st.selectbox("Entry", open_rows["id"].tolist(),
                                format_func=lambda i: f"#{i} {df[df['id']==i]['asset'].iloc[0]}")
            cpnl = st.number_input("Realized P&L", step=0.01)
            creason = st.text_input("Exit reason")
            if st.button("Close trade"):
                storage.close_journal_entry(int(cid), cpnl, creason)
                st.rerun()

        with st.expander("⚠️ Danger zone"):
            if st.checkbox("I understand this permanently deletes all journal entries and positions"):
                if st.button("Delete everything"):
                    storage.clear_all()
                    st.rerun()

# ---------------------------------------------------------------------
# TAB: Backtest
# ---------------------------------------------------------------------
with tab_track:
    st.subheader("Forward tracking")
    st.caption(
        "Every B-or-better setup from a universe scan is recorded with its plan frozen "
        "at that moment. Outcomes are then checked against what price actually did. "
        "Because the plan is fixed before the result is known, this is the most honest "
        "test the strategy can get."
    )
    st.info(
        "Works even while the app sleeps — pressing Update fetches every candle printed "
        "since each signal was created. But the database is **wiped on every redeploy**, "
        "so export the CSV regularly and re-import it after updating the app."
    )

    sig_df = storage.get_signals_df()
    open_count = int(sig_df["status"].isin(["PENDING", "FILLED"]).sum()) if not sig_df.empty else 0

    t1, t2 = st.columns(2)
    t1.metric("Signals tracked", len(sig_df))
    t2.metric("Still open", open_count)

    if st.button("🔄 Update outcomes", use_container_width=True, disabled=open_count == 0):
        bar = st.progress(0.0, text="Checking…")

        def _bars(ticker):
            if hyperliquid_data.is_hl_ticker(ticker):
                df, _err = hyperliquid_data.fetch_candles(ticker, "1h", 1000)
                return df
            df, _err = exchanges.fetch_binance_klines(ticker, "1h", limit=1000)
            if df is None:
                df, _err = exchanges.fetch_kraken_ohlc(ticker, "1h")
            return df

        counts = tracking.update_all(
            _bars, bar_hours=1.0,
            progress=lambda i, n, t: bar.progress(min(i / max(n, 1), 1.0),
                                                   text=f"{i}/{n} · {t}"))
        bar.empty()
        st.success(f"Checked {counts['checked']} · resolved {counts['resolved']}"
                   + (f" · {counts['failed']} could not be fetched" if counts["failed"] else ""))
        st.rerun()

    if sig_df.empty:
        st.info("No signals yet. Run a universe scan on the 🎯 Scanner tab with tracking "
                "switched on, then come back after a few days.")
    else:
        stats = tracking.tracked_stats()
        st.markdown("##### Live track record by grade")
        st.dataframe(pd.DataFrame([{
            "Grade": s_.grade, "Signals": s_.signals, "Filled": s_.filled,
            "Wins": s_.wins, "Losses": s_.losses, "Expired": s_.expired,
            "Open": s_.still_open,
            "Win rate": f"{s_.win_rate:.0%}" if s_.win_rate is not None else "—",
            "Avg R / trade": f"{s_.avg_r:+.2f}" if s_.avg_r is not None else "—",
            "Total R": f"{s_.total_r:+.1f}",
            "Enough data?": "yes" if s_.enough_data else "no (<30)",
        } for s_ in stats]), use_container_width=True, hide_index=True)
        resolved_total = sum(1 for _, r in sig_df.iterrows() if r["status"] in ("WIN", "LOSS"))
        if resolved_total < 30:
            st.caption(
                f"Only {resolved_total} resolved trade(s) so far. Under about 30 per grade, "
                f"results are dominated by luck — a few lucky wins can make any grade look "
                f"brilliant. Give it a few weeks.")

        show = sig_df.copy()
        for col in ("entry", "stop", "target"):
            show[col] = show[col].apply(format_price)
        st.dataframe(show[["created_utc", "label", "direction", "grade", "score",
                           "entry", "stop", "target", "status", "r_result"]],
                     use_container_width=True, hide_index=True)

    st.markdown("##### 🧠 Learn from the app's own trades")
    _tt = tracking.tracked_trades()
    _usable = [t for t in _tt if t.status in ("WIN", "LOSS") and t.features]
    st.caption(
        f"{len(_usable)} resolved trade(s) with a setup snapshot so far; about "
        f"{learning.MIN_TRADES} are needed. These are the strongest evidence available — each "
        f"plan was locked in before its result was known.")
    if st.button("🧠 Learn from tracked trades", use_container_width=True,
                 disabled=len(_usable) < learning.MIN_TRADES):
        _m = learning.learn(_tt, source=f"live tracked trades, {len(_usable)} trades")
        st.session_state.track_model = _m
        _md = _m.to_dict()
        _md["config"] = _config_signature(USE_TR, TR_PARAMS)
        storage.save_value("learned_model", _md)
    if st.session_state.get("track_model") is not None:
        _render_learned(st.session_state.track_model)
    if len(_usable) < learning.MIN_TRADES:
        st.caption("Keep scanning with tracking switched on; this unlocks once enough of the "
                   "app's own trades have resolved.")

    e1, e2 = st.columns(2)
    e1.download_button("⬇️ Export tracking CSV", data=storage.signals_to_csv_bytes(),
                       file_name=f"tracked_signals_{datetime.now().date()}.csv",
                       mime="text/csv", use_container_width=True)
    upload = e2.file_uploader("Restore from CSV", type=["csv"], key="sig_upload",
                               label_visibility="collapsed")
    if upload is not None and st.button("Import CSV"):
        try:
            added = storage.import_signals_csv(upload.getvalue())
            st.success(f"Restored {added} signal(s).")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
