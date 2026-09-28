"""
LS-AS Live Asset Screener  —  Streamlit app
Weekly screen (liquidity, ATR expansion, proximity to 4H/Daily liquidity) plus
live prices and trade levels (entry, stop, TP1, TP2) from the LS-SS 15M rules.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ticker, name, market, sub-class, suitability (1-3), $-volume multiplier, point value
#   multiplier: None = no real volume (forex), 0 = volume already in USD (crypto)
UNIVERSE = [
    ("ES=F",     "E-mini S&P 500",        "Stocks",      "Index futures", 3.0, 50,   50),
    ("NQ=F",     "E-mini Nasdaq 100",     "Stocks",      "Index futures", 3.0, 20,   20),
    ("SPY",      "S&P 500 ETF",           "Stocks",      "ETF",           3.0, 1,    1),
    ("QQQ",      "Nasdaq 100 ETF",        "Stocks",      "ETF",           3.0, 1,    1),
    ("VT",       "Total World Stock ETF", "Stocks",      "ETF",           3.0, 1,    1),
    ("VOO",      "Vanguard S&P 500 ETF",  "Stocks",      "ETF",           3.0, 1,    1),
    ("NVDA",     "NVIDIA",                "Stocks",      "Mega-cap",      2.0, 1,    1),
    ("AAPL",     "Apple",                 "Stocks",      "Mega-cap",      2.0, 1,    1),
    ("MSFT",     "Microsoft",             "Stocks",      "Mega-cap",      2.0, 1,    1),
    ("AMZN",     "Amazon",                "Stocks",      "Mega-cap",      2.0, 1,    1),
    ("GOOGL",    "Alphabet",              "Stocks",      "Mega-cap",      2.0, 1,    1),
    ("TSLA",     "Tesla",                 "Stocks",      "Mega-cap",      2.0, 1,    1),
    ("BTC-USD",  "Bitcoin",               "Crypto",      "Large-cap",     1.5, 0,    1),
    ("ETH-USD",  "Ethereum",              "Crypto",      "Large-cap",     1.5, 0,    1),
    ("GC=F",     "Gold",                  "Commodities", "Metal",         2.5, 100,  100),
    ("CL=F",     "WTI Crude Oil",         "Commodities", "Energy",        2.5, 1000, 1000),
    ("BZ=F",     "Brent Crude Oil",       "Commodities", "Energy",        2.5, 1000, 1000),
    ("EURUSD=X", "EUR/USD",               "Forex",       "Major",         3.0, None, 1),
    ("GBPUSD=X", "GBP/USD",               "Forex",       "Major",         3.0, None, 1),
    ("JPY=X",    "USD/JPY",               "Forex",       "Major",         3.0, None, 1),
    ("AUDUSD=X", "AUD/USD",               "Forex",       "Major",         3.0, None, 1),
    ("CAD=X",    "USD/CAD",               "Forex",       "Major",         3.0, None, 1),
    ("CHF=X",    "USD/CHF",               "Forex",       "Major",         3.0, None, 1),
]
INFO = {u[0]: u for u in UNIVERSE}
MARKETS = ["All", "Stocks", "Crypto", "Commodities", "Forex"]
USD_BASE = {"JPY=X", "CAD=X", "CHF=X"}         # P&L in quote currency → convert
K15, SCAN_BARS, SWEEP_EXPIRY, FVG_WAIT, ORDER_EXPIRY = 3, 96, 32, 3, 16
MIN_RR = 2.5
ACTIVE = ("Order working", "In trade", "In trade – TP1 hit, stop at BE")


# ─────────────────────────── data ───────────────────────────
def _download(tickers, period, interval):
    try:
        data = yf.download(list(tickers), period=period, interval=interval, group_by="ticker",
                           auto_adjust=False, progress=False, threads=True)
    except Exception:
        return {}
    out = {}
    if data is None or data.empty:
        return out
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        if len(df):
            out[t] = df
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(tickers):
    return _download(tickers, "1y", "1d")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_h1(tickers):
    return _download(tickers, "60d", "1h")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_m15(tickers):
    return _download(tickers, "30d", "15m")


@st.cache_data(ttl=25, show_spinner=False)
def fetch_live(tickers):
    raw = _download(tickers, "1d", "1m")
    return {t: float(df["Close"].iloc[-1]) for t, df in raw.items()}


# ─────────────────────────── helpers ───────────────────────────
def fmt(v, t=None):
    if v is None or pd.isna(v):
        return "–"
    if t in INFO and INFO[t][2] == "Forex":
        return f"{v:,.3f}" if v > 20 else f"{v:,.5f}"
    return f"{v:,.2f}" if abs(v) >= 10 else f"{v:,.4f}"


def wilder_atr(df, n=14):
    prev_c = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def to_4h(h1):
    if h1 is None or h1.empty:
        return None
    return (h1.resample("4h")
              .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
              .dropna(subset=["Close"]))


def last_pivots(df, k):
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    ph = pl = np.nan
    for i in range(len(df) - k - 1, k - 1, -1):
        if np.isnan(ph) and h[i] == h[i - k:i + k + 1].max():
            ph = h[i]
        if np.isnan(pl) and l[i] == l[i - k:i + k + 1].min():
            pl = l[i]
        if not (np.isnan(ph) or np.isnan(pl)):
            break
    return ph, pl


def pivot_flags(h, k):
    n = len(h)
    f = np.zeros(n, dtype=bool)
    for j in range(k, n - k):
        f[j] = h[j] == h[j - k:j + k + 1].max()
    return f


def nearest_unmitigated(h, ph, i, p):
    """Nearest confirmed swing high above p that price hasn't traded through as of bar i."""
    best = None
    for j in range(i - K15, max(-1, i - 400), -1):
        if ph[j] and h[j] > p and h[j] > h[j + 1:i + 1].max():
            if best is None or h[j] < best:
                best = h[j]
    return best


def stop_buffer(t, price):
    if INFO[t][2] == "Forex":
        return 0.03 if price > 20 else 0.0003          # 3 pips
    if t in ("ES=F", "NQ=F", "GC=F"):
        return 3.0                                      # 3 points
    if t in ("CL=F", "BZ=F"):
        return 0.05
    return price * 0.0005                               # ~0.05% for stocks / crypto


def reward_risk(entry, sl, tp1, tp2):
    risk = abs(entry - sl)
    if risk <= 0 or tp2 is None:
        return None
    r2 = abs(tp2 - entry) / risk
    return 0.5 * abs(tp1 - entry) / risk + 0.5 * r2 if tp1 is not None else r2


def position_size(t, risk_cash, entry, sl):
    dist = abs(entry - sl)
    if dist <= 0:
        return "–"
    _, _, market, sub, _, _, pv = INFO[t]
    if market == "Forex":
        units = risk_cash * entry / dist if t in USD_BASE else risk_cash / dist
        return f"{units:,.0f} units ({units / 100000:.2f} lots)"
    q = risk_cash / (dist * pv)
    if sub == "Index futures" or market == "Commodities":
        return f"{int(q)} contract(s)" if q >= 1 else f"{int(q * 10)} micro contract(s)"
    if market == "Crypto":
        return f"{q:,.4f} coins"
    return f"{int(q):,} shares"


# ─────────────────────────── weekly screen ───────────────────────────
def liquidity_levels(daily, h4, today):
    done = daily[daily.index < today]
    lv = {}
    if len(done):
        lv["Prev day high"] = done["High"].iloc[-1]
        lv["Prev day low"] = done["Low"].iloc[-1]
        ref = today if today.weekday() < 5 else today + pd.Timedelta(days=7 - today.weekday())
        week_start = ref - pd.Timedelta(days=ref.weekday())
        wk = done.resample("W-SUN").agg({"High": "max", "Low": "min"}).dropna()
        wk = wk[wk.index < week_start]
        if len(wk):
            lv["Prev week high"] = wk["High"].iloc[-1]
            lv["Prev week low"] = wk["Low"].iloc[-1]
        if len(done) > 10:
            lv["Daily swing high"], lv["Daily swing low"] = last_pivots(done, 3)
    if h4 is not None and len(h4) > 20:
        lv["4H swing high"], lv["4H swing low"] = last_pivots(h4, 5)
    return {k: float(v) for k, v in lv.items() if pd.notna(v)}


def analyze(tickers, daily_map, h1_map, live_map, min_adv_m, near_pct):
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows, levels_map = [], {}
    for t in tickers:
        _, name, market, sub, suit, mult, pv = INFO[t]
        base = {"Asset": name, "Ticker": t, "Market": market}
        d = daily_map.get(t)
        if d is None or len(d) < 30:
            rows.append({**base, "Status": "NO DATA", "Reason": "No data from Yahoo"})
            continue
        done = d[d.index < today]
        prev_close = float(done["Close"].iloc[-1]) if len(done) else float(d["Close"].iloc[-1])
        price = live_map.get(t) or float(d["Close"].iloc[-1])

        atr = wilder_atr(d)
        atr_now = float(atr.iloc[-1])
        atr_up = atr_now > float(atr.iloc[-6]) or atr_now >= float(atr.iloc[-20:].max())
        atr_ratio = atr_now / float(atr.iloc[-20:].mean())

        if mult is None:
            adv, vol_ok = np.nan, True
        else:
            dv = d["Volume"] if mult == 0 else d["Volume"] * d["Close"] * mult
            dv = dv[d.index < today].tail(20)
            adv = float(dv.mean()) if len(dv) else np.nan
            vol_ok = pd.notna(adv) and adv >= min_adv_m * 1e6

        lv = liquidity_levels(d, to_4h(h1_map.get(t)), today)
        levels_map[t] = lv
        above = {k: v for k, v in lv.items() if v > price}
        below = {k: v for k, v in lv.items() if v < price}
        bsl_n, bsl = min(above.items(), key=lambda kv: kv[1]) if above else (None, np.nan)
        ssl_n, ssl = max(below.items(), key=lambda kv: kv[1]) if below else (None, np.nan)
        d_bsl = (bsl - price) / price * 100 if above else np.inf
        d_ssl = (price - ssl) / price * 100 if below else np.inf
        dist = min(d_bsl, d_ssl)
        if d_ssl <= d_bsl:
            bias, lvl_n, lvl, opp_n, opp = "LONG", ssl_n, ssl, bsl_n, bsl
        else:
            bias, lvl_n, lvl, opp_n, opp = "SHORT", bsl_n, bsl, ssl_n, ssl

        passed = vol_ok and atr_up and dist <= near_pct
        reason = ("All filters passed" if passed else "Low $ volume" if not vol_ok else
                  "ATR not expanding" if not atr_up else f"Mid-range ({dist:.1f}% away)")
        prox = max(0.0, 1 - dist / near_pct) if np.isfinite(dist) else 0.0
        atr_part = min(max(atr_ratio - 1, 0), 0.5) / 0.5
        score = round(40 * suit / 3 + 35 * prox + 25 * atr_part, 1)

        rows.append({**base, "Status": "PASS" if passed else "SKIP", "Reason": reason,
                     "Score": score if passed else np.nan, "Price": price,
                     "Chg %": (price / prev_close - 1) * 100 if prev_close else np.nan,
                     "ADV ($M)": adv / 1e6 if pd.notna(adv) else np.nan,
                     "ATR ratio": round(atr_ratio, 2), "Bias": bias,
                     "Liquidity": lvl_n, "Level": lvl, "Opp name": opp_n, "Opp level": opp,
                     "Distance %": round(dist, 2) if np.isfinite(dist) else np.nan})
    df = pd.DataFrame(rows)
    order = {"PASS": 0, "SKIP": 1, "NO DATA": 2}
    for col in ("Score", "Distance %"):
        if col not in df:
            df[col] = np.nan
    df = df.sort_values(by=["Status", "Score", "Distance %"],
                        key=lambda s: s.map(order) if s.name == "Status" else s,
                        ascending=[True, False, True], na_position="last").reset_index(drop=True)
    return df, levels_map


# ─────────────────────────── 15M setup engine ───────────────────────────
def _simulate(h, l, i, entry, sl, tp1, tp2):
    filled = tp1hit = False
    for j in range(i + 1, len(h)):
        if not filled:
            if h[j] >= tp1:
                return "Cancelled – ran to TP1 before fill", j
            if j - i > ORDER_EXPIRY:
                return "Expired unfilled", j
            if l[j] <= entry:
                filled = True
                if l[j] <= sl:
                    return "Stopped out", j
        elif not tp1hit:
            if l[j] <= sl:
                return "Stopped out", j
            if h[j] >= tp1:
                tp1hit = True
                if h[j] >= tp2:
                    return "TP2 hit – full win", j
        else:
            if h[j] >= tp2:
                return "TP2 hit – full win", j
            if l[j] <= entry:
                return "TP1 hit, rest closed at break-even", j
    if not filled:
        return "Order working", None
    return ("In trade – TP1 hit, stop at BE" if tp1hit else "In trade"), None


def _scan_long(h, l, c, ssl, bsl, buf):
    """LONG side of the LS-SS state machine. Shorts reuse it on negated prices."""
    n = len(h)
    ph = pivot_flags(h, K15)
    setups, state, i = [], 0, max(K15 + 2, n - SCAN_BARS)
    ext = mss = sidx = midx = lvl = fvg = None
    while i < n:
        if state == 0:
            hits = [v for v in ssl if l[i] < v <= l[i - 1]]
            if hits:
                state, ext, sidx, lvl, fvg = 1, l[i], i, max(hits), None
                j = next((j for j in range(i - K15, K15 - 1, -1) if ph[j]), None)
                mss = h[j] if j is not None else None
        if state in (1, 2):
            if state == 2 and l[i] < ext:
                state = 0
            else:
                if state == 1 and l[i] < ext:
                    ext, sidx, fvg = l[i], i, None
                if state == 1 and ph[i - K15] and i - K15 < sidx:
                    mss = h[i - K15]
                if i - 2 >= sidx and l[i] > h[i - 2]:
                    fvg = (l[i], h[i - 2])
                if state == 1:
                    if mss is not None and c[i] > mss:
                        state, midx = 2, i
                    elif i - sidx > SWEEP_EXPIRY:
                        state = 0
                if state == 2:
                    if fvg:
                        entry, sl = fvg[0], ext - buf
                        tp1 = nearest_unmitigated(h, ph, i, entry)
                        above = [v for v in bsl if v > (tp1 if tp1 is not None else entry)]
                        tp2 = min(above) if above else None
                        rr = reward_risk(entry, sl, tp1, tp2)
                        s = {"bar": i, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                             "rr": rr, "level": lvl}
                        if tp1 is None or tp2 is None:
                            s["status"], s["end"] = "Skipped – no target", i
                        elif rr < MIN_RR:
                            s["status"], s["end"] = f"Skipped – RR {rr:.2f}", i
                        else:
                            s["status"], s["end"] = _simulate(h, l, i, entry, sl, tp1, tp2)
                        setups.append(s)
                        state = 0
                        if s["end"] is None:
                            break
                        i = max(i, s["end"])
                    elif i - midx > FVG_WAIT:
                        state = 0
        i += 1
    return setups


def detect_setup(m15, levels, buf):
    if m15 is None or len(m15) < 60:
        return None
    df = m15.tail(500)
    h, l, c = (df[col].to_numpy(float) for col in ("High", "Low", "Close"))
    ssl = [v for k, v in levels.items() if "low" in k]
    bsl = [v for k, v in levels.items() if "high" in k]
    found = []
    for s in _scan_long(h, l, c, ssl, bsl, buf):
        found.append({**s, "dir": "LONG"})
    for s in _scan_long(-l, -h, -c, [-v for v in bsl], [-v for v in ssl], buf):
        neg = {k: (-s[k] if s[k] is not None else None) for k in ("entry", "sl", "tp1", "tp2", "level")}
        found.append({**s, **neg, "dir": "SHORT"})
    if not found:
        return None
    s = max(found, key=lambda x: x["bar"])
    s["time"] = df.index[s["bar"]]
    s["active"] = s["status"] in ACTIVE
    return s


def projected_plan(row, m15, levels, buf):
    """Estimated levels while waiting for the sweep (firm up once a 15M MSS + FVG forms)."""
    if m15 is None or len(m15) < 60 or pd.isna(row.get("Level")):
        return None
    df = m15.tail(500)
    atr15 = float(wilder_atr(df).iloc[-1])
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    lvl, n = float(row["Level"]), len(df)
    if row["Bias"] == "LONG":
        entry, sl = lvl + 0.5 * atr15, lvl - atr15 - buf
        tp1 = nearest_unmitigated(h, pivot_flags(h, K15), n - 1, entry)
        above = [v for k, v in levels.items() if "high" in k and v > (tp1 or entry)]
        tp2 = min(above) if above else None
    else:
        nh = -l
        entry, sl = lvl - 0.5 * atr15, lvl + atr15 + buf
        t1 = nearest_unmitigated(nh, pivot_flags(nh, K15), n - 1, -entry)
        tp1 = -t1 if t1 is not None else None
        below = [v for k, v in levels.items() if "low" in k and v < (tp1 or entry)]
        tp2 = max(below) if below else None
    return {"dir": row["Bias"], "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "rr": reward_risk(entry, sl, tp1, tp2), "level": lvl,
            "status": f"Waiting for sweep of {row['Liquidity']}", "active": False}


# ─────────────────────────── UI ───────────────────────────
def plan_card(t, row, plan, risk_cash):
    kind = "🟢 LIVE SETUP" if plan.get("active") else "🟡 PROJECTED"
    st.markdown(f"**{kind}** — {plan['status']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Entry (limit)", fmt(plan["entry"], t))
    c2.metric("Stop loss", fmt(plan["sl"], t))
    c3.metric("TP1 (50%)", fmt(plan["tp1"], t))
    c4, c5, c6 = st.columns(3)
    c4.metric("TP2 (50%)", fmt(plan["tp2"], t))
    rr = plan.get("rr")
    c5.metric("Reward : risk", f"1 : {rr:.2f}" if rr else "–",
              ("meets 1:2.5" if rr and rr >= MIN_RR else "below 1:2.5") if rr else None,
              delta_color="normal" if rr and rr >= MIN_RR else "inverse")
    c6.metric("Size (1% risk)", position_size(t, risk_cash, plan["entry"], plan["sl"]))
    if plan.get("active"):
        st.caption(f"15M setup formed {plan['time']:%a %H:%M} UTC: swept {fmt(plan['level'], t)}, "
                   "MSS + FVG confirmed. Move stop to entry once TP1 fills.")
    else:
        st.caption(f"Estimate only. Wait for price to sweep {row['Liquidity']} ({fmt(row['Level'], t)}), "
                   "then a 15M MSS + FVG. The card switches to LIVE SETUP with exact levels when it forms.")


def main():
    st.set_page_config(page_title="LS-AS Live Screener", page_icon="🎯", layout="wide")
    st.title("🎯 LS-AS Live Screener")

    with st.sidebar:
        st.header("Settings")
        account = st.number_input("Total account ($)", min_value=1000.0, value=100000.0, step=1000.0)
        active_pct = st.slider("Active trading capital (%)", 5, 100, 20)
        risk_pct = st.number_input("Risk per trade (% of active)", 0.1, 5.0, 1.0, 0.1)
        min_adv = st.number_input("Min avg daily $ volume ($M)", 0.0, 100000.0, 500.0, 50.0)
        near_pct = st.number_input("Max distance to liquidity (%)", 0.1, 10.0, 2.0, 0.1)
        auto = st.toggle("Auto-refresh live prices", value=True)
        every = st.select_slider("Refresh every (seconds)", [15, 30, 60, 120], value=30)
        if st.button("🔄 Reload all data"):
            st.cache_data.clear()
    risk_cash = account * active_pct / 100 * risk_pct / 100

    market = st.radio("Market", MARKETS, horizontal=True)
    tickers = tuple(u[0] for u in UNIVERSE if market == "All" or u[2] == market)
    with st.spinner("Downloading market data…"):
        daily = fetch_daily(tickers)
        h1 = fetch_h1(tickers)
    if not daily:
        st.error("Couldn't download data from Yahoo Finance. Tap **Reload all data** in a minute.")
        return

    @st.fragment(run_every=every if auto else None)
    def live_section():
        m15 = fetch_m15(tickers)
        quotes = fetch_live(tickers)
        live_map = {t: quotes.get(t) or (float(m15[t]["Close"].iloc[-1]) if t in m15 else None)
                    for t in tickers}
        df, levels_map = analyze(tickers, daily, h1, live_map, min_adv, near_pct)

        plans = {}
        for _, row in df[df["Status"] != "NO DATA"].iterrows():
            t = row["Ticker"]
            buf = stop_buffer(t, row["Price"])
            setup = detect_setup(m15.get(t), levels_map.get(t, {}), buf)
            plans[t] = setup if setup and setup["active"] else projected_plan(row, m15.get(t),
                                                                              levels_map.get(t, {}), buf)

        passed = df[df["Status"] == "PASS"].copy()
        passed["live"] = passed["Ticker"].map(lambda t: bool(plans.get(t) and plans[t].get("active")))
        passed = passed.sort_values(["live", "Score"], ascending=[False, False])

        st.caption(f"Risk per trade: **${risk_cash:,.0f}** · Updated "
                   f"{datetime.now(timezone.utc):%H:%M:%S} UTC" + (f" · refreshing every {every}s" if auto else ""))

        # ── Top pick ──
        if passed.empty:
            st.warning(f"**No {market.lower() if market != 'All' else ''} asset passes all filters right now.** "
                       "Per the rules, sit out or check another market.")
            default_pick = df.iloc[0]["Ticker"]
        else:
            top = passed.iloc[0]
            t = top["Ticker"]
            default_pick = t
            st.success(f"### {top['Asset']} ({t}) — {top['Bias']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Live price", fmt(top["Price"], t),
                      f"{top['Chg %']:+.2f}%" if pd.notna(top["Chg %"]) else None)
            c2.metric("Score", f"{top['Score']:.0f} / 100")
            c3.metric("To liquidity", f"{top['Distance %']:.2f}%")
            if plans.get(t):
                plan_card(t, top, plans[t], risk_cash)
            if len(passed) > 1:
                ru = passed.iloc[1]
                st.info(f"**Second slot:** {ru['Asset']} ({ru['Ticker']}) — {ru['Bias']}, score {ru['Score']:.0f}. "
                        "Skip it if it's correlated with your first pick (e.g. ES + NQ, SPY + VOO).")

        # ── Market table ──
        st.subheader(f"{market} — live board")
        board = []
        for _, r in df.iterrows():
            t, p = r["Ticker"], plans.get(r["Ticker"]) or {}
            board.append({
                "Asset": f"{r['Asset']} ({t})",
                "Live": fmt(r.get("Price"), t),
                "Chg %": f"{r['Chg %']:+.2f}" if pd.notna(r.get("Chg %")) else "–",
                "Screen": "✅" if r["Status"] == "PASS" else f"— {r['Reason']}",
                "Bias": r.get("Bias", "–") if r["Status"] != "NO DATA" else "–",
                "Setup": ("🟢 " if p.get("active") else "🟡 ") + p["status"] if p else "–",
                "Entry": fmt(p.get("entry"), t), "Stop": fmt(p.get("sl"), t),
                "TP1": fmt(p.get("tp1"), t), "TP2": fmt(p.get("tp2"), t),
                "R:R": f"{p['rr']:.2f}" if p.get("rr") else "–",
            })
        st.dataframe(pd.DataFrame(board), hide_index=True)

        # ── Chart ──
        st.subheader("15M chart")
        opts = [t for t in df["Ticker"] if t in m15 or t in daily]
        pick = st.selectbox("Asset", opts, index=opts.index(default_pick) if default_pick in opts else 0,
                            format_func=lambda t: f"{INFO[t][1]} ({t})", key="chart_pick")
        src = m15.get(pick)
        d = src.tail(160) if src is not None else daily[pick].tail(90)
        fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                       close=d["Close"], name=pick))
        lo, hi = float(d["Low"].min()), float(d["High"].max())
        pad = (hi - lo) * 0.3
        for lname, val in levels_map.get(pick, {}).items():
            if lo - pad <= val <= hi + pad:
                fig.add_hline(y=val, line_dash="dot", line_width=1,
                              line_color="#26a69a" if "low" in lname else "#ef5350",
                              annotation_text=lname, annotation_position="top left", annotation_font_size=10)
        p = plans.get(pick) or {}
        for key, label, colr in (("entry", "Entry", "#2962ff"), ("sl", "Stop", "#d50000"),
                                 ("tp1", "TP1", "#00c853"), ("tp2", "TP2", "#00c853")):
            if p.get(key) is not None:
                fig.add_hline(y=p[key], line_width=2, line_color=colr, annotation_text=label,
                              annotation_position="bottom right", annotation_font_size=11)
        fig.update_layout(height=440, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, key="chart")

    live_section()

    with st.expander("Rules checklist"):
        st.markdown(
            "- Max **2 open trades** across all markets\n"
            "- No trades within 2 hours of CPI, NFP or rate decisions\n"
            "- Stop for the day at **−2%**, for the week at **−5%** (of active capital)\n"
            "- Move stop to break-even when TP1 is hit\n"
            "- 🟡 Projected levels are estimates; only act on 🟢 live setups")
    st.caption("Data: Yahoo Finance (may be delayed ~15 min for some futures/stocks). "
               "Educational tool, not financial advice.")


if __name__ == "__main__":
    main()
