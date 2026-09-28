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
        ema20 = float(d["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(d["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
        trend = ("Up" if ema20 > ema50 and price > ema50 else
                 "Down" if ema20 < ema50 and price < ema50 else "Neutral")

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
                     "ATR ratio": round(atr_ratio, 2), "Bias": bias, "Trend": trend,
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
                             "rr": rr, "level": lvl, "mss_bar": midx, "fvg_size": fvg[0] - fvg[1]}
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
    atr = wilder_atr(df).to_numpy(float)
    mb = s["mss_bar"]
    s["disp_atr"] = (h[mb] - l[mb]) / atr[mb] if atr[mb] > 0 else 0.0
    s["fvg_atr"] = s["fvg_size"] / atr[s["bar"]] if atr[s["bar"]] > 0 else 0.0
    s["level_name"] = next((k for k, v in levels.items() if abs(v - s["level"]) < 1e-9), "HTF level")
    s["time"] = df.index[s["bar"]]
    s["active"] = s["status"] in ACTIVE
    return s


# ─────────────────────────── A+ grading ───────────────────────────
def in_killzone(ts):
    """London 07:00–10:00 UTC or New York 12:00–16:00 UTC."""
    hr = ts.hour + ts.minute / 60
    return 7 <= hr < 10 or 12 <= hr < 16


def grade(row, s, min_rr, need_trend, need_session):
    checks = {
        "Passed weekly screen": row["Status"] == "PASS",
        "Live 15M setup (sweep + MSS + FVG)": bool(s.get("active")),
        f"Reward:risk ≥ 1:{min_rr:g}": bool(s.get("rr")) and s["rr"] >= min_rr,
        "Strong displacement (MSS candle ≥ 1.2× ATR)": s.get("disp_atr", 0) >= 1.2,
        "Clean FVG (≥ 0.25× ATR)": s.get("fvg_atr", 0) >= 0.25,
    }
    if need_trend:
        checks["With the daily trend"] = ((s["dir"] == "LONG" and row.get("Trend") == "Up") or
                                          (s["dir"] == "SHORT" and row.get("Trend") == "Down"))
    if need_session:
        checks["Formed in London / New York session"] = in_killzone(s["time"])
    return checks


# ─────────────────────────── UI ───────────────────────────
def setup_card(t, row, s, checks, risk_cash):
    st.success(f"### A+ {s['dir']} — {row['Asset']} ({t})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Live price", fmt(row["Price"], t),
              f"{row['Chg %']:+.2f}%" if pd.notna(row.get("Chg %")) else None)
    c2.metric("Status", s["status"])
    c3.metric("Reward : risk", f"1 : {s['rr']:.2f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Entry (limit)", fmt(s["entry"], t))
    c5.metric("Stop loss", fmt(s["sl"], t))
    c6.metric("Size (1% risk)", position_size(t, risk_cash, s["entry"], s["sl"]))
    c7, c8, _ = st.columns(3)
    c7.metric("TP1 (close 50%)", fmt(s["tp1"], t))
    c8.metric("TP2 (close 50%)", fmt(s["tp2"], t))
    st.caption(f"Swept **{s['level_name']}** ({fmt(s['level'], t)}) · setup formed "
               f"{s['time']:%a %H:%M} UTC · {len(checks)}/{len(checks)} checks: "
               + " · ".join(f"✅ {k}" for k in checks)
               + ". Move the stop to entry once TP1 fills.")


def main():
    st.set_page_config(page_title="LS-AS A+ Setups", page_icon="🎯", layout="wide")
    st.title("🎯 LS-AS A+ Setups")

    with st.sidebar:
        st.header("Account")
        account = st.number_input("Total account ($)", min_value=1000.0, value=100000.0, step=1000.0)
        active_pct = st.slider("Active trading capital (%)", 5, 100, 20)
        risk_pct = st.number_input("Risk per trade (% of active)", 0.1, 5.0, 1.0, 0.1)
        st.header("A+ criteria")
        min_rr = st.number_input("Min reward:risk", 2.5, 10.0, 3.0, 0.5)
        need_trend = st.toggle("Must agree with daily trend", value=True)
        need_session = st.toggle("Must form in London / NY session", value=True)
        st.header("Weekly screen")
        min_adv = st.number_input("Min avg daily $ volume ($M)", 0.0, 100000.0, 500.0, 50.0)
        near_pct = st.number_input("Max distance to liquidity (%)", 0.1, 10.0, 2.0, 0.1)
        st.header("Refresh")
        auto = st.toggle("Auto-refresh", value=True)
        every = st.select_slider("Every (seconds)", [15, 30, 60, 120], value=30)
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

        aplus, misses = [], []
        for _, row in df[df["Status"] != "NO DATA"].iterrows():
            t = row["Ticker"]
            s = detect_setup(m15.get(t), levels_map.get(t, {}), stop_buffer(t, row["Price"]))
            if not s or not s["active"]:
                continue
            checks = grade(row, s, min_rr, need_trend, need_session)
            if all(checks.values()):
                aplus.append((t, row, s, checks))
            else:
                misses.append({"Asset": f"{row['Asset']} ({t})", "Direction": s["dir"],
                               "Status": s["status"], "R:R": f"{s['rr']:.2f}" if s.get("rr") else "–",
                               "Missing": " · ".join(k for k, ok in checks.items() if not ok)})
        aplus.sort(key=lambda x: (x[2]["status"] != "Order working", -x[2]["rr"]))

        st.caption(f"Risk per trade **${risk_cash:,.0f}** · scanned {len(df)} assets · updated "
                   f"{datetime.now(timezone.utc):%H:%M:%S} UTC"
                   + (f" · rescanning every {every}s" if auto else ""))

        if not aplus:
            st.info(f"**No A+ setups right now{'' if market == 'All' else ' in ' + market}.** "
                    "Most of the time this is the correct answer — the app keeps scanning.")
            watch = df[df["Status"] == "PASS"]
            if len(watch):
                st.caption("Watching for sweeps: " + " · ".join(
                    f"{r['Asset']} ({'below' if r['Bias'] == 'LONG' else 'above'} {fmt(r['Level'], r['Ticker'])})"
                    for _, r in watch.head(6).iterrows()))
        else:
            if len(aplus) > 2:
                st.warning(f"{len(aplus)} A+ setups — the rules allow **max 2 open trades**. "
                           "Take the top two and avoid correlated pairs (e.g. ES + NQ).")
            for t, row, s, checks in aplus:
                setup_card(t, row, s, checks, risk_cash)

            st.subheader("15M chart")
            opts = [x[0] for x in aplus]
            pick = st.selectbox("Setup", opts, format_func=lambda t: f"{INFO[t][1]} ({t})", key="chart_pick")
            s = next(x[2] for x in aplus if x[0] == pick)
            d = m15[pick].tail(160)
            fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                           close=d["Close"], name=pick))
            for key, label, colr in (("level", "Swept level", "#9e9e9e"), ("entry", "Entry", "#2962ff"),
                                     ("sl", "Stop", "#d50000"), ("tp1", "TP1", "#00c853"),
                                     ("tp2", "TP2", "#00c853")):
                if s.get(key) is not None:
                    fig.add_hline(y=s[key], line_width=1 if key == "level" else 2, line_color=colr,
                                  line_dash="dot" if key == "level" else "solid", annotation_text=label,
                                  annotation_position="bottom right", annotation_font_size=11)
            fig.update_layout(height=440, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, key="chart")

        if misses:
            with st.expander(f"Filtered out: {len(misses)} live setup(s) that aren't A+ (don't trade)"):
                st.dataframe(pd.DataFrame(misses), hide_index=True)

    live_section()

    with st.expander("Rules checklist"):
        st.markdown(
            "- Only trade setups shown as **A+**\n"
            "- Max **2 open trades** across all markets\n"
            "- No trades within 2 hours of CPI, NFP or rate decisions\n"
            "- Stop for the day at **−2%**, for the week at **−5%** (of active capital)\n"
            "- Move stop to break-even when TP1 is hit")
    st.caption("Data: Yahoo Finance (some stocks/futures delayed ~15 min). "
               "Educational tool, not financial advice.")


if __name__ == "__main__":
    main()
