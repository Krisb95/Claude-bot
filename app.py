"""
LS-AS Weekly Asset Screener  —  Streamlit app
Screens the LS-AS universe for liquidity, ATR expansion and proximity to
4H/Daily liquidity pools, then recommends which asset(s) to trade this week.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ticker, name, tier, class, suitability (1-3), $-volume multiplier, point value
#   multiplier: None = no real volume (forex), 0 = volume already in USD (crypto)
UNIVERSE = [
    ("ES=F",     "E-mini S&P 500",       1, "Index futures", 3.0, 50,   50),
    ("NQ=F",     "E-mini Nasdaq 100",    1, "Index futures", 3.0, 20,   20),
    ("SPY",      "S&P 500 ETF",          1, "ETF",           3.0, 1,    1),
    ("QQQ",      "Nasdaq 100 ETF",       1, "ETF",           3.0, 1,    1),
    ("VT",       "Total World Stock ETF", 1, "ETF",          3.0, 1,    1),
    ("VOO",      "Vanguard S&P 500 ETF", 1, "ETF",           3.0, 1,    1),
    ("EURUSD=X", "EUR/USD",              2, "Forex",         3.0, None, 1),
    ("GBPUSD=X", "GBP/USD",              2, "Forex",         3.0, None, 1),
    ("JPY=X",    "USD/JPY",              2, "Forex",         3.0, None, 1),
    ("GC=F",     "Gold futures",         2, "Commodity",     2.5, 100,  100),
    ("CL=F",     "WTI Crude futures",    2, "Commodity",     2.5, 1000, 1000),
    ("BTC-USD",  "Bitcoin",              3, "Crypto",        1.5, 0,    1),
    ("ETH-USD",  "Ethereum",             3, "Crypto",        1.5, 0,    1),
    ("NVDA",     "NVIDIA",               3, "Mega-cap",      2.0, 1,    1),
    ("AAPL",     "Apple",                3, "Mega-cap",      2.0, 1,    1),
    ("MSFT",     "Microsoft",            3, "Mega-cap",      2.0, 1,    1),
    ("AMZN",     "Amazon",               3, "Mega-cap",      2.0, 1,    1),
    ("GOOGL",    "Alphabet",             3, "Mega-cap",      2.0, 1,    1),
    ("TSLA",     "Tesla",                3, "Mega-cap",      2.0, 1,    1),
]
INFO = {u[0]: u for u in UNIVERSE}


# ─────────────────────────── data ───────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(tickers, period, interval):
    data = yf.download(list(tickers), period=period, interval=interval, group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)
    out = {}
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


def to_4h(h1):
    if h1 is None or h1.empty:
        return None
    return (h1.resample("4h")
              .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
              .dropna(subset=["Close"]))


# ─────────────────────────── analysis ───────────────────────────
def last_pivots(df, k):
    """Most recent confirmed swing high / low (k bars each side)."""
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


def liquidity_levels(daily, h4, today):
    done = daily[daily.index < today]          # completed daily bars only
    lv = {}
    if len(done):
        lv["Prev day high"] = done["High"].iloc[-1]
        lv["Prev day low"] = done["Low"].iloc[-1]
        # on weekends, the week that just finished counts as "previous week"
        ref = today if today.weekday() < 5 else today + pd.Timedelta(days=7 - today.weekday())
        week_start = ref - pd.Timedelta(days=ref.weekday())
        wk = done.resample("W-SUN").agg({"High": "max", "Low": "min"}).dropna()
        wk = wk[wk.index < week_start]
        if len(wk):
            lv["Prev week high"] = wk["High"].iloc[-1]
            lv["Prev week low"] = wk["Low"].iloc[-1]
        if len(done) > 10:
            dph, dpl = last_pivots(done, 3)
            lv["Daily swing high"], lv["Daily swing low"] = dph, dpl
    if h4 is not None and len(h4) > 20:
        hph, hpl = last_pivots(h4, 5)
        lv["4H swing high"], lv["4H swing low"] = hph, hpl
    return {k: float(v) for k, v in lv.items() if pd.notna(v)}


def analyze(daily_map, h1_map, min_adv_m, near_pct):
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows, levels_map = [], {}
    for t, name, tier, cls, suit, mult, pv in UNIVERSE:
        base = {"Asset": name, "Ticker": t, "Tier": tier, "Class": cls}
        d = daily_map.get(t)
        if d is None or len(d) < 30:
            rows.append({**base, "Status": "NO DATA", "Reason": "No data from Yahoo", "Score": np.nan})
            continue
        close = float(d["Close"].iloc[-1])

        # ATR (Wilder, 14)
        prev_c = d["Close"].shift()
        tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev_c).abs(), (d["Low"] - prev_c).abs()],
                       axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        atr_now = float(atr.iloc[-1])
        atr_up = atr_now > float(atr.iloc[-6]) or atr_now >= float(atr.iloc[-20:].max())
        atr_ratio = atr_now / float(atr.iloc[-20:].mean())

        # Average daily $ volume (last 20 completed days)
        if mult is None:
            adv, vol_ok = np.nan, True
        else:
            dv = d["Volume"] if mult == 0 else d["Volume"] * d["Close"] * mult
            dv = dv[d.index < today].tail(20)
            adv = float(dv.mean()) if len(dv) else np.nan
            vol_ok = pd.notna(adv) and adv >= min_adv_m * 1e6

        # Nearest buy-side / sell-side liquidity
        lv = liquidity_levels(d, to_4h(h1_map.get(t)), today)
        levels_map[t] = lv
        above = {k: v for k, v in lv.items() if v > close}
        below = {k: v for k, v in lv.items() if v < close}
        bsl_n, bsl = min(above.items(), key=lambda kv: kv[1]) if above else (None, np.nan)
        ssl_n, ssl = max(below.items(), key=lambda kv: kv[1]) if below else (None, np.nan)
        d_bsl = (bsl - close) / close * 100 if above else np.inf
        d_ssl = (close - ssl) / close * 100 if below else np.inf
        dist = min(d_bsl, d_ssl)
        near_ok = dist <= near_pct

        if d_ssl <= d_bsl:
            side, lvl_n, lvl = "LONG", ssl_n, ssl
            plan = (f"Price is {d_ssl:.2f}% above **{ssl_n}** ({fmt(ssl)}). Wait for a sweep below it, "
                    f"then a 15M bullish MSS + FVG → long. Opposing target: "
                    + (f"{bsl_n} ({fmt(bsl)})." if above else "next buy-side high."))
        else:
            side, lvl_n, lvl = "SHORT", bsl_n, bsl
            plan = (f"Price is {d_bsl:.2f}% below **{bsl_n}** ({fmt(bsl)}). Wait for a sweep above it, "
                    f"then a 15M bearish MSS + FVG → short. Opposing target: "
                    + (f"{ssl_n} ({fmt(ssl)})." if below else "next sell-side low."))

        passed = vol_ok and atr_up and near_ok
        reason = ("All filters passed" if passed else
                  "Low $ volume" if not vol_ok else
                  "ATR not expanding" if not atr_up else
                  f"Mid-range ({dist:.1f}% from liquidity)")

        prox = max(0.0, 1 - dist / near_pct) if np.isfinite(dist) else 0.0
        atr_part = min(max(atr_ratio - 1, 0), 0.5) / 0.5
        score = round(40 * suit / 3 + 35 * prox + 25 * atr_part, 1)

        rows.append({**base, "Status": "PASS" if passed else "SKIP", "Reason": reason,
                     "Score": score if passed else np.nan, "Price": close,
                     "ADV ($M)": adv / 1e6 if pd.notna(adv) else np.nan,
                     "ATR trend": "Rising" if atr_up else "Flat/falling", "ATR ratio": round(atr_ratio, 2),
                     "Nearest liquidity": lvl_n, "Level": lvl,
                     "Distance %": round(dist, 2) if np.isfinite(dist) else np.nan,
                     "Bias": side, "Plan": plan, "Point value": pv})
    df = pd.DataFrame(rows)
    order = {"PASS": 0, "SKIP": 1, "NO DATA": 2}
    df = df.sort_values(by=["Status", "Score", "Distance %"],
                        key=lambda s: s.map(order) if s.name == "Status" else s,
                        ascending=[True, False, True], na_position="last").reset_index(drop=True)
    return df, levels_map


def fmt(v):
    if v is None or pd.isna(v):
        return "–"
    return f"{v:,.5f}" if abs(v) < 10 else f"{v:,.2f}"


# ─────────────────────────── UI ───────────────────────────
def main():
    st.set_page_config(page_title="LS-AS Screener", page_icon="🎯", layout="wide")
    st.title("🎯 LS-AS Weekly Asset Screener")
    st.caption("Filters: $ volume • expanding ATR • within reach of 4H/Daily liquidity. "
               "Run it Sunday before the open.")

    with st.sidebar:
        st.header("Settings")
        account = st.number_input("Total account ($)", min_value=1000.0, value=100000.0, step=1000.0)
        active_pct = st.slider("Active trading capital (%)", 5, 100, 20)
        risk_pct = st.number_input("Risk per trade (% of active)", 0.1, 5.0, 1.0, 0.1)
        min_adv = st.number_input("Min avg daily $ volume ($M)", 0.0, 100000.0, 500.0, 50.0)
        near_pct = st.number_input("Max distance to liquidity (%)", 0.1, 10.0, 2.0, 0.1)
        if st.button("🔄 Refresh data"):
            st.cache_data.clear()

    tickers = tuple(u[0] for u in UNIVERSE)
    with st.spinner("Downloading market data…"):
        daily = fetch(tickers, "1y", "1d")
        h1 = fetch(tickers, "60d", "1h")
    if not daily:
        st.error("Couldn't download data from Yahoo Finance. Tap **Refresh data** in a minute.")
        return

    df, levels_map = analyze(daily, h1, min_adv, near_pct)
    passed = df[df["Status"] == "PASS"]

    # ── Recommendation ──
    if passed.empty:
        st.warning("**No asset passes all filters this week.** Per the rules, the right move is to sit out.")
        default_pick = df.iloc[0]["Ticker"]
    else:
        top = passed.iloc[0]
        default_pick = top["Ticker"]
        st.success(f"### Recommended: {top['Asset']} ({top['Ticker']}) — {top['Bias']} bias")
        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{top['Score']:.0f} / 100")
        c2.metric("Distance to liquidity", f"{top['Distance %']:.2f}%")
        c3.metric("ATR vs 20-day avg", f"{top['ATR ratio']:.2f}×")
        st.markdown(f"**Plan:** {top['Plan']}")
        if len(passed) > 1:
            ru = passed.iloc[1]
            st.info(f"**Second slot (max 2 trades):** {ru['Asset']} ({ru['Ticker']}) — {ru['Bias']} bias, "
                    f"score {ru['Score']:.0f}. Avoid if it's highly correlated with your first pick "
                    f"(e.g. ES + NQ, SPY + VOO).")

    # ── Chart ──
    st.subheader("Chart & liquidity levels")
    names = {t: f"{INFO[t][1]} ({t})" for t in df["Ticker"] if t in daily}
    pick = st.selectbox("Asset", list(names), index=list(names).index(default_pick) if default_pick in names else 0,
                        format_func=lambda t: names[t])
    d = daily[pick].tail(90)
    fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                   close=d["Close"], name=pick))
    for lname, val in levels_map.get(pick, {}).items():
        color = "#26a69a" if "low" in lname else "#ef5350"
        fig.add_hline(y=val, line_dash="dot", line_color=color, annotation_text=lname,
                      annotation_position="top left", annotation_font_size=10)
    fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig)

    # ── Full table ──
    st.subheader("This week's screen")
    table = df[["Asset", "Ticker", "Status", "Reason", "Score", "Price", "ADV ($M)",
                "ATR trend", "Nearest liquidity", "Distance %", "Bias"]].copy()
    table["Price"] = table["Price"].map(fmt)
    table["ADV ($M)"] = table["ADV ($M)"].map(lambda v: "n/a" if pd.isna(v) else f"{v:,.0f}")
    st.dataframe(table, hide_index=True)

    # ── Position size ──
    st.subheader("Position size calculator")
    active = account * active_pct / 100
    risk_cash = active * risk_pct / 100
    st.write(f"Active capital **${active:,.0f}** → risk per trade **${risk_cash:,.0f}**")
    row = df[df["Ticker"] == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    entry = c1.number_input("Entry (FVG edge)", value=0.0, format="%.5f")
    stop = c2.number_input("Stop (beyond sweep)", value=0.0, format="%.5f")
    target = c3.number_input("TP2 target", value=0.0, format="%.5f")
    if entry > 0 and stop > 0 and entry != stop:
        dist = abs(entry - stop)
        pv = row.get("Point value", 1) or 1
        if pick == "JPY=X":                     # P&L is in JPY → convert to USD
            size = risk_cash * entry / dist
        else:
            size = risk_cash / (dist * pv)
        if INFO[pick][3] in ("Index futures", "Commodity"):
            contracts = int(size)
            st.metric("Size", f"{contracts} contract(s)")
            if contracts == 0:
                st.caption("Under 1 contract — use the micro version (MES / MNQ / MGC / MCL, 1/10 size) "
                           f"≈ {int(size * 10)} micro contract(s).")
        elif INFO[pick][3] == "Forex":
            st.metric("Size", f"{size:,.0f} units  ({size / 100000:.2f} lots)")
        elif INFO[pick][3] == "Crypto":
            st.metric("Size", f"{size:,.4f} coins")
        else:
            st.metric("Size", f"{int(size):,} shares")
        if target > 0:
            rr = abs(target - entry) / dist
            ok = rr >= 2.5
            st.write(f"Reward : risk = **1 : {rr:.2f}** " + ("✅ meets 1:2.5" if ok else "❌ below 1:2.5 — skip"))

    with st.expander("Weekly rules checklist"):
        st.markdown(
            "- Max **2 open trades** across all assets\n"
            "- No trades within 2 hours of CPI, NFP or rate decisions\n"
            "- Stop for the day at **−2%**, for the week at **−5%** (of active capital)\n"
            "- Move stop to break-even when TP1 is hit\n"
            "- Assets that are mid-range this week stay off the list until next Sunday")
    st.caption("Data: Yahoo Finance (may be delayed). Futures volume is front-month only. "
               "Educational tool, not financial advice.")


if __name__ == "__main__":
    main()
