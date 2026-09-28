# Bull Run Strategy V2

A discretionary trading support dashboard. It does **not** execute trades, connect
to any exchange, or place/modify orders — you enter everything manually on your own
platform. Setup scores measure checklist confluence only; they are not
win-probability estimates or profitability guarantees.

## Deploying to Streamlit Cloud

1. Put **every file below flat in your repository root** — no subfolder.
2. Commit and push.
3. In Streamlit Cloud, set **Main file path** to `app.py` (lowercase).
4. Reboot the app so it reinstalls dependencies.

```
your-repo/
├── app.py              ← main file — point Streamlit Cloud here
├── data_layer.py
├── scanner.py
├── scoring.py
├── readiness.py
├── risk_calc.py
├── technical.py
├── backtest.py
├── portfolio.py
├── stops.py
├── storage.py
├── requirements.txt
├── .gitignore
└── tests/              ← optional, local development only
```

**The modules import each other by plain name**, so they must be siblings of
`app.py`. If they end up inside a subfolder, imports break. If an old copy remains
at the root while new files sit in a subfolder, Python loads the *old* one and your
changes appear to do nothing.

### Confirming the right version is live

Under the app title you should see:

```
Build 2026-09-19-b6 ... · data persisted to SQLite
```

If that line is missing, the deployed code is not the code you just uploaded.

## The strategy — Trend Retrace

The default scanner strategy, as specified by the trader.

**Long** (short is the exact reverse):

1. **4H trend** — the last two closed 4H candles each made a higher high and a higher low.
2. **1H confirmation** — the last closed 1H candle is bullish.
3. **5m entry** — wait for price to retrace to a previous support on the 5m chart; enter there with a limit order.

**Re-entry after a stop-out:**

1. Wait for a closed 4H candle in the trade's direction.
2. On the 5m, wait for a retrace to support and re-enter **50%**.
3. Wait for another closed 1H candle in the trade's direction, then add the remaining **50%**.

**Not specified by the trader — adjustable defaults in the sidebar:**

- **Stop loss:** below the 5m support by 3x the 5m ATR (alternatively, below the last closed 4H candle). A 1x buffer was tested and rejected: trades finished in about 12 minutes, before any 1H candle could close, so the second 50% could never be added.
- **Take profit:** 3x the risk, and never below 3:1. A higher multiple does not create an edge on its own: on random data the win rate falls to exactly the break-even rate (33% at 2R, 25% at 3R, 20% at 4R).
- **Order expiry:** an unfilled limit is cancelled after a day, or if the 4H trend reverses.

**Scoring** is a checklist count — 4H trend 4 points, 1H confirmation 3, 5m support entry 3 — so a setup meeting all three rules scores 10/10. It is not a probability of profit.

Only **closed** candles count. Exchanges return the still-forming candle as the latest row; it is dropped, because a 1H candle that is green mid-hour can close red.

The original 10-point confluence strategy remains available from the sidebar for comparison.

## Tabs

| Tab | Purpose |
|---|---|
| 🌍 Market | Live prices with explicit data-health status (LIVE / DELAYED / STALE / UNAVAILABLE), bar interval, timestamps, retry |
| 🎯 Scanner | Auto-derives the setup checklist from real 1D / 4H / 1H structure; score and readiness kept separate |
| 📋 Positions | Open positions, portfolio risk, concentration flags, stop management with never-widen enforcement |
| 🧮 Risk | Position sizing with leverage, margin, fees, slippage, liquidation warnings, contract specs |
| 📓 Journal | Persistent trade log with CSV export |
| 🔁 Backtest | No-look-ahead replay engine with in/out-of-sample split and full metrics |

## Data persistence

Journal entries and positions are stored in a local SQLite file (`trading_data.db`),
so they survive page refreshes, reruns and app sleeps.

**On Streamlit Cloud the filesystem is ephemeral** — the database is wiped whenever
the app reboots or redeploys (including every code push). Export your journal to CSV
after any session that matters. Running locally gives you genuinely permanent storage,
since the file lives on your own disk.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Local running also avoids sharing a rate-limited IP with other Streamlit Cloud users,
which was part of the original Yahoo Finance data problem.

## Tests

```bash
python -m unittest discover -s tests
```

Expected: `Ran 133 tests ... OK`. All tests use mocks and synthetic data — no network
access or live market data required.

## Why `curl_cffi` is a dependency

Yahoo Finance tightened bot detection on its data endpoints, causing intermittent
"no price data" failures across all instrument types. `curl_cffi` lets `yfinance`
impersonate a real browser's TLS fingerprint, which is the current recommended fix.
The data layer uses it when available, falls back gracefully when not, and retries
transient failures before reporting them. If a fetch still fails, the app reports
UNAVAILABLE and blocks trade-readiness rather than showing a substituted price.

## Honest limitations

- **No exchange connection.** No order placement, status, or account sync.
- **The scanner is rule-based structural analysis, not discretionary chart reading.**
  "Liquidity" means equal-high/low clustering — there is no order-flow or
  volume-profile data behind it. A "sweep" is a close-based rule.
- **4H candles are resampled from 1H**, because the provider serves no native 4H
  interval. These can differ slightly from an exchange's own 4H candles.
- **The backtester replays signals correctly but does not contain the scanner's
  multi-timeframe strategy.** The included moving-average crossover is an example
  used to validate the engine's mechanics.
- **Liquidation warnings are simplified** — they ignore maintenance margin curves,
  funding rates, and exchange-specific liquidation engines.
- **Free equity data is exchange-delayed** by nature; the app labels it DELAYED
  rather than pretending it is live.
- **Evidence marked ❔ could not be evaluated** and scores zero. It is never assumed.

## Long-run profitability (expectancy)

"Profitable over the long run" means positive **expectancy**: the average result
per trade, across many trades, is above zero. It does not mean individual trades
don't lose. At 3:1, break-even is a 25% win rate; a system winning 30% of trades
makes money while still losing 7 trades in 10.

After a backtest, each setup in the scanner is labelled with its evidence —
**Proven +**, **Proven −**, **Unproven** or **Too few** — taken from the most
specific group of backtested trades with enough data (that coin and direction,
then that direction, then all trades). A filter can hide everything not proven
positive. Evidence is stamped with the settings it was measured under and is
not used if the stop, target or strategy change afterwards.

The Risk tab and backtest results show what a win rate feels like to trade:
typical and bad-case losing streaks and drawdowns. At a genuinely profitable
30% win rate and 1% risk, a run of ~10 losses and a ~13% drop from a peak are
typical over 100 trades — normal, not a sign the strategy has broken.

## Learning from trades

Every trade — backtested or forward-tracked — carries a snapshot of its setup at
signal time: 4H trend move, retrace depth to entry, stop width, 1H candle
strength, volatility, trading session and direction. The learner looks for kinds
of setup that lost, using the earlier 70% of trades, and adopts a lesson only if
it also held on the later 30% it never saw.

On synthetic tests it recovered a planted losing condition in 23 of 30 datasets
(finding a threshold of 2.73 against a true 2.7), and invented a false lesson
from pure noise in 4 of 60. When nothing survives the unseen-trade check, it
changes nothing and says so.

Adopted lessons can filter the scanner and are flagged on trade plan cards. It
learns only to skip losing kinds of setup; it doesn't invent new rules. Lessons
should be re-learned as trades accumulate, since markets change.

## Logging the trades you take

Every recommended setup — from a universe scan or a single-coin check — has an
**I took this trade** toggle. It records the trade in the journal as Open, with
your actual fill, the plan, and the setup snapshot the learner uses. Tick off
"I followed the rules" if you deviated; only rule-following trades teach the
strategy by default.

When the trade closes, complete it on the Journal tab: choose whether it hit the
take profit, hit the stop, or was closed manually, and correct the exit price,
entry, or final stop and target if needed. The result is recorded in R against
your **original** stop.

Completed trades feed the same learner as backtests and forward tracking, with
the same unseen-trade check. **Export the journal before every app update** —
Streamlit wipes the database on redeploy — and restore it with the import box.

## High-volume coins outside the top 20

The universe scan can rank coins by their **24-hour volume on Hyperliquid**
instead of by market cap, skipping the largest coins (top 20 by default) and
anything below a minimum volume. It uses Hyperliquid's own volume — the
liquidity you would trade into — and its own candles and mark prices, so coins
not listed on Binance can still be scanned. Results show 24h volume, open
interest and hourly funding.

Hyperliquid limits request rates, so candle requests are paced at one per
1.5 seconds (about 4.5 seconds per coin). Some coins trade in bundles of 1,000
(e.g. kPEPE); their prices are per 1,000 and are labelled so. Tickers from this
source are prefixed "HL:" so tracking and the journal always fetch from the
same place.

Caution: high volume away from the top coins often reflects a sharp move, news
or a pump. These markets are more volatile, and the strategy has not been
backtested on them.

## Reviewing an open trade

Each open trade on the Journal tab has a **Review this trade** button. It
re-checks the trade against your rules using fresh, closed candles:

- **Thesis** — is the 4H trend still in your direction, paused, or reversed?
- **Progress** — current R, and the best and worst reached since entry
- **Time** — whether it has gone nowhere for longer than the stale threshold
- **1H** — whether the last closed 1H candle runs against the trade
- **Structure** — whether a newer swing allows a *tighter* stop

It recommends **Hold**, **Tighten the stop**, **Consider closing early**, or
**Close — the reason you entered no longer holds**, and shows the cost of closing
now against the stop and the target. It never suggests widening a stop.

The backtest measures whether the review's early exits actually help, by
replaying history with and without them. Caution: in a 3:1 strategy many
winners go sideways first, so closing stale trades can cut off the wins that
pay for the losses. A broken 4H thesis is the clearer signal — though it is
rare, because most trades end within a few hours while a 4H reversal needs
at least eight hours to form.

## Fear & Greed, watchlists and the position calculator

**Fear & Greed** appears in the sidebar (0 = extreme fear, 100 = extreme greed).
CoinMarketCap's index is used when `COINMARKETCAP_API_KEY` is set in secrets;
otherwise the free Alternative.me feed, which needs no key. It is a whole-market
mood gauge and says nothing about any individual setup — so rather than acting
on it, the app records the band with every setup, letting the learner test
whether setups taken in greed or fear actually perform differently.

**My watchlist** is a scanner mode: type coins (commas, full names or tickers)
and it matches them against Hyperliquid's live market list. Old tickers and
names resolve via aliases (rndr → RENDER, stacks → STX, immutable → IMX).
Anything it can't match confidently is reported with suggestions rather than
guessed, because scanning the wrong coin looks identical to scanning the right
one.

**The position calculator** accepts the stop either as a price or as a
percentage from entry, and estimates the isolated-margin liquidation price. It
warns when a stop sits beyond liquidation — where the position would be closed
out before the stop could protect it. Liquidation figures are estimates:
exchanges raise maintenance margin for larger positions, so confirm on the
venue.

## Market tools

A dark theme with colour used to carry meaning — grades, data status, verdicts
and direction — always paired with text so nothing depends on colour alone.

The **Market tools** tab adds whole-market context:

- **Fear & Greed** with a gauge, source and daily direction
- **Altcoin season index** — how many of the top 50 coins beat Bitcoin over 90
  days (75+ altcoin season, 25 or below Bitcoin season)
- **Funding & open interest** on Hyperliquid — who is paying whom, annualised,
  with crowding flags and a turnover ratio
- **RSI heatmap** across 1h, 4h and 1d for the highest-volume markets

None of these is a trade signal and none has been shown to improve the
strategy. They are context. The Fear & Greed band is recorded with every setup
so the learner can test whether it actually matters.

Not included: whale tracking and liquidation feeds. Both need paid data or
per-wallet scraping — Hyperliquid exposes positions only for addresses you
already know, and there is no free liquidation stream — so building them would
mean showing something that looks like whale data but isn't.

## Look and market tools

The app uses a dark trading-desk theme (`.streamlit/config.toml`) with colour
that carries meaning rather than decoration — grades, data status, verdicts and
direction each have a fixed colour, always paired with a word or symbol so the
meaning survives without colour.

**Market tools** tab:

- **Fear & Greed** with a gauge (CoinMarketCap if a key is set, otherwise the
  free Alternative.me feed)
- **Altcoin season index** — how many of the top 50 coins beat Bitcoin over 90
  days; 75+ is conventionally "altcoin season", 25 or less "Bitcoin season"
- **Funding & open interest** on Hyperliquid — who is paying whom, crowding
  flags, open interest and turnover
- **RSI heatmap** across 1H, 4H and 1D

None of these is a trade signal and none has been shown to improve the
strategy. The Fear & Greed band is recorded with each setup so the learner can
test whether it matters.

**Not implemented:** whale tracking and liquidation feeds. Both need paid data
or per-wallet scraping — Hyperliquid exposes positions only for addresses you
already know, and there is no free liquidation stream. Building something that
looked like whale data without being it would be worse than leaving it out.

Reward:risk can be displayed as `3.00` (reward:risk) or `1:3` (risk:reward) —
the same trade written two ways. The scanner table shows essentials only by
default, with a toggle for every column.

## Bybit, 24h change and the coin glossary

**Bybit** is a selectable crypto data source (prices, candles, 24h change,
open interest, funding) and the default venue filter. Note that Bybit blocks
requests from some regions including the US, and Streamlit Cloud is US-hosted,
so these calls usually fail there — the app says so and falls back to
Binance/Kraken. Running the app locally from a region Bybit serves works
normally. Bybit's alternate domain (api.bytick.com) is tried automatically.

**The Market tab** shows the 24-hour change in both percent and dollars,
coloured green or red, alongside the price.

**"What coins do"** is a glossary: one plain-English line per coin covering 72
projects across 12 categories, searchable and filterable, ordered by market
cap. Descriptions are written from general knowledge and do not update
themselves, so verify anything before acting on it. Any other ticker can be
looked up live from CoinGecko, which returns the project's own description —
useful for what a coin does, not for whether it is any good.

## Laddered (scale-in) entries

Optional, and offered only when the 5m chart shows more than one confirmed
support below price — spaced at least 0.6 ATR apart and within 8 ATR. A single
support means a single entry; extra levels are never invented.

**The whole ladder risks one trade's risk.** Sizes are solved so that if every
level fills and the stop is then hit, the loss equals your chosen risk — not
that amount times the number of levels. Filling part of the ladder risks less.

The stop sits beyond the furthest level, so it is further from price than a
single entry would need: that is the cost of laddering. The gain is a better
average entry and a better chance of being filled at all; the loss is being
under-sized when price tags only the first level and runs.

Sizing can be equal across levels or weighted towards the better prices. The
backtest models single entries only, so there is no evidence yet that laddering
helps this strategy — it is never the default.
