# Strategy Lab

A separate Streamlit app for the slow question: **do these rules actually make
money?** The trading app answers "what should I do today"; this one answers
"is this worth doing at all". They share no data and can't interfere.

## What it does

Runs two strategies over the **same coins, the same daily candles, the same
period and the same costs**, so the only difference between them is the rules.

* **Swing Levels (3:1)** — daily trend, a level touched more than once, stop
  beyond the level, target at the next one. Wins rarely, wins big.
* **Mean Reversion (0.4:1)** — a stretched market at a level, small target,
  wide stop. Wins often, loses big.

## Reading the result

The table shows the win rate next to the rate each strategy needs just to break
even. **A high win rate is not an edge.** At a 0.4:1 target, a market with no
edge at all produces about 71% winners, so 75% is barely above nothing — while
a 3:1 strategy only needs 25%.

It also shows a **luck range**. If the gap between the two is smaller than
that, neither has been shown to be better, and the app says so rather than
picking a winner.

## What it cannot tell you

**The 5-minute strategy.** Only two or three days of 5-minute history is
reachable from a hosted server — far too short to conclude anything. Testing
that needs the app running on your own machine, where months are available.

**The future.** A few hundred days on a handful of coins tells you which rules
fitted that period. Forward tracking in the trading app is the only evidence
that counts.

## Deploying

1. Create a new GitHub repository and upload every file here to its root.
2. On share.streamlit.io, create a new app pointing at that repository with
   the main file set to `app.py`.

Nothing is written or saved, so this app can be redeployed freely without
affecting your trade journal.

## Tests

```
python3 -m unittest discover -s tests
```

347 tests covering the strategies, the data sources and the price checks.
