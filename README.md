# LS-AS A+ Setups

Streamlit app for the LS-AS strategy:
- Market filter: All / Stocks / Crypto / Commodities / Forex
- Weekly screen: $ volume, expanding ATR, within 2% of 4H/Daily liquidity
- Live prices (auto-refresh) and trade levels: entry, stop loss, TP1, TP2, reward:risk, position size
- 🟢 LIVE SETUP = a 15M sweep + MSS + FVG has formed (exact levels)
- Only A+ setups are shown (all checks must pass); others are listed as filtered out

## Deploy (free)
1. Create a public GitHub repo and upload `app.py` and `requirements.txt`
   (replace the old files if updating).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, set main file to `app.py`, click **Deploy**.

Data: Yahoo Finance (can be delayed). Educational tool, not financial advice.
