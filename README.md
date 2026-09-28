# LS-AS Weekly Asset Screener

Streamlit app that screens the LS-AS universe (index futures/ETFs, forex, commodities,
crypto, mega-caps) for $ volume, ATR expansion and proximity to 4H/Daily liquidity,
then recommends which asset to trade this week.

## Deploy (free)
1. Create a public GitHub repo and upload `app.py` and `requirements.txt`.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, set main file to `app.py`, click **Deploy**.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

Data: Yahoo Finance (may be delayed). Educational tool, not financial advice.
