# How to run the app on your computer

First time: about 10 minutes. After that: 10 seconds.

---

## WINDOWS

**Step 1 — Install the Python Install Manager**
Go to python.org/downloads and click the yellow Download button.
Open the downloaded file and click Install.
(Windows now gives you a "manager" rather than Python itself. That's correct —
you install Python through it in the next step. There is no longer an
"Add to PATH" tick box.)

**Step 2 — Unzip the app**
Right-click the app zip in Downloads, choose "Extract All", then Extract.
Open the folder that appears. You should see app.py inside it.

**Step 3 — Open the black command window in that folder**
Click once in the white address bar at the top of the folder.
Type:  cmd
Press Enter. A black window opens.

**Step 4 — Install Python itself (first time only)**
Type this and press Enter:

    py install 3.13

If it says it's already installed, that's fine — carry on.

**Step 5 — Install what the app needs (first time only)**
Type this and press Enter, then wait a few minutes:

    py -3.13 -m pip install -r requirements.txt

**Step 6 — Start the app**
Type this and press Enter:

    py -3.13 -m streamlit run app.py

Your browser opens with the app. Done.

**To stop it:** close the black window.
**Next time:** only Step 3 and Step 6.

---

## MAC

**Step 1 — Install Python**
Go to python.org/downloads, click the yellow Download button, open the file
and click through the installer.

**Step 2 — Unzip the app**
Double-click the zip in Downloads. A folder appears with app.py inside.

**Step 3 — Open Terminal in that folder**
Right-click the folder, choose Services, then "New Terminal at Folder".

**Step 4 — Install what the app needs (first time only)**
Type this and press Enter, then wait a few minutes:

    python3 -m pip install -r requirements.txt

**Step 5 — Start the app**
Type this and press Enter:

    python3 -m streamlit run app.py

Your browser opens with the app. Done.

**To stop it:** close the Terminal window.
**Next time:** only Step 3 and Step 5.

---

## IF SOMETHING GOES WRONG

**The prompt shows >>> and you get "SyntaxError: invalid syntax"**
You're inside Python itself, not the command prompt. Type this and press Enter:

    exit()

You'll return to a prompt that looks like a folder path ending in > , such as
C:\Users\You\Downloads\bull-run-strategy-streamlit>
That's where the commands go. Quick test:
  >>>        means Python — commands like py won't work here
  C:\...>    means the command prompt — everything in this guide goes here

**"can't open file app.py" or "No such file or directory"**
The command window is pointing at the wrong folder. Look at the path shown
before the > sign: it must be the folder containing app.py. To fix it, open
that folder, click the white address bar at the top, type cmd and press Enter.

**"py is not recognised" (Windows)**
The install manager didn't finish. Install it again from python.org/downloads,
then close and reopen the black window — it only picks up new commands when
it starts.

**"pip is not recognised"**
Don't type pip on its own. Use the longer commands above that start with
py -3.13 -m  (Windows) or python3 -m  (Mac). They work even when pip isn't on
the PATH, which is normal with the new install manager.

**"No module named streamlit"**
Step 5 (Windows) or Step 4 (Mac) didn't finish. Run it again and read the last
few lines for the reason.

**Nothing opens in the browser**
Type this into your browser's address bar:  localhost:8501

**"Port already in use"**
The app is already running in another window. Close it, or add a different
port to the end of the start command:  --server.port 8502

**A command seems stuck**
Installing can take several minutes with no visible progress. Leave it. If you
need to stop something, press Ctrl and C together.

---

## WHY IT'S WORTH DOING

1. Bybit works — real prices from the exchange you actually trade on.
2. Binance works — 60+ days of history for backtesting instead of 17.
3. Your journal stops being wiped every time you update the app.
4. Nothing runs on anyone else's server.
