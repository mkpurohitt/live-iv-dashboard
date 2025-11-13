# app.py
from flask import Flask, jsonify, request, render_template
from threading import Thread
import time
import math
from datetime import datetime
import pandas as pd
from kiteconnect import KiteConnect
from scipy.stats import norm
import copy
from apscheduler.schedulers.background import BackgroundScheduler #new
import threading#new
import os
import glob
import numpy as np 

previous_day_iv_lookup = {}
app = Flask(__name__)

# ------- Config --------
API_KEY = "kite api for data "
ACCESS_TOKEN = "kite access token generated"
INSTRUMENTS_CSV = "instruments.csv"

BATCH_SIZE = 200
POLL_INTERVAL = 1
SLEEP_BETWEEN_BATCHES = 0.1
MANDATORY_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX", "BANKEX"]
RISK_FREE_RATE = 0.065

# ------- Shared data --------
live_iv_data = []
data_lock = threading.Lock()#new 
previous_straddle_iv = {}

# ------- Initialize Kite Connect -------
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# Load instruments
inst = pd.read_csv(INSTRUMENTS_CSV)
if 'expiry' in inst.columns:
    inst.loc[:, 'expiry'] = pd.to_datetime(inst['expiry'], errors='coerce')

opts = inst[inst['segment'].str.contains("OPT", na=False)].copy()
opts.loc[:, 'expiry'] = pd.to_datetime(opts['expiry'], errors='coerce')

underlyings = sorted(opts['name'].dropna().unique().tolist())
for idx in MANDATORY_INDICES:
    if idx not in underlyings and idx in inst['name'].values:
        underlyings.append(idx)

print(f"Detected {len(underlyings)} F&O underlyings (including indices if present).")

# ------- Helpers -------


def load_previous_day_iv():
    """Load the latest iv_snapshot CSV before today, if it exists."""
    global previous_day_iv_lookup
    previous_day_iv_lookup.clear()

    # Find all saved IV snapshot files
    files = sorted(glob.glob("iv_snapshot_*.csv"))
    if not files:
        print("No previous IV snapshot files found.")
        return

    # Get the latest one that's not from today
    today_str = datetime.now().strftime("%Y%m%d")
    prev_files = [f for f in files if today_str not in f]
    if not prev_files:
        print("No earlier IV snapshot found.")
        return

    latest_file = prev_files[-1]
    print(f"Loading previous day IV data from {latest_file}")
    try:
        df_prev = pd.read_csv(latest_file)
        for _, row in df_prev.iterrows():
            u = row.get("underlying")
            if pd.isna(u):
                continue
            prev_ce = row.get("ce_iv_pct")
            prev_pe = row.get("pe_iv_pct")
            prev_straddle = row.get("straddle_iv")
            previous_day_iv_lookup[u] = {
                "prev_ce_iv": prev_ce,
                "prev_pe_iv": prev_pe,
                "prev_straddle_iv": prev_straddle
            }
    except Exception as e:
        print("Error loading previous IV snapshot:", e)

# Call it right after defining
load_previous_day_iv()

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def safe_quote_batch(keys):
    out = {}
    if not keys:
        return out
    try:
        res = kite.quote(keys)
        out.update(res)
        return out
    except Exception as e:
        if len(keys) == 1:
            print("quote error single:", keys[0], e)
            return {}
        mid = len(keys) // 2
        out.update(safe_quote_batch(keys[:mid]))
        out.update(safe_quote_batch(keys[mid:]))
        return out

def bs_price(option_type, S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, S - K) if option_type == 'CE' else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'CE':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility_newton(option_price, S, K, T, r, option_type, init_vol=0.25, max_iter=100, tol=1e-6):
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    sigma = init_vol
    for i in range(max_iter):
        price = bs_price(option_type, S, K, T, r, sigma)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        diff = price - option_price
        if abs(diff) < tol:
            return max(sigma, 1e-6)
        if vega < 1e-12:
            return None
        sigma = sigma - diff / vega
        if sigma <= 0:
            sigma = sigma * 0.5
    return None

def compute_time_to_expiry_years(expiry_ts):
    now = datetime.now()
    expiry_dt = expiry_ts.to_pydatetime() if hasattr(expiry_ts, 'to_pydatetime') else expiry_ts
    secs = (expiry_dt - now).total_seconds()
    if secs <= 0:
        return 1.0 / 365.0
    return secs / (365.0 * 24 * 3600)

# -------- Your IV fetch function (runs continuously) --------
def fetch_live_iv_data(shared_data_container):
    global previous_straddle_iv
    while True:
        start = datetime.now()
        records = []

        try:
            spot_keys = [f"NSE:{u}" for u in underlyings]
            spot_quotes = {}
            for batch in chunks(spot_keys, BATCH_SIZE):
                spot_quotes.update(safe_quote_batch(batch))
                time.sleep(SLEEP_BETWEEN_BATCHES)

            missing = [u for u in underlyings if f"NSE:{u}" not in spot_quotes]
            if missing:
                bse_keys = [f"BSE:{u}" for u in missing]
                for batch in chunks(bse_keys, BATCH_SIZE):
                    spot_quotes.update(safe_quote_batch(batch))
                    time.sleep(SLEEP_BETWEEN_BATCHES)

            atm_meta = []
            for u in underlyings:
                try:
                    sym_opts = opts[opts['name'] == u]
                    if sym_opts.empty:
                        continue
                    exps = sorted(sym_opts['expiry'].dropna().unique())
                    if not exps:
                        continue
                    nearest_exp = exps[0]
                    expiry_df = sym_opts[sym_opts['expiry'] == nearest_exp]
                    strikes = sorted(set(expiry_df['strike'].astype(float).tolist()))
                    if len(strikes) < 2:
                        strike_step = strikes[0] if strikes else 1.0
                    else:
                        diffs = [round(strikes[i+1] - strikes[i], 8) for i in range(len(strikes)-1)]
                        positive = [d for d in diffs if d > 0]
                        strike_step = min(positive) if positive else 1.0
                    atm_meta.append({'underlying': u, 'expiry': nearest_exp, 'expiry_df': expiry_df, 'strike_step': strike_step})
                except Exception:
                    continue

            option_keys = []
            mapping = []
            for m in atm_meta:
                u = m['underlying']
                spot_key_nse = f"NSE:{u}"
                spot_val = None
                if spot_key_nse in spot_quotes:
                    spot_val = spot_quotes[spot_key_nse].get('last_price')
                else:
                    bkey = f"BSE:{u}"
                    if bkey in spot_quotes:
                        spot_val = spot_quotes[bkey].get('last_price')
                if not spot_val:
                    continue

                step = float(m['strike_step']) if m['strike_step'] else 1.0
                mult = round(spot_val / step)
                atm_strike = round(mult * step, 8)

                expiry_df = m['expiry_df']
                ce_rows = expiry_df[(expiry_df['strike'] == atm_strike) & (expiry_df['instrument_type'] == 'CE')]
                pe_rows = expiry_df[(expiry_df['strike'] == atm_strike) & (expiry_df['instrument_type'] == 'PE')]

                if ce_rows.empty or pe_rows.empty:
                    continue

                ce_row = ce_rows.iloc[0]
                pe_row = pe_rows.iloc[0]
                exchange_ce = ce_row['exchange'] if 'exchange' in ce_row else 'NFO'
                exchange_pe = pe_row['exchange'] if 'exchange' in pe_row else 'NFO'
                ce_key = f"{exchange_ce}:{ce_row['tradingsymbol']}"
                pe_key = f"{exchange_pe}:{pe_row['tradingsymbol']}"

                option_keys.append(ce_key)
                option_keys.append(pe_key)

                mapping.append({
                    'underlying': u,
                    'spot': spot_val,
                    'expiry': m['expiry'],
                    'atm_strike': atm_strike,
                    'ce_key': ce_key,
                    'pe_key': pe_key
                })

            option_quotes = {}
            for batch in chunks(option_keys, BATCH_SIZE):
                option_quotes.update(safe_quote_batch(batch))
                time.sleep(SLEEP_BETWEEN_BATCHES)

            for meta in mapping:
                try:
                    cek = meta['ce_key']
                    pek = meta['pe_key']
                    if cek not in option_quotes or pek not in option_quotes:
                        continue
                    ce_q = option_quotes[cek]
                    pe_q = option_quotes[pek]
                    ce_ltp = ce_q.get('last_price', 0)
                    pe_ltp = pe_q.get('last_price', 0)
                    S = float(meta['spot'])
                    K = float(meta['atm_strike'])
                    T = compute_time_to_expiry_years(meta['expiry'])
                    r = RISK_FREE_RATE

                    ce_iv = None
                    pe_iv = None

                    try:
                        sigma_ce = implied_volatility_newton(ce_ltp, S, K, T, r, 'CE')
                        if sigma_ce:
                            ce_iv = round(sigma_ce * 100, 2)
                    except Exception:
                        ce_iv = None

                    try:
                        sigma_pe = implied_volatility_newton(pe_ltp, S, K, T, r, 'PE')
                        if sigma_pe:
                            pe_iv = round(sigma_pe * 100, 2)
                    except Exception:
                        pe_iv = None

                    # Calculate straddle IV trend: up=1, down=-1, no change=0
                    prev_straddle = previous_straddle_iv.get(meta['underlying'])
                    straddle_iv = None
                    trend = 0
                    if ce_iv is not None and pe_iv is not None:
                        straddle_iv = round((ce_iv + pe_iv) / 2, 2)
                        if prev_straddle is not None:
                            if straddle_iv > prev_straddle:
                                trend = 1
                            elif straddle_iv < prev_straddle:
                                trend = -1
                        previous_straddle_iv[meta['underlying']] = straddle_iv

                    records.append({
                        'underlying': meta['underlying'],
                        'spot': S,
                        'atm_strike': K,
                        'expiry': meta['expiry'].strftime('%Y-%m-%d') if hasattr(meta['expiry'], 'strftime') else str(meta['expiry']),
                        'ce_ltp': ce_ltp,
                        'pe_ltp': pe_ltp,
                        'ce_iv_pct': ce_iv,
                        'pe_iv_pct': pe_iv,
                        'straddle_iv': straddle_iv,
                        'trend': trend,
                        'timestamp': datetime.now().isoformat(),
                        'prev_ce_iv': previous_day_iv_lookup.get(meta['underlying'], {}).get("prev_ce_iv"),
                        'prev_pe_iv': previous_day_iv_lookup.get(meta['underlying'], {}).get("prev_pe_iv"),
                        'prev_straddle_iv': previous_day_iv_lookup.get(meta['underlying'], {}).get("prev_straddle_iv")
                    })
                except Exception:
                    continue

            # Update shared container atomically
            #shared_data_container.clear()
            #shared_data_container.extend(records)
            with data_lock:#
                shared_data_container.clear()#
                shared_data_container.extend(records)#new


        except Exception as ex:
            print("Exception in IV fetch thread:", ex)

        elapsed = (datetime.now() - start).total_seconds()
        to_sleep = max(0, POLL_INTERVAL - elapsed)
        time.sleep(to_sleep)
        ########

def save_iv_data():
    with data_lock:
        if not live_iv_data:
            print("No IV data to save at", datetime.now())
            return
        df = pd.DataFrame(live_iv_data)
        filename = f"iv_snapshot_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved IV data to {filename} at {datetime.now()}")
#######

# -------- Flask Routes --------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ivdata")
def api_ivdata():
    sort_order = request.args.get("sort", "desc")
    df = pd.DataFrame(copy.deepcopy(live_iv_data))
    df = df.replace([np.nan, np.inf, -np.inf], None)
    data_copy = df.to_dict(orient="records")
    data_copy = [d for d in data_copy if d['straddle_iv'] is not None]
    data_copy.sort(key=lambda x: x['straddle_iv'], reverse=(sort_order == "desc"))
    return jsonify(data_copy)


# -------- Run background IV fetch thread and start Flask --------
if __name__ == "__main__":
    thread = Thread(target=fetch_live_iv_data, args=(live_iv_data,), daemon=True)
    thread.start()

    ######new
    # Setup scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(save_iv_data, 'cron', hour=15, minute=30)  # 3:30 PM daily
    scheduler.start()
    #########
    app.run(port=5000,host="0.0.0.0",debug=True, threaded=True)
    #app.run(debug=True, threaded=True)
