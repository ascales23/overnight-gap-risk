"""
Shared feature library for the overnight gap-risk strategy.
Same features as the validated research model, but the term structure is built from the
LIVE puller store (marketdata/vix_futures, refreshed daily) unioned with the CBOE history
parquet for depth — so a decision made today uses today's curve, not a stale snapshot.
"""
import glob, os, datetime as dt
import numpy as np, pandas as pd

MD  = os.path.expanduser("~/marketdata")
VXH = os.path.expanduser("~/vol_carry/data/vx_futures_history.parquet")
FEATURES = ["vix_level","vix_chg5","ts_slope","roll_yield","rv20","vrp","ret5","ret20",
            "dist_ma50","dist_ma200","prior_intraday","prior_overnight","hyg_ret5",
            "dow_1","dow_2","dow_3","dow_4"]
MONTH = {"F":1,"G":2,"H":3,"J":4,"K":5,"M":6,"N":7,"Q":8,"U":9,"V":10,"X":11,"Z":12}

def _third_friday(y,m):
    d=dt.date(y,m,1); return d+dt.timedelta(days=(4-d.weekday())%7)+dt.timedelta(days=14)
def _settle_from_code(code):
    m=MONTH[code[2]]; y=2020+int(code[3]); nm=(m%12)+1; ny=y+(1 if m==12 else 0)
    return pd.Timestamp(_third_friday(ny,nm)-dt.timedelta(days=30))

def _daily(sym, typ="equities"):
    fs=sorted(glob.glob(f"{MD}/{typ}/{sym}/1day/*.parquet"))
    df=pd.concat([pd.read_parquet(f) for f in fs])
    df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df=df.drop_duplicates("date").set_index("date").sort_index()[["open","high","low","close"]]
    tol=1e-4
    return df[(df["close"]>=df["low"]*(1-tol))&(df["close"]<=df["high"]*(1+tol))]

def _front_two(long_df):
    """long_df: columns [date, expiry(Timestamp), price] -> per date f1,f2,dte1."""
    rows=[]
    for d,g in long_df[long_df["price"]>0].groupby("date"):
        g=g[g["expiry"]>d].sort_values("expiry")
        if len(g)>=2:
            rows.append((d,g["price"].iloc[0],g["price"].iloc[1],(g["expiry"].iloc[0]-d).days))
    return pd.DataFrame(rows,columns=["date","f1","f2","dte1"]).set_index("date")

def _term_structure():
    # CBOE history (depth)
    c=pd.read_parquet(VXH)
    c["price"]=c["settle"].where(c["settle"].astype(float)>0,c["close"])
    c["date"]=pd.to_datetime(c["trade_date"]).dt.normalize(); c["expiry"]=pd.to_datetime(c["expiry"])
    cboe=_front_two(c[["date","expiry","price"]])
    # LIVE marketdata vix_futures (freshness)
    rows=[]
    base=f"{MD}/vix_futures"
    if os.path.isdir(base):
        for code in os.listdir(base):
            fs=sorted(glob.glob(f"{base}/{code}/1day/*.parquet"))
            if not fs: continue
            df=pd.concat([pd.read_parquet(f) for f in fs])
            df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df["expiry"]=_settle_from_code(code); df["price"]=df["close"]
            rows.append(df[["date","expiry","price"]])
    live=_front_two(pd.concat(rows,ignore_index=True)) if rows else pd.DataFrame()
    # union: live overrides CBOE on overlapping dates (fresher)
    ts=cboe.copy()
    if len(live): ts=pd.concat([cboe.drop(index=live.index.intersection(cboe.index)),live]).sort_index()
    return ts

SNAP = os.path.expanduser("~/overnight_bot/data/today_snapshot.json")

def _apply_snapshot(spy, vix, hyg, ts, sim_store_asof=None):
    """Inject today's near-close snapshot as the latest row IF it is newer than the store
    (so the close job — which fires before the puller's post-close update — sees today)."""
    import json
    if sim_store_asof is not None:
        cut = pd.Timestamp(sim_store_asof)
        spy, vix, hyg, ts = spy[spy.index<=cut], vix[vix.index<=cut], hyg[hyg.index<=cut], ts[ts.index<=cut]
    if not os.path.exists(SNAP):
        return spy, vix, hyg, ts, None
    s = json.load(open(SNAP)); d = pd.Timestamp(s["date"])
    if d <= spy.index.max():
        return spy, vix, hyg, ts, None            # store already has this day
    spy = pd.concat([spy, pd.DataFrame({"open":s["spy_open"],"high":s["spy_high"],
        "low":s["spy_low"],"close":s["spy_close"]}, index=[d])])
    vix = pd.concat([vix, pd.Series({d:s["vix"]}, name="vix")])
    hyg = pd.concat([hyg, pd.Series({d:s["hyg_close"]}, name="hyg")])
    ts  = pd.concat([ts, pd.DataFrame({"f1":s["f1"],"f2":s["f2"],"dte1":s["dte1"]}, index=[d])])
    return spy.sort_index(), vix.sort_index(), hyg.sort_index(), ts.sort_index(), str(d.date())

def build_matrix(sim_store_asof=None):
    """Returns (feats_df[FEATURES], next_overnight Series, spy_close Series)."""
    spy=_daily("SPY"); vix=_daily("VIX","vix_index")["close"].rename("vix"); hyg=_daily("HYG")["close"].rename("hyg")
    ts=_term_structure()
    spy, vix, hyg, ts, injected = _apply_snapshot(spy, vix, hyg, ts, sim_store_asof)
    if injected: print(f"[featurelib] injected snapshot row for {injected}")
    r=pd.DataFrame(index=spy.index)
    r["overnight"]=spy["open"]/spy["close"].shift(1)-1
    r["intraday"] =spy["close"]/spy["open"]-1
    r["cc"]       =spy["close"]/spy["close"].shift(1)-1
    f=pd.DataFrame(index=spy.index)
    f["vix_level"]=vix; f["vix_chg5"]=vix-vix.shift(5)
    f=f.join(ts[["f1","f2","dte1"]])
    f["ts_slope"]=f["f2"]/f["f1"]
    f["roll_yield"]=np.log(f["f1"]/vix)*(252/f["dte1"].clip(lower=1))
    rv20=r["cc"].rolling(20).std()*np.sqrt(252)*100
    f["rv20"]=rv20; f["vrp"]=vix-rv20
    f["ret5"]=spy["close"]/spy["close"].shift(5)-1; f["ret20"]=spy["close"]/spy["close"].shift(20)-1
    f["dist_ma50"]=spy["close"]/spy["close"].rolling(50).mean()-1
    f["dist_ma200"]=spy["close"]/spy["close"].rolling(200).mean()-1
    f["prior_intraday"]=r["intraday"]; f["prior_overnight"]=r["overnight"]; f["hyg_ret5"]=hyg/hyg.shift(5)-1
    wd=f.index.weekday
    for d in (1,2,3,4): f[f"dow_{d}"]=(wd==d).astype(float)
    f=f.drop(columns=["f1","f2","dte1"])
    return f[FEATURES], r["overnight"].shift(-1).rename("next_overnight"), spy["close"]
