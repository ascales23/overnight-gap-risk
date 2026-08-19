"""
Near-close snapshot of today's close-state from IBKR, so the close job decides on TODAY
(not yesterday). Writes ~/overnight_bot/data/today_snapshot.json. Run with the ibkr_puller
venv (has ib_async). Scheduled ~15:38 ET; also works after hours (returns today's finals).
"""
import os, sys, json, datetime as dt
sys.path.insert(0, os.path.expanduser("~/ibkr_puller"))
import config
from ib_async import IB, Contract, Stock, Future, util

OUT = os.path.expanduser("~/overnight_bot/data/today_snapshot.json")

async def daybar(ib, contract, what="TRADES"):
    bars = await ib.reqHistoricalDataAsync(contract, endDateTime="", durationStr="3 D",
        barSizeSetting="1 day", whatToShow=what, useRTH=True, formatDate=2)
    return bars[-1] if bars else None

async def main():
    ib = IB()
    await ib.connectAsync(config.IB_HOST, config.IB_PORT, clientId=47, timeout=15)
    try:
        spy = Stock("SPY", "SMART", "USD"); hyg = Stock("HYG", "SMART", "USD")
        vix = Contract(symbol="VIX", secType="IND", exchange="CBOE", currency="USD")
        await ib.qualifyContractsAsync(spy, hyg, vix)
        det = await ib.reqContractDetailsAsync(Future("VIX", exchange="CFE", currency="USD"))
        today = dt.date.today().strftime("%Y%m%d")
        # monthly VX only (tradingClass 'VX'); weeklies are 'VX01'..'VX53' and lack daily bars
        fut = sorted((d.contract for d in det if d.contract.tradingClass == "VX"),
                     key=lambda c: c.lastTradeDateOrContractMonth)
        front = [c for c in fut if c.lastTradeDateOrContractMonth >= today][:2]
        spy_b = await daybar(ib, spy); vix_b = await daybar(ib, vix); hyg_b = await daybar(ib, hyg)
        f1_b = await daybar(ib, front[0]); f2_b = await daybar(ib, front[1])
        exp1 = dt.datetime.strptime(front[0].lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
        snap = dict(date=str(dt.date.today()), asof=dt.datetime.now().isoformat(timespec="seconds"),
            spy_open=spy_b.open, spy_high=spy_b.high, spy_low=spy_b.low, spy_close=spy_b.close,
            vix=vix_b.close, hyg_close=hyg_b.close, f1=f1_b.close, f2=f2_b.close,
            dte1=(exp1 - dt.date.today()).days, f1_expiry=front[0].lastTradeDateOrContractMonth)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(snap, open(OUT, "w"), indent=1)
        print(json.dumps(snap, indent=1))
    finally:
        ib.disconnect()

if __name__ == "__main__":
    util.run(main())
