import os, csv, datetime as dt
LEDGER = os.path.expanduser("~/overnight_bot/logs/ledger.csv")
FIELDS = ["ts","event","asof","decision","p_gap","threshold","symbol","qty","order_id","price","note"]
def record(**row):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    new = not os.path.exists(LEDGER)
    row["ts"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
