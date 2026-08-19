"""
Decision brain: build today's features from the live store, score the gap-risk model,
emit HOLD / STAND_ASIDE for tonight's overnight hold. Pure read — places no orders.
"""
import os, json
import pandas as pd, xgboost as xgb
from featurelib import build_matrix, FEATURES

MODELS = os.path.expanduser("~/overnight_bot/models")

def decide():
    meta = json.load(open(f"{MODELS}/gap_risk.json"))
    m = xgb.XGBClassifier(); m.load_model(f"{MODELS}/gap_risk.ubj")
    X, _, close = build_matrix()
    X = X.dropna()
    asof = X.index[-1]; last = X.iloc[[-1]]
    p = float(m.predict_proba(last[FEATURES])[:, 1][0])
    decision = "STAND_ASIDE" if p >= meta["threshold"] else "HOLD"
    stale = int((pd.Timestamp.now().normalize() - asof).days)
    return dict(asof=str(asof.date()), stale_days=stale, p_gap=round(p, 4),
        threshold=round(meta["threshold"], 4), decision=decision,
        spy_close=round(float(close.loc[asof]), 2),
        vix=round(float(last["vix_level"].iloc[0]), 2),
        ts_slope=round(float(last["ts_slope"].iloc[0]), 3),
        roll_yield=round(float(last["roll_yield"].iloc[0]), 2))

if __name__ == "__main__":
    print(json.dumps(decide(), indent=1))
