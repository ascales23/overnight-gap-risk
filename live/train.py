"""
Train the deployment gap-risk model on ALL available history and persist it.
XGBoost (scale-invariant → no scaler to drift). Saves model + threshold + metadata.
Stand-aside threshold = 84th pct of in-sample P(gap) → flat on the ~16% riskiest nights
(the research config that lifted overnight Sharpe 0.71→1.02).
"""
import os, json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from featurelib import build_matrix, FEATURES

MODELS = os.path.expanduser("~/overnight_bot/models"); os.makedirs(MODELS, exist_ok=True)
DANGER = -0.01
STAND_ASIDE_FRAC = 0.16

def main():
    X, yov, _ = build_matrix()
    df = X.join(yov).dropna()
    Xtr, y = df[FEATURES], (df["next_overnight"] < DANGER).astype(int)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="logloss", n_jobs=4, verbosity=0)
    m.fit(Xtr, y)
    p = m.predict_proba(Xtr)[:, 1]
    thr = float(np.quantile(p, 1 - STAND_ASIDE_FRAC))
    m.save_model(f"{MODELS}/gap_risk.ubj")
    meta = dict(features=FEATURES, threshold=thr, danger_thresh=DANGER,
        stand_aside_frac=STAND_ASIDE_FRAC, trained_at=str(pd.Timestamp.now()),
        n=int(len(df)), base_rate=float(y.mean()), insample_auc=float(roc_auc_score(y, p)),
        train_start=str(df.index.min().date()), train_end=str(df.index.max().date()))
    json.dump(meta, open(f"{MODELS}/gap_risk.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print("\nNOTE: insample_auc is optimistic; the trustworthy number is the walk-forward OOS AUC 0.71.")

if __name__ == "__main__":
    main()
