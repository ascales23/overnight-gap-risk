"""
Seed-robustness sweep for the overnight gap-risk strategy.

Why: XGBoost with subsample/colsample < 1 is seed-dependent, and the stand-aside threshold
sits near a percentile boundary, so a few near-threshold nights flip between seeds. Picking a
single seed (even a fixed one) reports an arbitrary point. This sweep runs the identical
deployment-faithful walk-forward under many seeds and reports the DISTRIBUTION of outcomes,
which is the honest central estimate plus its spread.

Run: python robustness.py [n_seeds]   (default 50)
"""
import os, sys
import numpy as np, pandas as pd, xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "data", "overnight_features.parquet")
DANGER, STAND_ASIDE = -0.01, 0.16

def run_once(df, feats, yret, y, seed):
    rows = []
    for Y in sorted({d.year for d in df.index}):
        if Y < 2016: continue
        tr, te = df.index.year < Y, df.index.year == Y
        if tr.sum() < 300 or te.sum() == 0 or y[tr].sum() < 10: continue
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss", tree_method="exact",
            random_state=seed, n_jobs=1, verbosity=0)
        m.fit(df[tr][feats], y[tr])
        thr = float(np.quantile(m.predict_proba(df[tr][feats])[:, 1], 1 - STAND_ASIDE))
        pte = m.predict_proba(df[te][feats])[:, 1]
        for dt_, p, r in zip(df.index[te], pte, yret[te]):
            rows.append((dt_, p < thr, r))
    R = pd.DataFrame(rows, columns=["date", "hold", "ovn"]).set_index("date").dropna()
    s = R["ovn"].where(R["hold"], 0.0)
    eq = (1 + s).cumprod()
    return dict(sharpe=s.mean() / s.std() * np.sqrt(252),
                maxdd=(eq / eq.cummax() - 1).min(), worst=s.min(),
                final=eq.iloc[-1], invested=R["hold"].mean())

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    df = pd.read_parquet(F)
    for d in (1, 2, 3, 4): df[f"dow_{d}"] = (df["dow"] == d).astype(float)
    df = df.drop(columns=["dow"])
    yret = df.pop("y_overnight_next"); feats = list(df.columns)
    y = (yret < DANGER).astype(int)

    res = [run_once(df, feats, yret, y, s) for s in range(n)]
    def col(k): return np.array([r[k] for r in res])
    print(f"Seed sweep: {n} seeds, deployment-faithful walk-forward OOS 2016-2026\n")
    print(f"{'metric':<12}{'mean':>9}{'median':>9}{'std':>8}{'min':>9}{'max':>9}")
    for k, fmt in [("sharpe", "{:.2f}"), ("maxdd", "{:.1%}"), ("worst", "{:.2%}"),
                   ("final", "{:.2f}x"), ("invested", "{:.0%}")]:
        c = col(k)
        row = [fmt.format(v) for v in (c.mean(), np.median(c), c.std(), c.min(), c.max())]
        print(f"{k:<12}" + "".join(f"{v:>9}" for v in row))
    sh = col("sharpe")
    print(f"\nNaive baseline Sharpe 0.71 for reference. "
          f"Strategy beats naive in {int((sh > 0.71).mean() * 100)}% of seeds.")

if __name__ == "__main__":
    main()
