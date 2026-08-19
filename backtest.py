"""
Walk-forward backtest of the DEPLOYED overnight strategy:
  long SPY overnight when the gap-risk model says HOLD, flat otherwise.
Deployment-faithful: expanding annual refit; each fold's stand-aside threshold = 84th pct
of that fold's TRAIN scores (exactly how the live model sets it — no lookahead). OOS 2016-2026.
Exports weekly equity + drawdown curves for the HTML tearsheet.
"""
import os, json
import numpy as np, pandas as pd, xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
F   = os.path.join(HERE, "data", "overnight_features.parquet")
OUT = os.path.join(HERE, "data", "strategy_curve.json")
DANGER, STAND_ASIDE = -0.01, 0.16

def model(spw):
    # random_state + single-thread + exact tree method => deterministic and reproducible
    # across machines (the stand-aside threshold sits near a percentile boundary, so small
    # numerical differences would otherwise flip a few nights and move the Sharpe/drawdown).
    return xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="logloss", tree_method="exact",
        random_state=0, n_jobs=1, verbosity=0)

def stats(x, inv=None):
    x = x.dropna(); eq = (1+x).cumprod()
    return dict(ann=x.mean()*252, sharpe=x.mean()/x.std()*np.sqrt(252),
        maxdd=(eq/eq.cummax()-1).min(), worst=x.min(), final=eq.iloc[-1],
        hit=(x != 0).mean() and (x[x != 0] > 0).mean(), invested=inv)

def main():
    df = pd.read_parquet(F)
    for d in (1,2,3,4): df[f"dow_{d}"] = (df["dow"] == d).astype(float)
    df = df.drop(columns=["dow"])
    yret = df.pop("y_overnight_next"); feats = list(df.columns)
    y = (yret < DANGER).astype(int)
    rows = []
    for Y in sorted({d.year for d in df.index}):
        if Y < 2016: continue
        tr, te = df.index.year < Y, df.index.year == Y
        if tr.sum() < 300 or te.sum() == 0 or y[tr].sum() < 10: continue
        spw = (y[tr] == 0).sum()/max((y[tr] == 1).sum(), 1)
        m = model(spw); m.fit(df[tr][feats], y[tr])
        thr = float(np.quantile(m.predict_proba(df[tr][feats])[:, 1], 1-STAND_ASIDE))
        pte = m.predict_proba(df[te][feats])[:, 1]
        for dt_, p, r in zip(df.index[te], pte, yret[te]):
            rows.append((dt_, p < thr, r))
    R = pd.DataFrame(rows, columns=["date", "hold", "ovn"]).set_index("date").dropna()
    naive = R["ovn"]; strat = R["ovn"].where(R["hold"], 0.0)
    sn, ss = stats(naive), stats(strat, R["hold"].mean())
    print(f"OOS {R.index.min().date()}→{R.index.max().date()}  ({len(R)} nights)")
    print(f"  NAIVE always-long : ann {sn['ann']:+.2%}  Sharpe {sn['sharpe']:.2f}  maxDD {sn['maxdd']:.1%}  worst {sn['worst']:.2%}  final {sn['final']:.2f}x")
    print(f"  STRATEGY (filtered): ann {ss['ann']:+.2%}  Sharpe {ss['sharpe']:.2f}  maxDD {ss['maxdd']:.1%}  worst {ss['worst']:.2%}  final {ss['final']:.2f}x  invested {ss['invested']*100:.0f}%")

    # weekly export
    curve = pd.DataFrame(index=R.index)
    curve["naive_eq"] = (1+naive.fillna(0)).cumprod(); curve["strat_eq"] = (1+strat.fillna(0)).cumprod()
    curve["naive_dd"] = curve["naive_eq"]/curve["naive_eq"].cummax()-1
    curve["strat_dd"] = curve["strat_eq"]/curve["strat_eq"].cummax()-1
    wk = curve.groupby([curve.index.isocalendar().year, curve.index.isocalendar().week]).tail(1)
    exp = dict(dates=[str(d.date()) for d in wk.index],
        naive_eq=[round(v, 4) for v in wk["naive_eq"]], strat_eq=[round(v, 4) for v in wk["strat_eq"]],
        naive_dd=[round(v, 4) for v in wk["naive_dd"]], strat_dd=[round(v, 4) for v in wk["strat_dd"]],
        stats=dict(naive={k: round(float(v), 4) for k, v in sn.items() if v is not None},
                   strat={k: round(float(v), 4) for k, v in ss.items() if v is not None}))
    json.dump(exp, open(OUT, "w"))
    print(f"  exported {len(wk)} weekly points -> {OUT}")

if __name__ == "__main__":
    main()
