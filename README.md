# Overnight Gap-Risk Strategy

**Hold SPY overnight (close to open); stand aside on the nights a gap-risk model flags as dangerous.**

The organizing finding behind this project is that the overnight *return* is close to
efficient (not predictable), while the overnight *downside risk* is predictable and
concentrated at the one-day horizon. So the strategy does not try to time direction. It
holds the well-documented overnight equity premium on most nights and steps aside on the
riskiest ~17% of nights, as scored out-of-sample by a gradient-boosted classifier.

The result keeps most of a naive always-long book's return while materially improving the
risk profile: a higher Sharpe, a smaller maximum drawdown, and a worst single night about a
third the size.

> This is a research and **paper-trading** project. No real capital is deployed. All figures
> below are walk-forward out-of-sample or live paper results, and are stated with their
> limitations.

---

## Results (walk-forward out-of-sample, 2016 to 2026, 2,661 nights)

The model's training is mildly seed-sensitive (see [Reproducibility](#reproducibility-and-robustness)),
so the strategy figures are reported as the median over a 50-seed sweep, with the naive
always-long book as the fixed baseline:

| Metric              | Naive always-long | Strategy (median of 50 seeds) | Strategy range      |
|---------------------|:-----------------:|:-----------------------------:|:-------------------:|
| Sharpe ratio        | 0.71              | **0.93**                      | 0.84 to 1.07        |
| Max drawdown        | -29.4%            | **-20.1%**                    | -23.0% to -16.6%    |
| Worst single night  | -10.45%           | **-3.41%**                    | -3.99% to -3.41%    |
| Terminal wealth     | 2.21x             | 2.00x                         | 1.86x to 2.21x      |
| Nights invested     | 100%              | 83%                           | 82% to 83%          |

**The strategy beats the naive book's Sharpe in 100% of the 50 seeds.** It gives up a little
return (2.21x to ~2.0x terminal wealth) in exchange for a higher Sharpe, a drawdown roughly 9
points shallower, and a worst night cut from -10.45% to -3.41%. The worst-night reduction is
essentially seed-invariant; the direction of the effect always holds.

The `gap-risk` classifier itself scores **~0.76 walk-forward AUC** on the P(overnight < -1%)
target, versus 0.64 for a term-structure (contango) rule alone.

Run `python backtest.py` for a single deterministic draw (Sharpe 0.89) and
`python robustness.py` to reproduce the full seed distribution above.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python backtest.py
```

Expected output (deterministic; may vary by ~0.02 Sharpe across CPU architecture, see
Reproducibility below):

```
OOS 2016-01-04→2026-08-04  (2661 nights)
  NAIVE always-long : ann +8.17%  Sharpe 0.71  maxDD -29.4%  worst -10.45%  final 2.21x
  STRATEGY (filtered): ann +6.60%  Sharpe 0.89  maxDD -21.7%  worst -3.41%  final 1.95x  invested 82%
  exported 553 weekly points -> data/strategy_curve.json
```

Only `numpy`, `pandas`, `xgboost`, and `pyarrow` are needed to run the backtest. The bundled
`data/overnight_features.parquet` makes it fully self-contained: no external data download,
no API keys, no broker account.

---

## Methodology

The backtest is **deployment-faithful**: it mirrors exactly how the live model makes a
decision, so the out-of-sample numbers are honest about what a real deployment would have
done.

- **Target.** A binary "dangerous night" label: next-session overnight return below -1%
  (a 4.8% base rate). Classification, not regression, because return direction is not
  predictable but tail risk is.
- **Walk-forward, expanding window.** For each year Y from 2016 on, the model trains only on
  data strictly before Y and is evaluated on Y. Nothing from the test year (or later) informs
  the model or the threshold.
- **Threshold set on train only.** Each fold stands aside on the nights whose predicted risk
  exceeds the 84th percentile of that fold's *training* scores. This reproduces the live
  policy of staying flat on roughly the riskiest 16% of nights, with no lookahead.
- **Model.** XGBoost, depth-3, 150 trees, heavily regularized (`min_child_weight=20`,
  `reg_lambda=5`, `subsample`/`colsample=0.8`). Gradient-boosted trees decisively beat a
  logistic model (0.74 AUC but only 0.33 strategy Sharpe) and an MLP (0.55 AUC) on this small
  tabular dataset. AUC is not P&L, so the model is selected on economics, not just AUC.
- **Features (17).** VIX level and 5-day change, VIX-futures term-structure slope and roll
  yield, 20-day realized vol and variance risk premium, 5- and 20-day SPY returns, distance
  from the 50- and 200-day moving averages, prior intraday and overnight returns, a 5-day HYG
  (credit) return, and day-of-week dummies.

---

## Reproducibility and robustness

`backtest.py` is pinned for determinism (`random_state=0`, `n_jobs=1`, `tree_method="exact"`),
so it prints the same result on every run on a given machine: Sharpe 0.89. But a single seed
is one sample. XGBoost uses row and column subsampling, and the stand-aside threshold sits near
a percentile boundary of the risk scores, so a few near-threshold nights flip between seeds and
carry outsized weight. The Sharpe and drawdown therefore move from seed to seed; the direction
of the effect does not.

`robustness.py` runs the identical walk-forward under 50 seeds and reports the distribution:

- Sharpe: median **0.93**, range 0.84 to 1.07
- Max drawdown: median **-20.1%**, range -23.0% to -16.6%
- Worst single night: **-3.41%** (essentially seed-invariant)
- Beats the naive baseline (Sharpe 0.71) in **100% of seeds**

So the honest central estimate is Sharpe ~0.93 and max drawdown ~-20%. A flattering
0.89-to-1.07 Sharpe draw (with a shallower -16% drawdown) and a -21.7% seed-0 draw both sit
inside this distribution near its edges; neither on its own is "the" number. What is robust:
the sign of the effect (it always beats naive) and the worst-night reduction to -3.41%.

## Repository layout

```
overnight-gap-risk/
├── backtest.py                     # the walk-forward backtest, single deterministic seed (run this)
├── robustness.py                   # same backtest across 50 seeds -> the result distribution
├── data/
│   ├── overnight_features.parquet  # 3,069 nights, 2014-2026; engineered features + label
│   └── strategy_curve.json         # weekly equity + drawdown curves (backtest output)
├── models/
│   ├── gap_risk.ubj                # trained XGBoost model (full-history fit)
│   └── gap_risk.json               # threshold + metadata (feature list, base rate, dates)
├── live/                           # deployment reference (see note below)
│   ├── featurelib.py               # feature construction from the market-data store
│   ├── train.py                    # fits the persisted production model
│   ├── decide.py                   # scores today's features -> HOLD / STAND_ASIDE
│   ├── broker.py                   # Alpaca paper adapter, hard paper-only safety gate
│   ├── run_close.py                # 15:45 ET: market-on-close BUY on HOLD nights
│   ├── run_open.py                 # 09:31 ET: sell the overnight position at the open
│   ├── snapshot.py                 # near-close IBKR snapshot to remove a 1-day feature lag
│   ├── ledger.py                   # append-only trade ledger
│   ├── config.py                   # symbol + target fraction
│   └── .env.example                # template for your own paper keys (real .env is git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

## Data provenance

`data/overnight_features.parquet` is a derived feature matrix built from daily bars for SPY,
the VIX index, the VIX-futures term structure (CBOE settlements unioned with live futures),
and HYG. All inputs are standard market data. The feature construction is in
`live/featurelib.py` for reference. A known data-integrity trap handled during construction:
pre-2021 vendor daily bars carried a dividend-adjusted close against raw open/high/low, which
manufactures a fake overnight drift; contaminated rows are detected via a close-outside-of-[low,high]
test and dropped.

## About the `live/` folder

These modules are the actual code running the strategy in **paper** on Alpaca. They are
included as a reference implementation of the research-to-production path (safety gates,
scheduling, deterministic ledger). They are **not runnable from this repo alone**: they read
from a private market-data pipeline (`~/marketdata`) and require your own Alpaca paper keys
via `live/.env`. The reproducible, self-contained artifact in this repo is `backtest.py`.

`broker.py` will refuse to place an order unless the API key begins with `PK` (Alpaca's paper
prefix) and the endpoint is the paper URL, so it cannot touch a live account by
misconfiguration.

## Limitations

- Paper trading, not real capital, and live for a short window. This demonstrates a clean
  research-to-production path with real safety rails, not a track record.
- The model handles *predictable* danger. Some overnight tails are exogenous and carry no
  vol-state warning (for example the August 2024 yen-carry unwind, which struck while the
  curve was still in contango). Those belong to structural hedges, not this filter.
- SPY trades only in regular hours, so the close-to-open gap is captured via market-on-close
  and market-on-open orders rather than a 23-hour instrument.

## License

MIT. See [LICENSE](LICENSE).

Author: **Alexander Scales**
