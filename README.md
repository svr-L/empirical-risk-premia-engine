# Empirical Risk Premia Engine — v0.2.4

Multi-asset **pricing, factor-risk decomposition, identification and empirical conditional-premia research**. The shared Equity and Rates modules from v0.2.3 remain intact, with the original **Rates v0.8 Fama–Bliss / Cochrane–Piazzesi-style empirical bond-premia layer ported as an actual Python package**, not a notebook-only appendix.

## Immediate use

**Colab:** open `notebooks/Rates_Premia_v0_2_4_PORTABLE.ipynb`. It bundles the actual modules in a compressed self-contained payload. Upload the official Treasury zero-curve and FEDFUNDS CSVs if your Colab cannot reach public endpoints. This version uses the **same funding/curve loader as v0.2.3**, no unverified new data provider.

**Python:** from repo root:

```bash
python -m pip install -e '.[test]'
python scripts/run_rates_v024.py --as-of 2026-10-31 --output outputs/rates_v024
pytest -q
```

For the complete **five-model macro grid**, include original levels:

```bash
python scripts/run_rates_v024.py \
  --curve-csv /path/to/feds200533.csv \
  --funding-csv /path/to/FEDFUNDS.csv --funding-source local \
  --macro-csv /path/to/original_macro.csv --require-full-grid
```

The macro CSV needs `Date,CPI,INDPRO,BREAKEVEN10Y`. CPI is original **CPIAUCSL** (index level), INDPRO is original **INDPRO** (index level), BREAKEVEN10Y is original **T10YIE** (percent per year; the code divides by 100). For consistency with historical v0.8 it uses 12-month percentage changes of CPI/INDPRO shifted by one month; observed Fed Funds shifted by one month; and current breakeven T10YIE. Do not substitute different macro series or already-computed YoY changes. The `examples/` template contains column names only, **no fabricated data**. If missing, the code runs FB/CP and explicitly skips BASE_STATE/FB_STATE/CP_STATE; `--require-full-grid` instead throws an error.

## Precisely what is now integrated

1. **Risk / exposure:** Kim–Wright zero-coupon 1Y–10Y curve, lagged covariance, pre-OOS standardized PCA Level/Slope/Curvature, analytic duration sensitivity of zero-coupon excess returns.
2. **Funded risk factor portfolios:** exact cross-factor-orthogonality of 2Y–10Y portfolios, time-varying covariance/normalization, explicit cash funding, realization lag, historical returns decomposed into funded carry/roll and curve repricing. These are ***candidate tradable exposures***, not proof that each PC is independently priced.
3. **Multi-horizon bond excess returns (original v0.8):** H=3,6,12 months zero-coupon repricing and actual compounded cash over H; start-date factor weights held fixed for construction of the H-month **target**; known-horizon funded carry and target curve repricing separated.
4. **Predictor families:** `BASE_STATE`, `FB` (Fama–Bliss forward spreads), `FB_STATE`, `CP` (walk-forward, original five-forward first-stage common-return predictor), `CP_STATE`. The CP object is a *predictor* of premia, not automatically a fourth priced shock. Two additional matched-horizon baselines: `UNCONDITIONAL`, `CARRY_ONLY`.
5. **Walk-forward:** for an H-month observation originating at `s`, its target may enter the training data for time `t` only if `s+H <= t`. Features standardised exclusively inside each training fold. Original ridge settings, confidence about **expected mean**, time-varying volatility scaled/capped factor allocation, next-month holdings, underlying gross cap, and one-way turnover costs reproduced.
6. **Evaluation:** 12M standard Fama–Bliss regression diagnostic (HAC t), forecast-vs-unconditional OOS R², MAE/correlation, Sharpe/CAGR/MaxDD/turnover, original pairwise strategy t-stat plus extra HAC t-stat, 1/5/10bp cost grid, CP coefficient history endpoint, historical last allocations.
7. **Cross-repo exports:** `export_snapshot` remains the stable standard factor/exposure schema for Alpha Slicer/Referee; `BondPremiaResearchResult.export()` adds all horizon-specific predictions and realized portfolio outcomes for Premia Rotation TAA.

## API

```python
from erp import RatesData, RatesPremiaEngine, RatesEmpiricalBondPremiaEngine, load_v08_macro_csv

data=RatesData.from_fred(curve_source='board',funding_source='mirror')
rates=RatesPremiaEngine(data)
snapshot=rates.fit(as_of='2026-10-31')
macro=load_v08_macro_csv('/path/to/original_macro.csv')  # omit for FB/CP-only run
research=RatesEmpiricalBondPremiaEngine(rates).fit(
    snapshot, macro_raw=macro, horizons=(3,6,12),min_train=60,
    require_full_grid=True)
research.export('outputs/rates_v024')
```

## Reproducibility, scope and known limitations

- **Parities verified:** 22 extracted original v0.8 functional definitions match notebook source byte-for-byte in AST source text; solver, zero-coupon logic and return decomposition remain tested against v0.8 at function level; 32 Python tests passed (including deterministic synthetic five-family, full 3/6/12M fifteen-model grid and as-of tests). Historical v0.8 executed notebook source is included under `legacy/rates_v0_8_reference.ipynb`.
- **Historical result numerical parity not asserted:** current Board/FEDFUNDS and macro snapshots may be revised, or macro may be absent. To reproduce *the exact historical metrics of v0.8*, rerun both pipelines on the **same cached original source data**, same end date and model settings. The change in the empirical sample prevents comparing published summaries at face value.
- **Original v0.8 sample convention retained:** the retrospective forecast grid only evaluates forecast dates with a realized H-month target in the dataset, thus it does **not** produce an actionable latest-date 3/6/12M prediction for dates in the final H months. This is a historical-protocol replication. A truly deployable last-date forecast, common comparison sample and pairwise bootstrap belong in v0.9.
- **No hidden macro substitution:** missing CPI/INDPRO/T10YIE does not result in invented STATE predictors. This also means `FB`/`CP`-only results must not be described as a validated full five-family live run.
- **Macro vintages:** `as_of` removes future rows, but FRED historical revised values are not true point-in-time released vintages; release lags need formal treatment in v0.9.
- **Conditional trade superiority not established:** v0.8 previously found some FB predictability without robust conditional strategy outperformance. This port changes no economic hypotheses or signal weights.
- The original v0.7 1-month macro-state experiment remains preserved inside the v0.8 notebook under `legacy/`, but the *integrated research API* exposes the **v0.8 3/6/12-month grid**, as specified for this release. No new forward-looking distribution / Kalman / SPCA is introduced here.

## Structure

```
src/erp/core.py               common factor & pricing snapshot
src/erp/equity.py             FF/AQR Equity factor models
src/erp/rates.py              funded Treasury factor portfolios
src/erp/rates_research.py     original v0.8 empirical forecast grid + wrapper
src/erp/data.py               data loaders from 0.2.3
scripts/run_rates_v024.py     CLI
scripts/build_portable_rates_v024.py
notebooks/Rates_Premia_v0_2_4_PORTABLE.ipynb
legacy/rates_v0_8_reference.ipynb
tests/test_rates_v08_port.py
```

**Runtime note:** the exact v0.8 walk-forward regressions refit repeatedly for every origin/horizon/factor. A long-sample 15-model stress run may be computationally expensive; the 32-test suite includes a shorter complete-grid smoke test. No accelerated approximations were silently substituted into the source-parity port.
