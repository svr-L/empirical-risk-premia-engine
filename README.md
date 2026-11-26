# Empirical Risk Premia Engine — v0.12.0 (Equity priced-factor challengers and automatic macro; Rates v0.9 retained)

## NEW v0.12 — Equity residual-priced-risk validation + identified policy announcements

**Start here:** `notebooks/Equity_Factor_Discovery_and_Pricing_v0_12_PORTABLE.ipynb`. This stand-alone Colab notebook embeds **ERP 0.12.0**; no `pip install -e` or local source folder required. It preserves all v0.11 baseline cells (including automatic detection of uploaded `fredgraph*.csv/zip`) and adds:

1. Rolling 120-month estimation of FF/AQR-neutral residual RP factors with annual rebalances, past-only sign alignment, subspace cosine stability, turnover, formation FF beta neutrality, forward factor returns and HAC tests. `gamma=0` is ordinary residual PCA; `gamma=12` adds the formation alpha *mean-pricing* term, a residual adaptation of the augmented-mean RP-PCA objective. **Not** exact published full Lettau–Pelger inference.
2. Prespecified common calendar and heldout test assets for (a) conventional two-pass **Fama–MacBeth** pricing and (b) **two-step HAC-weighted shrinkage GMM/SDF**, both formation-only, both scored against identical monthly OOS realizations. Separately report annualized **OOS mean-pricing RMSE**, formation lambda HAC t-statistics and a **paired 12-month moving-block bootstrap** of differences in monthly loss. Pricing horse race uses *fixed* pre-OOS factor weights; rolling factor returns are deliberately a separate analysis and never retroactively inserted into the training sample.
3. Actual FOMC monetary-policy/news decomposition **MP_median / CBI_median** from [Jarociński–Karadi updated author dataset](https://github.com/marekjarocinski/jkshocks_update_fed) with automatic download and persistent cache. End-of-source months are never filled as zeros. Both same-month factor exposure regressions and **one-calendar-month-lag predictive** regressions plus a **cross-sectional FMB shock-price diagnostic** against FF controls. A **genuine CPI release surprise** may be added using pre-announcement `actual,consensus` release-level observations, but no CPI surprise is fabricated from FRED data.

**Important limitations:** the FMB is not Shanken/weak-factor corrected, GMM weighting is shrunk (no standard J test), overlapping sorted portfolios are not independent stock universes, event-study shocks require high-frequency identification/sign restrictions, publicly provided survey-based CPI release surprises are not automatically available here, and historically revised FF/FRED data are not announcement-vintage PIT. Neither lower residual error nor risk price t-statistics prove causal priced risk or profitable TAA rotation. **No new real-market results in this release:** execute notebook in Colab to obtain them.

To run offline: upload source `shocks_fed_jk_m.csv` to Colab root or set `JK_LOCAL_PATH='/content/.../shocks_fed_jk_m.csv'`; `FRED_LOCAL_PATH` remains supported. Output: `erp_equity_v012_results/` including 7 new rolling/pricing CSVs, shock mapping / cross-sectional lambda CSVs when available, macro and v0.11 reports. See `EQUITY_V012_RESEARCH.md` and `RELEASE_VALIDATION_V012.md`.

---

## v0.11 archived — what to run

**Run `notebooks/Equity_Factor_Discovery_and_Macro_v0_11_PORTABLE.ipynb`**.
The source is fully embedded; it executes the v0.10 Equity baseline, then:

1. formation-residual eigenvalue circular-shift null with max-eigenvalue test and formation subspace stability;
2. predeclared pricing-weighted residual PCA (**RP-PCA-inspired, not exact published RP-PCA**), forward FMP premia with HAC/Holm, holdout regularized linear SDF/GMM, and a **Giglio–Xiu-style three-pass point-estimate diagnostic** (not robust inference);
3. automatic FRED CPI, industrial production, 2Y/10Y Treasury, Baa credit spread and VIX observations; FF/RPC contemporaneous economic-change mapping and strictly lagged expanding expected-return forecasts.

**Where to obtain the data:** downloader runs automatically with caching. If blocked, use the unmodified [six-series FRED original CSV/ZIP](https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL%2CINDPRO%2CDGS10%2CDGS2%2CBAA10Y%2CVIXCLS) as `FRED_LOCAL_PATH` (no manual CSV merging). Kennedy French and original AQR files continue to load using the existing Equity module, with AQR workbook fallback. The macro changes are **observed variations, not identified structural economic surprises**; FRED revised data are not historical release vintages.

**Scientific details and exact limitations:** `EQUITY_V011_RESEARCH.md`; `README_v011_QUICKSTART.md`; `RELEASE_VALIDATION_V011.md`. The v0.10 baseline and Rates v0.9 source code are preserved unchanged. The new real-data v0.11 analyses have **not yet been executed in Colab**; passing synthetic tests does not constitute evidence of priced risk premia.

---


## Scope and two active research workstreams

**Equity original v0.2.3 standard factors + new v0.10 residual discovery** and **Rates v0.9.0 (unchanged in this release)** live in the same modular repository and use the same `erp` snapshot/export conventions. The new Rates layer **adds**, never overwrites, the existing v0.8/v0.2.5 research forecasts.

**Equity:** canonical US FF3/FF5 family-specific factor definitions, Carhart/MOM, published AQR BAB and QMJ long-short factors; 25 Size/BM, 30 Industries, 25 Size/OP, 25 Size/INV, 25 Size/MOM test portfolios, pooled 130, HAC/GRS, FMB/identification diagnostics, spanning, OOS exposure attribution and factor snapshots. Equity standard pricing remains an unchanged benchmark; the NEW Equity residual research implementation is in `src/erp/equity_residual.py` and `notebooks/Equity_Residual_Factor_Research_v0_10_PORTABLE.ipynb`. Next Equity research: reconcile FMB/traded factor prices and intercept; condition on test-universe identification, then add priced-factor challenger GMM/SDF and conditional factor returns.

**Rates:** risk extraction on 1Y–10Y Kim–Wright zero curve; funded Level/Slope/Curvature factor portfolios; exact cross-factor neutrality, zero-coupon carry + curve repricing; legacy v0.8 Fama–Bliss, Cochrane–Piazzesi-style, BASE_STATE, FB_STATE, CP_STATE, unconditional and carry-only; 3/6/12-month forecasting with monthly execution; `FEDFUNDS` and CPIAUCSL/INDPRO/T10YIE original input conventions. **PCA is a yield-risk representation, not proof of a priced risk premium.**

### v0.10 Equity: priced residual risk research

This release adds training-only residual PCA for FF6/BAB/QMJ, factor-neutral gross-normalized self-financing candidate portfolios, strict post-training HAC/Holm mean and spanning-alpha tests, universe-heldout and time-heldout pricing/attribution, one-step regularized linear SDF/GMM on held-out assets, and OPTIONAL macro-shock interpretation, one-step nontraded macro SDF/GMM, and lagged-state expected-return forecasts. **No real-data new Equity results have yet been generated and there is no alpha or new risk price claim**. See `EQUITY_RESIDUAL_RESEARCH.md` for exact statistical conditions, CSVs, important sample-overlap/tradability limits and unimplemented methods.

### v0.9 Rates: why and what is new

The v0.2.5 real-data run showed FB OOS predictive gains but conditional rotation did not beat static/carry benchmarks. Prior model families had incomparable forecast samples (STATE vs non-STATE). The v0.9 layer implements:

1. **Strictly matched forecast origins per horizon.** Intersect all included forecast models (original five, `DIRECT_FB`, `DIRECT_CP`), `UNCONDITIONAL`, `CARRY_ONLY`, and fully observed returns for **all three factors**. Score the *same H-month total funded excess-return target* across models, rather than comparing different targets or calendars.
2. **Paired inference.** For each date, average the three factor squared-loss gains `L_unconditional - L_model`; HAC test with `max(6,H-1)` lags, and circular moving-block bootstrap (`max(12,2H)` months). Forecast origin is `t`, realized overlapping return is `t→t+H`. Report OOS R² and RMSE/MAE by model and factor with sample dates. Approximate p-values/CI are exploratory, multiple comparisons are **not** corrected.
3. **Strictly matched realized strategy months per horizon.** Intersect all original v0.8 and direct-target strategies, evaluate realized next-month returns and cash on exactly the same months. Compare paired annualized net excess spreads versus **both** unconditional and carry-only, using HAC `max(12,H)` lags and monthly block bootstrap. Document cash, gross overlay P&L, transaction cost, turnover, factor/underlying gross and fraction of inactive allocations.
4. **Direct vs decomposed-target challenger.** The original learner is re-used without editing its function: compare `known carry at t + predicted H-month curve repricing` with a *direct regression on H-month total funded excess return*, for FB and CP. Same factors, training cutoffs, ridge penalty, prediction horizon, portfolio constraints, execution protocol and costs. The direct learner's generic internal `curve_mean_h` field denotes the direct **TOTAL** return; the interface uses `total_mean_h` consistently for scores and strategies.
5. **CP first-stage diagnostics.** Every 12th forecast origin, use only data whose H-month outcome was known by t; report condition number of standardized 1Y–5Y forward-rate matrix and coefficient sensitivity to **predeclared ridge penalties**. This is *diagnostic only*: the original v0.8 CP forecasting engine is not re-tuned with OOS outcomes.
6. **No-future-imputation CPI/IP** in the wrapper: `pct_change(12, fill_method=None)`; no alteration to the 20+ legacy function definitions covered by exact-source regression tests.

**Critical boundaries:** The v0.9 run cannot be claimed successful on *historical real data* until the new notebook is executed. Existing historical v0.2.5 results are **not recalculated** here. OOS refers to label chronology and fixed PCA calibration for the post-2002 experiment; current revised macro vendors do not provide point-in-time publication vintages. Risk-PC portfolios are mechanically priced in-sample only via realized returns; standalone premium identification (RP-PCA/GMM/P-Q models) remains a separate research frontier. Do not interpret statistical loss gain as trading alpha.

## Usage

### Colab standalone

- **Rates research:** `notebooks/Rates_Premia_v0_9_PORTABLE.ipynb` (embedded ERP 0.9; same curve/funding/macro workflow as v0.2.5, followed by the new strict common-sample analysis).
- **Equity standard benchmark:** `notebooks/Equity_Premia_v0_10_PORTABLE.ipynb` (embedded ERP 0.9, original Equity research cells/behavior unchanged).

**NEW Equity research:** `notebooks/Equity_Residual_Factor_Research_v0_10_PORTABLE.ipynb` with self-contained ERP source (no repo install), predeclared sample/universes, all core tests and CSV export. Optional economic shock and monthly lagged state panels are never invented or silently substituted.

Rates notebook preserves `AS_OF='2026-10-31'`, `HORIZONS=(3,6,12)` and five-model full-grid with macro inputs. If FRED cannot be reached from Colab, upload the *original* combined FRED CSV/ZIP with CPIAUCSL, INDPRO, T10YIE; no substitute variables. For fully offline funding, upload original FEDFUNDS and Kim–Wright Board curve. Supply genuine monthly macro levels (not transformations). The project never silently replaces zero-coupon yields with par yields or option-implied rates.

Rate new knobs:
```
V09_DIRECT_MODELS=('FB','CP')
V09_BOOTSTRAP_REPS=399
V09_BOOTSTRAP_SEED=202609
```

### Package use

```bash
pip install -e '.[test,notebook]'
pytest -q
python scripts/offline_smoke.py # synthetic Equity + funded Rates integration
```

API:
```python
from erp import RatesData, RatesPremiaEngine, RatesEmpiricalBondPremiaEngine, analyze_rates_v09
# rates = RatesPremiaEngine(RatesData.from_fred(...)); snapshot = rates.fit(...)
# research = RatesEmpiricalBondPremiaEngine(rates).fit(snapshot,macro_raw=macro,...)
# v09 = analyze_rates_v09(rates,research,macro_raw=macro,
#                         direct_models=('FB','CP'),min_train=60,ridge=1.,cost_bps=5.)
# v09.export('rates_v09_reports')
```

### Export schema

The legacy `research.export` retains all `rates_v08_*.csv`; `v09.export()` separately produces:

| CSV | Meaning |
|---|---|
| `rates_v09_common_samples.csv` | Common forecast-origin and realized P&L spans for each horizon |
| `rates_v09_predictive.csv` | Common-target pooled OOS R², RMSE, HAC and bootstrap mean loss gain |
| `rates_v09_predictive_by_factor.csv` | Individual Level, Slope, Curvature OOS R² and diagnostics |
| `rates_v09_predictive_losses_by_date.csv` | Auditable paired per-date mean loss gains |
| `rates_v09_rotation.csv` | Matched Sharpe/CAGR/MaxDD/turnover |
| `rates_v09_rotation_spreads.csv` | Paired net strategy gains against unconditional and carry-only |
| `rates_v09_signal_execution.csv` | Overlay gross P&L, paid costs, turnover, active share |
| `rates_v09_cp_first_stage.csv` | CP standardized predictor conditioning, ridge sensitivity |
| `rates_v09_direct_forecast_*` | Auditable date×factor direct return forecasts |
| `rates_v09_manifest.json` | Protocol, horizon and vintage limitations |

## Next research milestones

**Rates:** Based on *actual v0.9* common-window results, investigate whether simple FB is robust at H=3/6/12 and why forecast skill does not convert into dynamic allocation gains. Once matched experiment is closed, add pricing-oriented factors (RP-PCA, SDF/GMM or affine P/Q) without discarding PCA risk controls.

**Equity:** Now add predeclared residual PCA/FMP as traded premia factors and the 130 test portfolios; compare `FMB λ`, traded factor means and model-implied pricing across distinct test universes, checking FMB cross-section intercept/rank/dispersion, and subsequently add robust SDF/GMM. **Fama–MacBeth and traded factor return means are not interchangeable unconditional parameters** unless the model identifies the traded factors appropriately.

Alpha Slicer and Tail-Aware Alpha Referee remain *separate* research questions and repositories; consume common factor data/metadata via exports rather than sharing their internal certification logic with this engine. Rates premia rotation is a candidate input for TAA, not an achieved trading outcome.

## v0.12.2 — rolling economic follow-up
Use `notebooks/Equity_Factor_Discovery_and_Pricing_v0_12_2_PORTABLE.ipynb`. See `EQUITY_V0122_RESEARCH.md` and `RELEASE_VALIDATION_V0122.md`. Dedicated common-calendar economic mapping and pricing for FF/AQR and all six RP rolling returns; FRED upload validation and fallback. Previous discovery methods and Rates source are unchanged.
