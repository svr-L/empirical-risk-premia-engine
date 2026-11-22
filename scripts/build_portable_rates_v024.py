"""Generate the single-file Colab Rates v0.2.4 with the real ERP library embedded."""
from pathlib import Path
import io,zipfile,base64,nbformat
ROOT=Path(__file__).resolve().parents[1]
mem=io.BytesIO()
with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for f in (ROOT/'src'/'erp').rglob('*.py'):
        z.write(f,f.relative_to(ROOT/'src'))
encoded=base64.b64encode(mem.getvalue()).decode('ascii')
startup=f'''# Portable ERP v0.2.4 — actual Python modules packaged inside this notebook.
import io, zipfile, base64, sys, subprocess, importlib.util
from pathlib import Path
runtime_root=Path.cwd()/'erp_v024_notebook_runtime'
(runtime_root/'src').mkdir(parents=True,exist_ok=True)
_PAYLOAD={encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_PAYLOAD))) as archive:
    archive.extractall(runtime_root/'src')
for mod,pkg in {{'numpy':'numpy','pandas':'pandas','scipy':'scipy','statsmodels':'statsmodels','requests':'requests','openpyxl':'openpyxl'}}.items():
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
sys.path.insert(0,str(runtime_root/'src'))
for _key in list(sys.modules):
    if _key=='erp' or _key.startswith('erp.'): del sys.modules[_key]
from erp import __version__, RatesData, RatesPremiaEngine, RatesEmpiricalBondPremiaEngine, load_v08_macro_csv, export_snapshot
import pandas as pd
from IPython.display import display
print('ERP:',__version__,'| Rates v0.8 forecasting layer + v0.2.3 funded adapter')
'''
n=nbformat.v4.new_notebook()
n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
m=nbformat.v4.new_markdown_cell
c=nbformat.v4.new_code_cell
n.cells=[
 m('''# Empirical Risk Premia Engine v0.2.4 — Rates: full v0.8 forecasting layer

**Recovered research pipeline:** Federal Reserve Kim–Wright fitted zero curve → vol-standardized PCA (risk exposures) → exact funded factor-mimicking portfolios → *carry and curve repricing* → **Fama–Bliss / Cochrane–Piazzesi-style empirical bond-premia forecasts** → 3/6/12M targets → mean-confidence rotation, monthly execution, transaction costs, OOS prediction & strategy evaluation.

Preserves five historical predictor families **BASE_STATE, FB, FB_STATE, CP, CP_STATE** and **UNCONDITIONAL / CARRY_ONLY** benchmarks. The v0.8 predictor, signal and backtest functions are copied unchanged into `erp.rates_research` and covered by source-parity tests. The funded Rates adapter retains v0.2.3 cash/no-stale-funding safeguards.

**Crucial:** *Risk PCA is NOT treated as proof that Level/Slope/Curvature are independently priced shocks.* Factor portfolio returns are forecast targets. FB/CP predictor variables are not additional traded risk factors.

**Data:** No proxy Treasury par yields, no imputed macro values. Historical revised observations are **not genuine historical publication vintages**.'''),
 c(startup),
 m('''## Configuration and external data

The curve loads from one [Federal Reserve Board Kim–Wright CSV](https://www.federalreserve.gov/data/yield-curve-tables/feds200533.csv) with fallback to a locally uploaded file. Funding is the original **FEDFUNDS** series, now with local/mirror options established in v0.2.3. To avoid another remote provider failure, upload cached originals when possible.

**For all five model families**, also supply a monthly CSV with four columns: `Date,CPI,INDPRO,BREAKEVEN10Y`. CPI and INDPRO are *original level indices* from FRED (`CPIAUCSL`, `INDPRO`); `BREAKEVEN10Y` is the original `T10YIE` value **in percent points**, not decimals. The model builds percent changes and lagging itself, so do **not** upload already-differenced predictors. If left `None`, the two valid non-macro families **FB and CP** still run and the missing three STATE models are explicitly reported, never fabricated. Choose `REQUIRE_FULL_GRID=True` to make missing macro a hard error.'''),
 c('''AS_OF='2026-10-31'
CACHE_DIR=Path('erp_data_cache')
OUT_DIR=Path('erp_v024_output')
CURVE_SOURCE='board'
BOARD_CSV_PATH=None         # e.g. '/content/feds200533.csv'
FUNDING_SOURCE='mirror'    # v0.2.3 fallback; 'local' with CSV below for no network
FEDFUNDS_CSV_PATH=None      # e.g. '/content/FEDFUNDS.csv'
MACRO_CSV_PATH=None         # e.g. '/content/macro_state.csv' (original CPI, INDPRO, T10YIE levels)
REQUIRE_FULL_GRID=False    # True requires original macro files; all 5 models
HORIZONS=(3,6,12)
MIN_TRAIN=60
RIDGE=1.0
COST_BPS=5.0
COST_GRID=(1.0,5.0,10.0)
'''),
 m('''## 1 — Funded Rates decomposition (preserved v0.2.3)

`as_of` is a strict observation cutoff. The output `snapshot.factor_premia` is the **unconditional mean of funded factor portfolio returns**; the new conditional estimates are in the separate results object.'''),
 c('''data=RatesData.from_fred(
    cache_dir=CACHE_DIR,curve_source=CURVE_SOURCE,
    local_curve_path=BOARD_CSV_PATH,
    local_fedfunds_path=FEDFUNDS_CSV_PATH,
    funding_source=FUNDING_SOURCE,
)
rates=RatesPremiaEngine(data)
snapshot=rates.fit(as_of=AS_OF,pca_train_end='2002-12-31')
print('Observation cutoff:',snapshot.as_of)
print('Funded factor portfolio mean returns (monthly decimals):')
display(snapshot.factor_premia.to_frame('mean_monthly'))
print('PCA explained variance (risk, NOT risk premium):')
display(rates.pca.explained_variance_ratio.to_frame('fraction'))
'''),
 m('''## 2 — Original v0.8 empirical bond-premia research grid

Forecast targets use actual **H-month excess bond returns**, realized cash over H months, and *funded carry known at forecast date*; curve repricing is the residual. Training labels for a forecast made at `t` are from origins `s <= t-H`, never incomplete overlapping H-month outcomes. Forecasts target 3/6/12 months while **trades are monthly**, exactly as in v0.8.'''),
 c('''macro_raw=load_v08_macro_csv(MACRO_CSV_PATH) if MACRO_CSV_PATH else None
research=RatesEmpiricalBondPremiaEngine(rates).fit(
    snapshot=snapshot,macro_raw=macro_raw,horizons=HORIZONS,
    min_train=MIN_TRAIN,ridge=RIDGE,cost_bps=COST_BPS,
    cost_sensitivity=COST_GRID,require_full_grid=REQUIRE_FULL_GRID,
)
print('Models requested:',research.metadata['models_requested'])
print('Models executed:',research.metadata['models_run'])
print('Models skipped for missing original macro data:',research.metadata['models_skipped_for_missing_macro'])
print('Forecast counts (date × factor):',[(str(k),len(v)) for k,v in research.forecasts.items()])
'''),
 m('''## 3 — Classic Fama–Bliss 12M diagnostic & predictive diagnostics

Horizon-specific forecast diagnostics compare the model's **curve-repricing** forecasts to expanding unconditional means. `OOS_R2_vs_uncond` can be negative even when a model has nonzero correlation. Do not compare models on different samples without re-aligning; a fully matched common-sample horse race belongs to v0.9.'''),
 c('''if not research.fb_12m_diagnostics.empty:
    print('Fama–Bliss canonical 12M diagnostics (HAC t):')
    display(research.fb_12m_diagnostics)
print('OOS predictive diagnostics, by horizon and model (factor averages):')
if len(research.predictive_diagnostics):
    display(research.predictive_diagnostics.groupby(['horizon','model'])[['corr','OOS_R2_vs_uncond','MAE','n']].mean().round(4))
'''),
 m('''## 4 — Rotation with funded exposures, monthly execution and 5-bp cost

Exactly the v0.8 mean-confidence signal: estimated mean uncertainty is not conflated with next realized-return volatility. Factor allocations are volatility-scaled and capped; underlying maturity gross exposure is capped separately. Results contain cash return, factor overlay return and turnover costs.'''),
 c('''print('Performance matrix (CAGR, Sharpe, MaxDD):')
if len(research.strategy_results):
    display(research.strategy_results[['CAGR','Sharpe','MaxDD','AvgMonthlyTurnover']].round(4))
print('Conditional vs horizon-matched unconditional strategy, classic and HAC t:')
display(research.conditional_value_add.round(4))
print('Cost sensitivity (gross exposure and turnover accounted):')
if len(research.cost_sensitivity):
    display(research.cost_sensitivity.pivot_table(index=['horizon','model'],columns='cost_bps',values='Sharpe').round(3))
'''),
 m('''## 5 — CP coefficients and current *historical* allocations

These allocation dates are **the last origins with realized H-month targets in the retrospective v0.8 protocol**, not live deployment recommendations for the latest curve date. This distinction is preserved intentionally during source parity; latest-date deployable signals can be developed separately with their own no-leakage tests.'''),
 c('''display(research.cp_coefficients.tail(8))
display(research.latest_allocations.tail(14))
'''),
 m('''## 6 — Export common snapshot and full v0.8 research result

The core snapshot remains compatible with the downstream Alpha Slicer/Referee factor library; new CSV files give TAA access to historical predictions, factor targets, weights and realized strategy returns. No change is made to those downstream repositories.'''),
 c('''root=OUT_DIR/'rates_v024'
snapshot_dir=export_snapshot(snapshot,root/'core_factor_snapshot')
rates.export_auxiliary(snapshot_dir)
research_dir=research.export(root/'empirical_bond_premia')
print('Core exported to:',snapshot_dir)
print('FB/CP research exported to:',research_dir)
print('Files:',len(list(research_dir.glob('*'))))
'''),
 m('''## Validation limits and next research release

v0.2.4 is a **source-parity port**, not an improvement claim. Same forecast families, original v0.8 windows, standardized ridge and CP stage, horizon target calendar, and cash/carry decomposition. Macro-based models need original CPI/INDPRO/T10YIE observations; no automatic model substitution. Original v0.8 notebook is bundled in the source repo for review. Regression tests include exact source correspondence of historical functions and synthetic five-family end-to-end integration. **Historical-data numerical parity on the exact original FRED vintage remains unverified** because that complete original data snapshot has not been supplied.

v0.9 research agenda: align common samples across all models, investigate CP collinearity and the pricing-vs-repricing target, robust HAC/bootstrap comparisons, and bring forecast origins to the latest date without using incomplete realized labels.''')
]
nbformat.validate(n)
p=ROOT/'notebooks'/'Rates_Premia_v0_2_4_PORTABLE.ipynb'
nbformat.write(n,p)
print(str(p),p.stat().st_size)
