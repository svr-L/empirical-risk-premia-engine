#!/usr/bin/env python3
"""Build self-contained Colab notebook from actual ERP v0.11 source files."""
from __future__ import annotations
from pathlib import Path
import base64, gzip, tarfile, io
import nbformat
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'src'/'erp'
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tf:
    for src in sorted(source.glob('*.py')):
        tf.add(src,arcname='erp/'+src.name)
compressed=base64.b64encode(gzip.compress(buf.getvalue(),compresslevel=9)).decode()
bootstrap=f'''import sys,base64,gzip,tarfile,io
from pathlib import Path
EMBEDDED=\"{compressed}\"
RUNTIME=Path('/content/erp_v011_notebook_runtime') if Path('/content').exists() else Path.cwd()/'erp_v011_notebook_runtime'
RUNTIME.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(EMBEDDED))),mode='r:') as tf:
    tf.extractall(RUNTIME/'src',filter='data' if sys.version_info>=(3,12) else None)
sys.path.insert(0,str(RUNTIME/'src'))
import numpy as np,pandas as pd
from erp import (__version__,EquityData,analyze_equity_residuals,
    analyze_equity_factor_extensions,load_equity_fred_macro)
print('Embedded ERP source:', __version__)
'''
md=nbformat.v4.new_markdown_cell
code=nbformat.v4.new_code_cell
n=nbformat.v4.new_notebook()
n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'erp_version':'0.11.0',
    'title':'Equity Latent Priced Risks & Macro Factor Decomposition v0.11'}
n.cells=[
md('''# ERP 0.11 — Equity: residual covariance, pricing-oriented factors, macro decomposition

**Three predeclared research gates:** (1) significance/stability of residual covariance factors; (2) PCA vs *RP-PCA-inspired* pricing-weighted residual factors, linear SDF/GMM and *Giglio–Xiu-style* three-pass point estimates; (3) automatic FRED macro/cross-asset data for FF and residual factor interpretation and lagged conditional forecasting.

**Do not conflate:** residual eigenvalues, mean factor alphas, cross-sectional risk prices, ex-post risk attribution and OOS expected-return predictions. Economic change proxies are not identified shocks. FRED latest revised data are not release-vintage point-in-time series. The GX diagnostic is not the published estimator's robust inference; the RP method is an explicitly simplified *alpha-penalized residual PCA*, not a full publication-level implementation.

Market data: original Kenneth French monthly factors/test assets and AQR BAB/QMJ; FRED `CPIAUCSL, INDPRO, DGS10, DGS2, BAA10Y, VIXCLS` for economic proxies and state predictors. No fabricated data.'''),
code(bootstrap),
md('''## 1. Fixed discovery / test design

First 80 portfolios: 25 Size–B/M, 30 Industries, 25 Size–Momentum. Holdout 50: 25 Size–Profitability + 25 Size–Investment. **Constituent stocks can overlap** despite distinct portfolio definitions. Train through December 2005, OOS from January 2006. `GAMMAS` and `N_PCS` **must not be selected by validation performance**.
'''),
code('''MODEL='FF6_BAB_QMJ'
AS_OF='2026-10-31'
TRAIN_END='2005-12-31'
DISCOVERY=('SIZE_BM_25','INDUSTRY_30','SIZE_MOM_25')
VALIDATION=('SIZE_OP_25','SIZE_INV_25')
N_PCS=5
GAMMAS=(0.,12.,60.) # predeclared gamma=0 is the original PCA baseline
NULL_DRAWS=199
NULL_SEED=7331
MACRO_STATE_LAG_MONTHS=2
AQR_PATHS={} # original AQR workbook fallback if source blocked
FRED_LOCAL_PATH=None # OPTIONAL: original combined FRED CSV/ZIP from browser
CACHE=Path('erp_cache')
OUT=Path('erp_equity_v011_results');OUT.mkdir(parents=True,exist_ok=True)
'''),
md('''## 2. Published Equity data, baseline v0.10 (unchanged)

The baseline remains untouched. Factor portfolios are fixed after formation, not continuously traded at zero cost. Neutrality against model factors holds in training, not necessarily OOS.'''),
code('''data=EquityData.from_public_sources(cache_dir=str(CACHE),aqr=True,aqr_paths=AQR_PATHS)
base=analyze_equity_residuals(data,model=MODEL,discovery_universes=DISCOVERY,
    validation_universes=VALIDATION,train_end=TRAIN_END,as_of=AS_OF,n_pcs=N_PCS)
print('Training/OOS months:',base.metadata['formation_months'],base.metadata['oos_months'])
print('Original residual PC premia:');display(base.fmp_premia_tests.round(5))
print('Baseline mean-pricing errors / attribution:');display(base.holdout_metrics.round(5))
'''),
md('''## 3. Test the existence of synchronous residual structure

Circular-shift each discovery-asset residual independently `NULL_DRAWS` times, preserving each asset's time path and autocorrelation while destroying synchronous cross-asset dependence. The rankwise eigenvalue null and conservative max-eigenvalue FWER comparison test common residual *covariance*, **not priced risks**. Compare principal subspaces between two formation-only halves to assess instability; this is not a formal confidence interval.'''),
code('''ext=analyze_equity_factor_extensions(data,base,n_pcs=N_PCS,null_draws=NULL_DRAWS,
    null_seed=NULL_SEED,rp_gamma=GAMMAS,gx_k=N_PCS)
print('Residual eigenvalue null:');display(ext.eigen_null.round(6))
print('Formation subspace stability:');display(ext.subspace_stability.round(5))
'''),
md('''## 4. Pricing-oriented residual factor challengers + SDF + three-pass

Formation covariance `C` is augmented by `gamma × a a′` using **training OLS alpha vector** `a` rather than zero-mean residual scores. This is an RP-PCA-*inspired* test: not the precise Lettau–Pelger procedure. Each direction is projected into net-dollar and FF/AQR-beta neutral portfolios using training betas. Mean and spanning-alpha tests use OOS months and HAC; Holm correction covers the whole predeclared gamma × PC family. Held-out asset means and SDF/GMM are also scored OOS.

Giglio–Xiu-style three passes: return PCA, cross-sectional lambda from training heldout assets, projection of **observed factors** onto latent factors. It is a point-estimate research diagnostic, not weak-factor robust inference.'''),
code('''print('Predeclared pricing-oriented portfolios:');display(ext.pricing_oriented_premia.round(5))
print('Heldout mean-pricing errors, regularized SDF:');display(ext.pricing_oriented_sdf.round(5))
print('Three-pass observed FF/AQR premia diagnostics:');display(ext.three_pass_observed.round(5))
print('Three-pass validation errors:');display(ext.three_pass_diagnostics.round(5))
'''),
md('''## 5. Download macro & cross-asset proxies automatically from FRED

`CPIAUCSL`: Consumer Price Index; `INDPRO`: Industrial Production; `DGS10` and `DGS2`: Treasury yields; `BAA10Y`: Moody's Baa–10Y Treasury spread; `VIXCLS`: CBOE volatility index. Monthly official observations are resampled without forward fill. Features: log YoY CPI/IP, 10Y–2Y slope, credit spread and log VIX. **First differences are statistical economic changes, not event-study identified macro surprises.**

The loader attempts a combined FRED download, then per-series downloads, then caches. If Colab cannot reach FRED, download the **original** combined CSV from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL%2CINDPRO%2CDGS10%2CDGS2%2CBAA10Y%2CVIXCLS`, place its path in `FRED_LOCAL_PATH`, and re-run this cell. Do not fill unavailable values or substitute synthetic data.'''),
code('''macro=None
try:
    macro=load_equity_fred_macro(cache_dir=CACHE/'macro',path=FRED_LOCAL_PATH,as_of=AS_OF)
    print('FRED source:',macro.provenance['source'])
    print('First complete state month:',macro.provenance.get('first_complete_state_month'))
    display(macro.states.tail(6).round(4));display(macro.changes.tail(6).round(4))
except (ConnectionError,FileNotFoundError,ValueError) as exc:
    print('Economic-data acquisition unavailable:',exc)
    print('Factor discovery/pricing completed; economic decomposition NOT executed.')
'''),
md('''## 6. FF and RPC economic factor mapping + conditional premia

**Contemporaneous interpretation:** regress FF/AQR and RPC returns on *monthly changes* of macro/cross-asset variables using formation dates, then evaluate the estimated mapping on OOS months. A high R² does **not** identify a causal macro risk premium.

**Predictability:** state variables are lagged by two complete calendar months and a Ridge forecast of factor expected returns is refitted expanding-window. Its OOS R² is compared with an expanding unconditional mean. Data are latest revised FRED historical observations, **not point-in-time vintages**. The optional one-step SDF numbers using economic changes are **proxy-priced moment diagnostics**, not validated prices of economically identified macro risks.'''),
code('''macro_result=None
if macro is not None:
    macro_result=analyze_equity_residuals(data,model=MODEL,discovery_universes=DISCOVERY,
        validation_universes=VALIDATION,train_end=TRAIN_END,as_of=AS_OF,
        n_pcs=N_PCS,macro_shocks=macro.changes,state_predictors=macro.states,
        state_lag_months=MACRO_STATE_LAG_MONTHS)
    print('FF + RPC factor exposures to economic changes (NOT risk prices):')
    display(macro_result.macro_exposures.round(5))
    print('Cross-time stability / interpretation diagnostics:')
    display(macro_result.macro_diagnostics.round(5))
    print('Lagged macro-state factor return forecasting, OOS vs unconditional:')
    display(macro_result.state_scores.round(5))
    print('Tentative one-step macro proxy SDF — NOT identified shock risk prices:')
    display(macro_result.macro_sdf_pricing.round(5))
else:
    print('Macro mapping and forecast remain UNESTIMATED; no proxy or synthetic input used.')
'''),
md('''## 7. Audit exports and interpretation

All CSVs are separate. Conclude only when residual eigenvalue significance, OOS pricing, and OOS conditional prediction each survive their own tests. Neither residual R² nor FF6 alpha alone proves a new priced factor. No TAA factor rotation alpha is claimed.'''),
code('''base.export(OUT/'equity_baseline_v010')
ext.export(OUT/'equity_extensions_v011')
if macro is not None:
    macro.export(OUT/'public_FRED_inputs')
    (OUT/'public_FRED_inputs'/'source_metadata.json').write_text(__import__('json').dumps(macro.provenance,indent=2))
    if macro_result is not None:macro_result.export(OUT/'equity_economic_mapping')
print('Saved to:',OUT.resolve())
for x in sorted(OUT.rglob('*.csv')):print(' -',x)
''')]
path=ROOT/'notebooks'/'Equity_Factor_Discovery_and_Macro_v0_11_PORTABLE.ipynb'
nbformat.write(n,path)
print('Created',path, 'cells',len(n.cells), 'payload chars',len(compressed))
