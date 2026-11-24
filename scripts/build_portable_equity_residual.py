"""Build a focused self-contained Equity residual research Colab notebook and refresh Equity benchmark."""
from pathlib import Path
import base64,io,zipfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
mem=io.BytesIO()
with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted((ROOT/'src'/'erp').glob('*.py')):
        z.write(p,p.relative_to(ROOT/'src'))
payload=base64.b64encode(mem.getvalue()).decode()

bootstrap='''# Portable ERP 0.10.0 — standalone source embedded, no package install required.
from pathlib import Path
import io,zipfile,base64,sys,subprocess,importlib.util
RUNTIME=Path.cwd()/'erp_v010_runtime'
(RUNTIME/'src').mkdir(parents=True,exist_ok=True)
PAYLOAD=__PAYLOAD__
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as zf:
    zf.extractall(RUNTIME/'src')
for pkg in ['numpy','pandas','scipy','statsmodels','requests','openpyxl']:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
sys.path.insert(0,str(RUNTIME/'src'))
for name in list(sys.modules):
    if name=='erp' or name.startswith('erp.'):
        del sys.modules[name]
import numpy as np,pandas as pd
from IPython.display import display
from erp import (__version__,EquityData,EquityPremiaEngine,analyze_equity_residuals,
                 export_snapshot,RatesData,RatesPremiaEngine,RatesEmpiricalBondPremiaEngine,
                 load_macro_panel,load_macro_series_files,analyze_rates_v09)
print('Loaded embedded ERP',__version__)
'''.replace('__PAYLOAD__',repr(payload))

n=nbformat.v4.new_notebook()
n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},'erp_version':'0.10.0',
            'title':'Equity Residual Priced Factors Research'}
n.cells=[
 nbformat.v4.new_markdown_cell('''# Empirical Risk Premia Engine — Equity residual factor discovery (v0.10)

**Objective:** search for incremental systematic equity risks after **FF6 + BAB + QMJ**, build factor-neutral self-financing portfolios, test *out-of-sample* factor premia and incremental pricing on held-out test portfolios, and optionally map both known/new factors to economic shocks and forecast with **lagged** state variables.

The **FF6/BAB/QMJ factor benchmarks, Rates v0.9 and all existing CSV definitions remain unchanged** in this release.

**Design discipline:** formation-only PCA; formation-only risk neutrality; no OOS sign flipping or selection of `K`; no use of zero-mean in-sample OLS residual scores as premiums; disjoint *portfolio definitions* between discovery and validation (underlying constituent overlap remains possible). OOS R² from realized factor returns is **attribution, not predictability**. Fixed-weight long-short portfolios omit financing friction, short-borrow, transaction and market-impact costs. Vendor histories are not point-in-time vintages.
'''),
 nbformat.v4.new_code_cell(bootstrap),
 nbformat.v4.new_markdown_cell('''## 1. Predeclared design

Portfolio definitions: 25 Size–Value, 30 Industries and 25 Size–Momentum used **only for factor discovery**; 25 Size–Profitability and 25 Size–Investment used **only to validate**, although constituent stocks overlap economically. 2005-12-31 is the fixed final **formation month**; 2006–cutoff is a held-out evaluation period. Change settings before examining outcomes, never to optimize on validation results.

AQR original Excel downloads can be supplied via `AQR_PATHS` if the provider is blocked. No synthetic data substitute.
'''),
 nbformat.v4.new_code_cell('''from pathlib import Path
MODEL='FF6_BAB_QMJ'
AS_OF='2026-10-31'
TRAIN_END='2005-12-31'
N_PCS=5  # PREDECLARE before viewing the holdout results
DISCOVERY=('SIZE_BM_25','INDUSTRY_30','SIZE_MOM_25')
VALIDATION=('SIZE_OP_25','SIZE_INV_25')
USE_AQR=True
AQR_PATHS={}  # e.g. {'BAB':'/content/BAB.xlsx','QMJ':'/content/QMJ.xlsx'}
CACHE=Path('erp_cache')
OUT=Path('erp_equity_residual_v010_output')
OUT.mkdir(exist_ok=True,parents=True)
print('Design:',MODEL,'train_end',TRAIN_END,'as_of',AS_OF,'pcs',N_PCS)
'''),
 nbformat.v4.new_markdown_cell('''## 2. Load published factors and held-out test portfolios

This pulls the public Kenneth French monthly series and AQR original BAB/QMJ. No market data are fabricated. If public providers are blocked, use local original workbooks/cache.
'''),
 nbformat.v4.new_code_cell('''data=EquityData.from_public_sources(cache_dir=str(CACHE),aqr=USE_AQR,aqr_paths=AQR_PATHS)
print('Available universes:',[(key,len(df),df.shape[1]) for key,df in data.universes.items()])
print('Published AQR factors:',None if data.aqr is None else data.aqr.columns.tolist())
'''),
 nbformat.v4.new_markdown_cell('''## 3. Discovery → factor-neutral portfolios → OOS premia and pricing

- PCA on **OLS residual covariance in training** (the residual PC scores themselves have zero in-sample mean);
- self-financing, gross-1 mimicking portfolios projecting PCs onto the formation-period FF/AQR-beta-null subspace;
- future standalone mean / FF-spanning alpha tests with HAC and Holm corrections;
- incremental out-of-sample risk attribution, means and one-step identity-weighted **linear SDF/GMM** pricing comparison on separate test portfolio definitions.

These outputs distinguish *risk captured*, *additional premium found*, *pricing errors* and *forecasting*. No nonlinear pricing/Giglio–Xiu estimation is claimed.
'''),
 nbformat.v4.new_code_cell('''result=analyze_equity_residuals(
    data,model=MODEL,discovery_universes=DISCOVERY,
    validation_universes=VALIDATION,train_end=TRAIN_END,as_of=AS_OF,
    n_pcs=N_PCS,hac_lags=6,sdf_ridge=1e-4)
print('Train months:',result.metadata['formation_months'],'| OOS months:',result.metadata['oos_months'])
print('Residual covariance shares:');display(result.residual_eigenvalues.round(5))
print('Factor neutrality (formation period):');display(result.neutrality_tests.round(8))
print('Factor premia (strictly OOS):');display(result.fmp_premia_tests.round(5))
print('Held-out OOS attribution:');display(result.holdout_metrics.round(5))
print('One-step linear SDF/GMM pricing:');display(result.sdf_gmm_comparison.round(5))
'''),
 nbformat.v4.new_markdown_cell('''## 4. Optional: macroeconomic interpretation and lagged-state forecasts

**Separate objects:** economic *shocks* can explain contemporaneous FF/RPC returns, but their OLS coefficients are **not risk prices**. A separate **one-step linear macro SDF/GMM** using held-out portfolio training returns estimates tentative nontraded macro prices of risk and macro-implied FF/RPC premia; these are not causally identified or weak-factor robust. A separately supplied monthly **state** panel (e.g. publicly released lagged CPI inflation, credit spreads, IP growth) can inform expanding-window forecasts. If using revised FRED macro histories, results are **not point-in-time publication-vintage evidence**.

Set `ECONOMIC_SHOCKS_CSV` to a month-indexed CSV of genuine economic **innovations** for contemporaneous explanatory mapping (not raw levels). Set `STATE_LEVELS_CSV` for a month-indexed CSV of observable state variables. The state signal is automatically lagged by `STATE_LAG_MONTHS` monthly observations; release vintages/availability delays beyond this are the researcher's responsibility.

This cell is optional and will *not* fabricate macros or download substitutes if paths are unset.
'''),
 nbformat.v4.new_code_cell('''ECONOMIC_SHOCKS_CSV=None  # e.g. '/content/my_macro_innovations.csv' (DATE + numeric shock columns)
STATE_LEVELS_CSV=None     # e.g. '/content/my_lagged_state_variables.csv' (DATE + numeric columns)
STATE_LAG_MONTHS=2

def read_monthly_panel(filepath):
    if filepath is None:return None
    x=pd.read_csv(filepath,index_col=0,parse_dates=True)
    x.index=pd.DatetimeIndex(x.index)+pd.offsets.MonthEnd(0)
    if x.index.has_duplicates:raise ValueError('Duplicate monthly dates')
    return x.sort_index()

if ECONOMIC_SHOCKS_CSV or STATE_LEVELS_CSV:
    econ=read_monthly_panel(ECONOMIC_SHOCKS_CSV)
    states=read_monthly_panel(STATE_LEVELS_CSV)
    result=analyze_equity_residuals(
        data,model=MODEL,discovery_universes=DISCOVERY,validation_universes=VALIDATION,
        train_end=TRAIN_END,as_of=AS_OF,n_pcs=N_PCS,
        macro_shocks=econ,state_predictors=states,state_lag_months=STATE_LAG_MONTHS)
    if econ is not None:
        print('Macroeconomic exposures (descriptive only):'); display(result.macro_exposures.round(5))
        display(result.macro_diagnostics.round(5))
        print('Tentative nontraded macro SDF/GMM risk prices:');display(result.macro_sdf_coefficients.round(5))
        print('Macro-implied factor premia (one-step):');display(result.macro_sdf_pricing.round(5))
    if states is not None:
        print('Expanding OOS conditional forecasts (lagged macro state):');display(result.state_scores.round(5))
else:
    print('No macro data supplied; economic mapping and conditional state forecasting left unestimated.')
'''),
 nbformat.v4.new_markdown_cell('''## 5. Export all auditable tables

CSV outputs: residual eigenvectors/variance, neutral weights, realized factor portfolio returns, HAC alpha + Holm, holdout attribution/pricing and SDF/GMM, optional economic exposures, macro-SDF pricing and lagged-state forecasts. Manifest documents methods, sample dates and interpretation limits.
'''),
 nbformat.v4.new_code_cell('''folder=result.export(OUT/'discovery_results')
print('Exported',folder)
print('Files:',*[p.name for p in sorted(folder.iterdir())],sep='\\n  ')
'''),
 nbformat.v4.new_markdown_cell('''## Interpretation / next research gate

1. **Residual eigenvalue ≠ priced premium**. Null OOS mean and factor-spanning alpha are predeclared and Holm-corrected; genuine systematic pricing still requires economic cross-sectional robustness and independent assets.
2. **Partial risk neutrality is only guaranteed at formation**. Inspect OOS factor betas (`oos_empirical_factor_beta_max`), because a frozen factor-mimicking portfolio can reacquire standard exposures.
3. **Test portfolios overlap in constituents** and long-short positions in sorted portfolios may not be directly investible with the assumed costs. This is a *research diagnostic*, not a live trading result.
4. **SDF/GMM** here is a one-step identity-weighted linear SDF moment estimator, NOT efficient two-step GMM/HJ or Giglio–Xiu three-pass. Those remain planned.
5. **Economic factor mapping is not pricing identification** and macro vintage/lookahead remains an outstanding limitation.
''')]
npath=ROOT/'notebooks'/'Equity_Residual_Factor_Research_v0_10_PORTABLE.ipynb'
nbformat.write(n,npath)

# Preserve existing full Equity notebook behavior, but embed current code and rename release.
existing=nbformat.read(ROOT/'notebooks'/'Equity_Premia_v0_9_PORTABLE.ipynb',as_version=4)
existing.cells[1].source=bootstrap
existing.cells[0].source=existing.cells[0].source.replace('v0.9','v0.10').replace('unchanged research layer','existing factor-pricing benchmark')+'\n\nFor the independent residual factor research, use `Equity_Residual_Factor_Research_v0_10_PORTABLE.ipynb`.\n'
full_path=ROOT/'notebooks'/'Equity_Premia_v0_10_PORTABLE.ipynb'
nbformat.write(existing,full_path)
print(npath,full_path)
