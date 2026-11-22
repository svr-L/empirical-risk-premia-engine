from pathlib import Path
import zipfile, io, base64, nbformat, shutil, json
ROOT=Path(__file__).resolve().parents[1]
# embed only user importable modules, no tests; notebooks remain runnable when uploaded alone.
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
 for f in (ROOT/'src'/'erp').rglob('*.py'):
  archive.write(f,f.relative_to(ROOT/'src'))
payload=base64.b64encode(buffer.getvalue()).decode()
startup=f'''# Offline-bundled erp v0.2.3 sources: this notebook also works when uploaded alone to Colab.
import io, zipfile, base64, sys, subprocess, importlib.util
from pathlib import Path
ROOT = Path.cwd() / "erp_v023_notebook_runtime"
(ROOT / "src").mkdir(parents=True, exist_ok=True)
PAYLOAD = "{payload}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as zf:
    zf.extractall(ROOT / "src")
for module, package in {{"numpy":"numpy","pandas":"pandas","scipy":"scipy","statsmodels":"statsmodels","requests":"requests","openpyxl":"openpyxl"}}.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",package])
sys.path.insert(0,str(ROOT / "src"))
# In a reused Colab runtime, purge previously imported older portable packages.
for _name in list(sys.modules):
    if _name == "erp" or _name.startswith("erp."):
        del sys.modules[_name]
import numpy as np
import pandas as pd
from IPython.display import display
from erp import __version__, export_snapshot, EquityData, EquityPremiaEngine, RatesData, RatesPremiaEngine
print("ERP version:",__version__, "| portable package embedded and loaded")
'''

m=lambda t: nbformat.v4.new_markdown_cell(t)
c=lambda t: nbformat.v4.new_code_cell(t)
equity=nbformat.v4.new_notebook()
equity['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
equity['cells']=[
 m('''# Empirical Risk Premia Engine v0.2.3 — Equity
### Public FF + AQR | Fama–MacBeth identification | Multi-universe | shared consumer exports

This notebook embeds the exact v0.2.3 modules; **no repo installation is required for Colab.** All source code is also included in the separate modular repository ZIP.

Research objective: measure asset risk exposures, factor pricing and identification, **not** generate trading recommendations. `as_of` truncates monthly observations but vendor files are revised datasets rather than true historical publication vintages.'''),
 c(startup),
 m('''## Configuration

Select a standard/augmented model and research universe. Set `USE_AQR = False` for a French-only offline-vendor compatibility run (`MODEL="FF6"`). If AQR disallows auto-download, upload the *original monthly* `.xlsx` files to your notebook directory and specify local paths. No long-only ETF substitution.'''),
 c('''MODEL = "FF6_BAB_QMJ"    # CAPM FF3 CARHART4 FF5 FF6 FF6_BAB FF6_QMJ FF6_BAB_QMJ FF3_BAB_QMJ
UNIVERSE = "POOLED_130"   # SIZE_BM_25 INDUSTRY_30 SIZE_OP_25 SIZE_INV_25 SIZE_MOM_25 POOLED_130
AS_OF = "2026-10-31"      # strict ISO date cutoff; change when new data become available
BETA_WINDOW = 60
USE_AQR = True
AQR_PATHS = {}             # e.g. {"BAB":"/content/BAB.xlsx", "QMJ":"/content/QMJ.xlsx"}
CACHE_DIR = Path("erp_data_cache")
OUT = Path("erp_v02_output")
print("Model:", MODEL, "| Universe:",UNIVERSE, "| as_of:",AS_OF)'''),
 m('''## 1 — Official public factors and test portfolios

Downloads 25 Size×B/M, 30 Industries, 25 Size×Profitability, 25 Size×Investment and 25 Size×Momentum, plus published BAB and QMJ **monthly long/short** factors. Uses full sample intersection and preserves FF3-family vs FF5-family SMB conventions.'''),
 c('''data = EquityData.from_public_sources(
    cache_dir=CACHE_DIR, aqr=USE_AQR, aqr_paths=AQR_PATHS,
)
print("Asset universes:",[(x, y.shape) for x,y in data.universes.items()])
print("AQR factors:", None if data.aqr is None else data.aqr.dropna().shape)
engine = EquityPremiaEngine(data)
f,rf = data.factors(MODEL)
display(f.loc[:AS_OF].tail())'''),
 m('''## 2 — Model horse race: identical dates

The horse race computes cross-model time-series alpha/errors on **exactly the same calendar months**. GRS and FMB diagnostics are reported separately; don't confuse lower in-sample alpha RMSE with stronger out-of-sample return forecasts.'''),
 c('''MODELS = ["CAPM", "FF3", "CARHART4", "FF5", "FF6"]
if USE_AQR:
    MODELS += ["FF6_BAB", "FF6_QMJ", "FF6_BAB_QMJ"]
common_models = engine.compare_models(MODELS, UNIVERSE, as_of=AS_OF, beta_window=BETA_WINDOW)
display(common_models.sort_values("alpha_rmse_ann"))'''),
 m('''## 3 — Fit selected factor specification, extract risks, errors and premia

Ex post factor attribution: `factor_component` is calculated from **realized current factor return**, using only beta estimated from prior months. It is not an ex-ante expected return forecast.'''),
 c('''snapshot = engine.fit(MODEL, UNIVERSE, as_of=AS_OF, beta_window=BETA_WINDOW, run_fmb=True)
print("Effective data through", snapshot.as_of, "| months",snapshot.metadata['n_months'])
print("GRS (standard, if identifiable):",snapshot.metadata['grs'])
display(snapshot.fit_diagnostics.sort_values("alpha_ann").head(12))
display(snapshot.betas.head())
display(snapshot.factor_premia.to_frame("mean_excess_monthly"))
print("Static cross-sectional prices of risk (monthly) and pricing errors:")
display(snapshot.static_cs_premia.to_frame("static_CS_lambda_monthly"))
display(snapshot.static_pricing_errors.to_frame("static_CS_error_monthly").head(8))
print("Static CS diagnostics:", snapshot.metadata["static_cross_sectional_pricing"])'''),
 m('''## 4 — Fama–MacBeth: prices of risk + identification quality

Lagged betas use `t-60,...,t-1`. At each month t, contemporaneous cross-sectional returns determine a realized λ_t. The table juxtaposes λ estimates (and HAC t) with cross-sectional beta dispersion, partial dispersion, auxiliary R² and scaled condition number. This is **not** Shanken-adjusted inference.'''),
 c('''display(snapshot.fmb_diagnostics.round(5))
print("Historical traded-factor means vs FMB lambda estimates (monthly decimals):")
a = snapshot.factor_premia.rename("traded_factor_mean")
b = snapshot.fmb_diagnostics["lambda_monthly"].rename("FMB_lambda")
display(pd.concat([a,b],axis=1).round(5))
print("Last 5 monthly lambda observations:")
display(snapshot.fmb_lambdas.tail().round(4))
if USE_AQR:
    print("Official-factor spanning vs FF6 (not a time-series trading signal):")
    display(pd.DataFrame([engine.spanning_test(target="BAB", against_model="FF6", as_of=AS_OF),
                          engine.spanning_test(target="QMJ", against_model="FF6", as_of=AS_OF)]))'''),
 m('''## 5 — Risk exposure / alpha / residual decomposition

`alpha` is the in-sample regression intercept. `residual_returns` are idiosyncratic regression residuals. They do **not** establish a monetizable future alpha.'''),
 c('''print("Alpha (annualized, first 10 assets)")
display(snapshot.alpha.head(10).mul(12).to_frame("fitted_alpha_ann"))
print("Realized factor-attribution sample (beta known strictly before each observation)")
display(snapshot.oos_attribution.tail(12))
print("Fitted residual covariance (first 5 assets)")
display(snapshot.residuals.iloc[:,-5:].cov().round(5))'''),
 m('''## 6 — Export to common file schema

Exports CSV tables plus `manifest.json` and explicit meanings/limitations, consumable by Slicer/Referee without modifying their own statistical logic.'''),
 c('''destination = export_snapshot(snapshot, OUT / (UNIVERSE + "__" + MODEL))
print("Exported:",destination)
print("Files:",sorted(x.name for x in destination.iterdir()))
from erp.integrations import to_alpha_referee_factor_library
print("Alpha Referee factor library:", list(to_alpha_referee_factor_library(snapshot)["factors"].columns))
# Alpha Slicer v0.5.1 integration, only if you have actual strategy & benchmark time series:
# from erp.integrations import to_alpha_slicer_inputs
# slicer_kwargs = to_alpha_slicer_inputs(snapshot, strategy_net=real_net,
#                                       strategy_gross=real_gross, benchmark=real_benchmark)
# slicer_result = your_existing_slicer.fit(**slicer_kwargs)'''),
 m('''## 7 — Research interpretation

Interpret each result separately: **factor fit**, **cross-sectional price of risk**, **identification strength**, **spanning**, **OOS attribution** and **predictive usefulness** are different questions. Re-run the same dates for comparisons with published QMJ/BAB; do not infer a universal best model. The historical v0.1 notebook remains unchanged and is included in the source ZIP as provenance.'''),
]
rates=nbformat.v4.new_notebook()
rates['metadata']=equity['metadata'].copy()
rates['cells']=[
 m('''# Empirical Risk Premia Engine v0.2.3 — Funded Rates adapter

Portable notebook with embedded v0.2 engine. Reuses the validated v0.8 **Level/Slope/Curvature funded factor** construction, emits common outputs, but deliberately does **not** reimplement the v0.8 Fama–Bliss/CP conditional trading grid. Source: original Federal Reserve Board Kim-Wright bulk CSV (10 zero yields, one request), with exact monthly FEDFUNDS funding. No Treasury par-yield substitute.'''),
 c(startup),
 m('''### Reliable source configuration\n\n`CURVE_SOURCE="board"` uses one official Federal Reserve CSV rather than 10 slow FRED requests. If automated access fails, open `https://www.federalreserve.gov/data/yield-curve-tables/feds200533.csv` in your browser, upload the file to Colab and assign `BOARD_CSV_PATH`. Funding uses the **exact monthly FEDFUNDS** series. The funding provider is a separate publicly available CSV mirror of the original FRED/FEDFUNDS monthly series (not DBnomics), with optional recent-month Board H.15 update. The notebook will **warn and truncate its sample** if funding values do not extend to the selected `AS_OF`, rather than carry a stale rate forward. For a fully network-independent funding load, download either `https://eco3min.fr/dataset/federal-funds-rate.csv` or `https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS` in the browser and upload the CSV; assign `FEDFUNDS_CSV_PATH`. Neither funding nor curve is silently substituted.'''),
 c('''AS_OF = "2026-10-31"
OUT = Path('erp_v02_output')
CACHE_DIR = Path('erp_data_cache')
CURVE_SOURCE = 'board'   # single official Fed Board CSV
BOARD_CSV_PATH = None    # e.g. '/content/feds200533.csv' (upload original Board CSV)
FUNDING_SOURCE = 'mirror' # 'mirror' (default), 'fred', 'auto', 'local'
FEDFUNDS_CSV_PATH = None # e.g. '/content/FEDFUNDS.csv' (local source takes precedence)
'''),
 m('''## 1 — Zero curve, funding proxy and standardized PCA

Ex-post factor extraction: lagged EWMA risk and static PCA calibration ending 2002-12-31; positive Level means upward yields. No ex-ante forecast is claimed.'''),
 c('''rates_data = RatesData.from_fred(
    cache_dir=CACHE_DIR, curve_source=CURVE_SOURCE,
    local_curve_path=BOARD_CSV_PATH,
    local_fedfunds_path=FEDFUNDS_CSV_PATH, funding_source=FUNDING_SOURCE,
)
rates = RatesPremiaEngine(rates_data)
snapshot = rates.fit(as_of=AS_OF)
print(snapshot.metadata)
display(snapshot.factor_premia.to_frame('mean_monthly'))
display(rates.pca.loadings)'''),
 m('''## 2 — Current funded mimicking portfolios and risk decomposition'''),
 c('''print('Current risky weights and cash funding')
display(rates.latest_weights.round(4))
display(rates.latest_mimic_diagnostics.round(8))
print('Current one-sigma yield-shock ZC exposures')
display(rates.latest_analytic_exposures.round(5))
print('Carry and curve decomposition')
display(pd.concat({'FundedCarry':rates.factor_carry.mean(), 'RealizedCurve':rates.factor_curve_component.mean()},axis=1))'''),
 m('''## 3 — Export standard core schema + rates-specific analytic modules'''),
 c('''destination=export_snapshot(snapshot,OUT/'rates_PCA3_FUNDED')
rates.export_auxiliary(destination)
print('Exported:', destination)
print('Files:',sorted(x.name for x in destination.iterdir()))'''),
]
for n,path in [(equity,'Equity_Premia_v0_2_3_PORTABLE.ipynb'),(rates,'Rates_Premia_v0_2_3_PORTABLE.ipynb')]:
 nbformat.validate(n)
 nbformat.write(n,ROOT/'notebooks'/path)
 print(path,'bytes',(ROOT/'notebooks'/path).stat().st_size)
