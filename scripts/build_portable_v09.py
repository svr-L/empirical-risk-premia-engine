"""Regenerate BOTH portable notebooks with exact ERP v0.9 packaged source.

Neither notebook relies on the machine having the repo installed; all `.py`
files are embedded in a compressed encoded ZIP. The Equity notebook retains
its original analytic cells unchanged (except relabeling/paths).
"""
from pathlib import Path
import base64,io,zipfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
mem=io.BytesIO()
with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for f in sorted((ROOT/'src'/'erp').rglob('*.py')):
        z.write(f,f.relative_to(ROOT/'src'))
payload=base64.b64encode(mem.getvalue()).decode('ascii')

BOOTSTRAP=r'''# Portable ERP 0.9.0 — source modules embedded, requires no repo installation.
import io,zipfile,base64,sys,subprocess,importlib.util
from pathlib import Path
RUNTIME=Path.cwd()/'erp_v090_notebook_runtime'
(RUNTIME/'src').mkdir(parents=True,exist_ok=True)
PAYLOAD=__PAYLOAD__
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as zf:
    zf.extractall(RUNTIME/'src')
for name,pkg in {'numpy':'numpy','pandas':'pandas','scipy':'scipy','statsmodels':'statsmodels','requests':'requests','openpyxl':'openpyxl'}.items():
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
sys.path.insert(0,str(RUNTIME/'src'))
for key in list(sys.modules):
    if key=='erp' or key.startswith('erp.'):
        del sys.modules[key]
import numpy as np
import pandas as pd
from IPython.display import display
from erp import (__version__, RatesData,RatesPremiaEngine,RatesEmpiricalBondPremiaEngine,
                 load_macro_panel,load_macro_series_files,export_snapshot,
                 EquityData,EquityPremiaEngine,analyze_rates_v09)
print('ERP:',__version__,'— standalone source loaded')
'''.replace('__PAYLOAD__',repr(payload))

rates=nbformat.read(ROOT/'notebooks'/'Rates_Premia_v0_2_5_PORTABLE.ipynb',as_version=4)
rates.cells[0].source='''# Empirical Risk Premia Engine — Rates v0.9\n\n**Rates v0.8/v0.2.5 remains the immutable baseline.** This notebook executes its original funded PCA factor portfolios, carry/curve repricing, five original predictor families, and monthly strategy accounting. The new research layer compares *all* forecasts and strategies on common periods, tests overlapping-horizon loss differentials with HAC and moving-block bootstrap, and challenges carry+repricing with direct total-excess-return targets for FB and CP.\n\n**No claim that the three PCA factors are independently priced.** PCA is a risk representation; the source of premia remains an empirical question. Revised vendor historical macro observations are not PIT publication vintages.'''
rates.cells[1].source=BOOTSTRAP
rates.cells[3].source=rates.cells[3].source.replace("OUT_DIR=Path('erp_v025_output')","OUT_DIR=Path('erp_v09_output')")+'''\nV09_DIRECT_MODELS=('FB','CP')  # direct total-return target challengers; no post-hoc model tuning\nV09_BOOTSTRAP_REPS=399\nV09_BOOTSTRAP_SEED=202609\n'''
rates.cells[15].source=rates.cells[15].source.replace("root=OUT_DIR/'rates_v025'","root=OUT_DIR/'rates_v09'")
rates.cells[16].source='''## 7 — New v0.9 strictly matched forecast and portfolio tests\n\n**Within each horizon**, the common forecast-origin dates are the intersection of every original family, both existing benchmarks and the direct-target challengers, for **all three factor portfolios simultaneously**. The monthly strategy sample is intersected independently across every portfolio strategy. Do not compare these two sets of dates without respecting the H-month forecast vs next-month execution difference.\n\n- All predictive scores below use **total H-month funded excess returns**, including the known carry in decomposed forecast models.\n- DIRECT_FB and DIRECT_CP use the *same original learner* trained on total excess returns (not curve residuals) as a controlled target comparison.\n- HAC lags >= H-1 and paired block bootstrap account approximately for overlapping holding-period targets. Bootstrap confidence intervals are exploratory, not a solution to model-selection multiplicity.\n- The CP coefficient stability table uses annual checkpoints and standardized-feature ridge **only as a sensitivity diagnostic**, not as an optimized alternative trading strategy.\n- The forecast origins requiring realized H-month returns stop H months before the dataset ends: no operational as-of forecasts are claimed.'''
rates.cells.append(nbformat.v4.new_code_cell('''v09=analyze_rates_v09(\n    rates,research,macro_raw=macro_raw,direct_models=V09_DIRECT_MODELS,\n    min_train=MIN_TRAIN,ridge=RIDGE,cost_bps=COST_BPS,\n    bootstrap_reps=V09_BOOTSTRAP_REPS,seed=V09_BOOTSTRAP_SEED\n)\nprint('COMMON forecast and realized-trading samples:')\ndisplay(v09.common_samples)\nprint('OOS TOTAL EXCESS RETURN scores (all factors, common dates):')\ndisplay(v09.predictive.sort_values(['horizon','pooled_oos_r2_total_vs_uncond'],ascending=[True,False]).round(5))\nprint('By factor:')\ndisplay(v09.predictive_by_factor.sort_values(['horizon','factor','model']).round(5))\n'''))
rates.cells.append(nbformat.v4.new_markdown_cell('''## 8 — Is factor rotation adding value after costs?\n\nSame monthly realized return dates per horizon for all strategies, same cash benchmark and funded exposure constraints, with paired HAC/block confidence intervals for differences against both UNCONDITIONAL and CARRY_ONLY. **A positive OOS R² does not imply positive timing alpha.**'''))
rates.cells.append(nbformat.v4.new_code_cell('''print('STRICTLY MATCHED portfolio Sharpe / return:')\ndisplay(v09.rotation.sort_values(['horizon','model']).round(5))\nprint('Paired mean return spreads vs BOTH benchmarks, with uncertainty:')\ndisplay(v09.rotation_spreads.sort_values(['horizon','benchmark','model']).round(6))\nprint('Execution anatomy: overlay gross return, transaction-cost drag and gross/flat exposure:')\ndisplay(v09.signal_execution.round(6))\n'''))
rates.cells.append(nbformat.v4.new_markdown_cell('''## 9 — CP conditioning and safe exports\n\nForward X conditioning, standardized coefficient norms and fixed ridge sensitivity; no parameter is selected using held-out outcomes. The exported **v0.8** CSV and **v0.9** tables coexist under the same output directory and the factor snapshots have the same interface as Equity. Prices, macro and curves are not historical publication vintages.'''))
rates.cells.append(nbformat.v4.new_code_cell('''print('CP forward first-stage annual checkpoints (first rows):')\ndisplay(v09.cp_first_stage.head(15).round(6))\noutput=v09.export(root/'rates_v09_common_sample')\nprint('v0.9 common-sample report exported:',output)\nprint('files:',[p.name for p in sorted(output.iterdir())])\n'''))
nbformat.validate(rates)
nbformat.write(rates,ROOT/'notebooks'/'Rates_Premia_v0_9_PORTABLE.ipynb')

eq=nbformat.read(ROOT/'notebooks'/'Equity_Premia_v0_2_3_PORTABLE.ipynb',as_version=4)
eq.cells[0].source='''# Empirical Risk Premia Engine v0.9 — Equity module (unchanged research layer)\n\n**Equity has not been abandoned.** CAPM/FF3/Carhart4/FF5/FF6, BAB, QMJ, 130 test portfolios, Fama–MacBeth identification, HAC, spanning and OOS risk attribution remain available. This packaging update embeds the **same shared ERP 0.9 source** used by the new Rates research notebook.\n\nThe new Rates v0.9 econometric experiments have not been claimed to improve Equity premia estimation. Equity next: robust FMB/traded-factor reconciliation, identification per test universe, then challenger GMM/SDF and historical conditional premia forecasting.'''
eq.cells[1].source=BOOTSTRAP
eq.cells[3].source=eq.cells[3].source.replace('erp_v02_output','erp_v09_equity_output')
nbformat.validate(eq)
nbformat.write(eq,ROOT/'notebooks'/'Equity_Premia_v0_9_PORTABLE.ipynb')
print('Rates notebook',len(rates.cells),(ROOT/'notebooks'/'Rates_Premia_v0_9_PORTABLE.ipynb').stat().st_size)
print('Equity notebook',len(eq.cells),(ROOT/'notebooks'/'Equity_Premia_v0_9_PORTABLE.ipynb').stat().st_size)
