from pathlib import Path
import nbformat,base64,gzip,io,tarfile,re
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'notebooks'/'Equity_Factor_Discovery_and_Pricing_v0_12_1_JK_FIX.ipynb'
nb=nbformat.read(base,as_version=4)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for p in sorted((ROOT/'src'/'erp').glob('*.py')):tar.add(p,arcname='erp/'+p.name)
blob=base64.b64encode(gzip.compress(buf.getvalue(),compresslevel=9)).decode()
s=nb.cells[1].source
s=re.sub(r'EMBEDDED="[^"]+"',f'EMBEDDED="{blob}"',s,count=1)
s=s.replace('erp_v012_notebook_runtime','erp_v0122_notebook_runtime')
s=s.replace("import numpy as np,pandas as pd", "# Avoid reusing a previously imported ERP in the same Colab runtime.\nfor key in list(sys.modules):\n    if key=='erp' or key.startswith('erp.'): del sys.modules[key]\nimport numpy as np,pandas as pd")
s+='\nfrom erp import analyze_rolling_shocks\nfrom erp.equity_macro import find_complete_equity_fred_file\nassert __version__ == "0.12.2"\n'
nb.cells[1].source=s
nb.cells[0].source='''# Equity Risk Premia — v0.12.2: rolling RP economic mapping

Preserves the original v0.12.1 research and adds a dedicated 2006–2011 formation / 2012 onward validation for **all six rolling factors** and FF/AQR. JK contemporaneous, lagged and pricing tests share an identical complete-case calendar. FRED files are validated for all six series before use. FRED changes remain proxies, not identified surprises. The new split is retrospective, not independent confirmation of RP12_PC2.

'''+nb.cells[0].source
nb.cells[3].source=nb.cells[3].source.replace('erp_equity_v012_results','erp_equity_v0122_results')+'\nSHOCK_TRAIN_END="2011-12-31"\nPROMPT_FRED_UPLOAD=True # set False for noninteractive runs\n'
nb.cells[11].source='''# Select a complete Equity file, never accidentally a partial Rates download.
from pathlib import Path
macro=None
try:
    fred_path,rejected=find_complete_equity_fred_file([Path('/content'),Path.cwd(),CACHE/'macro'],explicit=FRED_LOCAL_PATH)
    print('Rejected local candidates:',rejected)
    if fred_path is not None: print('Using local FRED:',fred_path)
    try:
        macro=load_equity_fred_macro(cache_dir=CACHE/'macro',path=fred_path,as_of=AS_OF,timeout=8)
    except (ConnectionError,ValueError,FileNotFoundError) as exc:
        if not PROMPT_FRED_UPLOAD: raise
        try: from google.colab import files
        except ImportError: raise exc
        print('Upload the original combined six-series FRED CSV/ZIP (e.g. fredgraph (5).zip). Cancel to continue without FRED.')
        print('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL,INDPRO,DGS10,DGS2,BAA10Y,VIXCLS')
        uploaded=files.upload()
        for filename in uploaded:
            try:
                path,_=find_complete_equity_fred_file([],explicit=filename)
                macro=load_equity_fred_macro(cache_dir=CACHE/'macro',path=path,as_of=AS_OF)
                break
            except (ValueError,OSError) as e: print('Rejected',filename,e)
    if macro is not None:
        print('FRED source:',macro.provenance['source']); display(macro.states.tail(6));display(macro.changes.tail(6))
    else: print('FRED not estimated; no data substituted.')
except (ConnectionError,FileNotFoundError,ValueError) as exc:
    print('FRED not estimated:',exc)
'''
# Preserve historical shock pricing, only FF mapping in legacy section, explicit separation.
nb.cells[21].source=nb.cells[21].source.replace('analyze_equity_shock_mapping(f,pricing.rolling_returns,jk,','analyze_equity_shock_mapping(f,pd.DataFrame(index=f.index),jk,').replace('jk_ff_rpc_exposure_and_predictability.csv','jk_legacy_ff_exposure_and_predictability.csv')
# append new cells before final audit
idx=24
md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
nb.cells.insert(idx,md('''## 12. Dedicated rolling-factor economic test — new in v0.12.2

All six rolling factors are now genuinely observed during the dedicated 2006–2011 mapping training period. Use a common complete-case calendar for FF, RP, holdout assets and current/lagged shocks. Pricing compares FF, FF+JK, FF+each RP family and FF+each RP family+JK under FMB and GMM. The shock-only SDF contribution is a descriptive covariance-pricing projection calibrated on the holdout portfolios, not structural causal decomposition. Historical OOS dates are reused from earlier research. HAC uses consecutive complete-case observations; weak-identification inference is not validated.'''))
nb.cells.insert(idx+1,code('''ff,rf=data.factors(MODEL)
holdout=pd.concat([data.excess_returns(k,rf).add_prefix(k+'__') for k in VALIDATION],axis=1)
rolling_economic={}
economic_inputs={}
if jk is not None: economic_inputs['JK']=jk.shocks[['MP_median','CBI_median']]
if macro is not None: economic_inputs['FRED_PROXY']=macro.changes
for label,shock_panel in economic_inputs.items():
    try:
        result=analyze_rolling_shocks(ff,pricing.rolling_returns,holdout,shock_panel,
            train_end=SHOCK_TRAIN_END,as_of=AS_OF,group_label=label)
        rolling_economic[label]=result
        result.export(OUT/('rolling_economic_'+label))
        print(label,'common sample:',result.metadata)
        print('Factor mapping (all FF and RP, including FF-controlled RP):');display(result.mapping.round(5))
        print('Heldout pricing on identical dates:');display(result.pricing.round(6))
        print('FMB risk prices (exploratory):');display(result.lambdas[result.lambdas.factor.isin(shock_panel.columns)].round(6))
        print('Shock-only SDF implied premia, not causal attribution:');display(result.premium_mapping.round(5))
    except ValueError as e: print(label,'dedicated RP economic study not estimated:',e)
if not rolling_economic: print('No dedicated economic test completed; inspect input/coverage messages.')
'''))
nb.cells[-2].source=nb.cells[-2].source.replace('## 12.','## 13.')
nb.cells[-1].source+='\nprint("Dedicated rolling economic studies completed:",list(rolling_economic))\nprint("FRED proxies available:",macro is not None)\n'
nb.metadata['erp_version']='0.12.2'
for c in nb.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
out=ROOT/'notebooks'/'Equity_Factor_Discovery_and_Pricing_v0_12_2_PORTABLE.ipynb'
nbformat.write(nb,out); print(out)
