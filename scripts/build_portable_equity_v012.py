"""Build v0.12 Colab notebook retaining v0.11 baselines + new experiments."""
from pathlib import Path
import nbformat,base64,gzip,io,tarfile,re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks'/'Equity_Factor_Discovery_and_Macro_v0_11_AUTO_FRED.ipynb'
if not BASE.exists():
    BASE=Path('/mnt/data/Equity_Factor_Discovery_and_Macro_v0_11_AUTO_FRED.ipynb')
nb=nbformat.read(BASE,as_version=4)
source=ROOT/'src'/'erp'
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for file in sorted(source.glob('*.py')):
        tar.add(file,arcname='erp/'+file.name)
blob=base64.b64encode(gzip.compress(buf.getvalue(),compresslevel=9)).decode('ascii')
original=nb.cells[1].source
original=re.sub(r'EMBEDDED="[^"]+"',f'EMBEDDED="{blob}"',original,count=1)
original=original.replace('erp_v011_notebook_runtime','erp_v012_notebook_runtime')
original=original.replace('from erp import (__version__,EquityData,analyze_equity_residuals,\n    analyze_equity_factor_extensions,load_equity_fred_macro)',
'''from erp import (__version__,EquityData,analyze_equity_residuals,
    analyze_equity_factor_extensions,load_equity_fred_macro,
    analyze_equity_v012,load_jk_monetary_shocks,
    load_inflation_release_surprises,analyze_equity_shock_mapping,
    estimate_jk_shock_prices)''')
nb.cells[1].source=original
nb.metadata['erp_version']='0.12.0'
nb.metadata['title']='ERP Equity: rolling RP, comparable pricing, announcement shocks v0.12'
nb.cells[0].source='''# Equity Risk Premia — v0.12: Stability, priced factors and identified shocks

This standalone notebook **preserves the v0.11 published FF6/BAB/QMJ & residual-factor experiments** and adds three strictly distinct research blocks:

1. **Rolling residual & pricing-weighted PCs**, with each month traded using only previously observable portfolio returns.
2. **Common-sample Fama–MacBeth and two-step shrinkage GMM/SDF**, comparing holdout pricing using fixed formation-only portfolios, HAC and paired time-block bootstrap.
3. **Public high-frequency identified policy surprises:** Jarociński–Karadi monthly MP & CBI shock decomposition. Separate option to load genuine CPI-release surprises constructed from actual minus pre-announcement consensus.

**Important:** The RP method is a *residual/FF-conditional augmented-mean adaptation* of RP-PCA, **not** the complete Lettau–Pelger estimators and inference. FMB does not include Shanken corrections and GMM with shrinkage does not support standard J-test inference. Characteristic-sorted holdout portfolios can share constituent stocks. The high-frequency Fed shock identification relies on event-window/sign restrictions; no series of CPI surprises is inferred from FRED-level CPI. All results are research diagnostics, not deployed trading premia.'''
nb.cells[3].source=nb.cells[3].source.replace("OUT=Path('erp_equity_v011_results')","OUT=Path('erp_equity_v012_results')")
nb.cells[14].source=nb.cells[14].source.replace('## 7. Audit exports and interpretation','## 7. v0.11 baseline exports and interpretation')
md=nbformat.v4.new_markdown_cell
code=nbformat.v4.new_code_cell
nb.cells.extend([
md('''## 8. Rolling vs static: are the residual factors stable, and is pricing orientation useful?

**Freeze only at each formation date:** rolling windows of 120 months, rebalance annually. Each FMP is dollar-neutral and neutral to formation FF/AQR betas, using actual discovery-portfolio excess returns. Signs are aligned to the *previously known* factor direction. Annualized OOS premia are **not** automatically priced systematic risks. Prespecified gamma=0 vs gamma=12, first 3 PCs; no tuning on OOS results.'''),
code('''ROLLING_WINDOW_MONTHS=120
ROLLING_REBALANCE_MONTHS=12
V012_N_PCS=3
V012_GAMMAS=(0.,12.)
pricing=analyze_equity_v012(data,base,formation_window=ROLLING_WINDOW_MONTHS,
    rebalance_months=ROLLING_REBALANCE_MONTHS,n_pcs=V012_N_PCS,
    gammas=V012_GAMMAS,bootstrap_draws=399)
print('Rolling PC premia:');display(pricing.rolling_premia.round(5))
print('First/last formation, turnover and FF neutrality:')
display(pricing.rolling_diagnostics.head(6).round(5))
display(pricing.rolling_diagnostics.tail(6).round(5))
pricing.export(OUT/'equity_v012_rolling_and_pricing')
'''),
md('''## 9. FMB vs two-step GMM on precisely the same holdout assets and dates

Compare FF6/BAB/QMJ alone, FF+3 *plain residual PCs* (gamma=0), and FF+3 *augmented-mean PCs* (gamma=12). FMB formation betas and mean lambda estimates are fixed before the OOS period; two-step GMM uses shrinkage HAC weighting and the SDF pricing moments. Rolling FMP returns are **not** retrospectively inserted into the fixed-formation horse race. The bootstrap resamples **monthly loss differences** in common 12-month blocks; do not select gamma based on OOS results.'''),
code('''print('Common-sample pricing: annualized mean RMSE, FMB vs two-step GMM')
display(pricing.pricing_comparison.round(6))
print('FMB training lambda estimates and HAC t-stats:')
display(pricing.fmb_lambdas.round(5))
print('Paired block-bootstrap differences vs original FF model:')
display(pricing.bootstrap_comparison.round(7))
'''),
md('''## 10. Actual FOMC policy-information shocks — Jarociński–Karadi

The updated [author dataset](https://github.com/marekjarocinski/jkshocks_update_fed) contains monthly *MP_median* and *CBI_median* high-frequency announcement shocks. The two add to the announcement policy surprise. The panel is monthly since 1990, **not the same as changes in the level of the policy rate**; source currently ends March 2026, so months after that are missing, not zero. First attempt public GitHub, then cached copy, then a local original CSV (`shocks_fed_jk_m.csv`). This is independent of FRED macro download.

**Same-month fit** measures shock-factor comovement and is not a tradeable month-ahead predictor. The lag-1 experiment tests whether shocks observed during month t predict factor returns in month t+1. These are observational factor-response diagnostics, not yet an identified macro risk-premium estimator.'''),
code('''from pathlib import Path
# Optional, if GitHub unavailable: upload author CSV and use its actual path here.
JK_LOCAL_PATH=None
if JK_LOCAL_PATH is None:
    candidates=sorted(Path('/content').glob('shocks_fed_jk_m*.csv')) if Path('/content').exists() else []
    if candidates:
        JK_LOCAL_PATH=str(candidates[-1]);print('Found uploaded JK file:',JK_LOCAL_PATH)
jk=None;shock_mapping=None
try:
    jk=load_jk_monetary_shocks(cache_dir=CACHE/'shock_data',path=JK_LOCAL_PATH,as_of=AS_OF)
    print('JK source:',jk.provenance['source'],'ends',jk.shocks.index.max())
    f,_=data.factors(MODEL)
    shock_mapping=analyze_equity_shock_mapping(f,pricing.rolling_returns,jk,
        train_end=TRAIN_END)
    if shock_mapping.empty:
        print('Insufficient common train/OOS months for JK analysis')
    else:
        display(shock_mapping.round(5))
    jk.export(OUT/'jk_announcement_shocks')
    shock_mapping.to_csv(OUT/'jk_ff_rpc_exposure_and_predictability.csv',index=False)
    # Pricing: nontraded-shock beta prices inferred cross-sectionally, FF controls.
    # This is a point-estimate diagnostic under high-frequency identifying assumptions.
    _,rf=data.factors(MODEL)
    holdout=pd.concat([data.excess_returns(k,rf).add_prefix(k+'__')
         for k in VALIDATION],axis=1)
    shock_price_summary,shock_price_lambda=estimate_jk_shock_prices(
         holdout,f,jk,train_end=TRAIN_END)
    print('JK nontraded-shock prices, heldout common sample (exploratory):')
    display(shock_price_summary.round(6))
    display(shock_price_lambda.loc[shock_price_lambda.factor.isin(
         ['MP_median','CBI_median'])].round(6))
    shock_price_summary.to_csv(OUT/'jk_common_sample_mean_pricing.csv',index=False)
    shock_price_lambda.to_csv(OUT/'jk_shock_fmb_lambda_hac.csv',index=False)
except (ConnectionError,ValueError,FileNotFoundError) as e:
    print('JK shocks unavailable: ',e)
    print('Download https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/main/shocks_fed_jk_m.csv')
    print('Upload the original CSV and re-run this cell; FRED levels are not used as a substitute.')
'''),
md('''## 11. Inflation *release* surprises — optional separate input

There is no reliable automatic public consensus archive in this release. **Do not substitute** FRED CPI changes or Cleveland Fed expectations for verified announcement surprises. If you possess release-level data with `release_date,actual,consensus,consensus_source` (same rate units, pre-release consensus), load the CSV/ZIP. Results are separate from monetary-policy shocks. The parser cannot independently verify when a provider recorded its consensus.'''),
code('''CPI_RELEASE_CSV_PATH=None
inflation_shocks=None
if CPI_RELEASE_CSV_PATH is not None:
    inflation_shocks=load_inflation_release_surprises(CPI_RELEASE_CSV_PATH,as_of=AS_OF)
    ff,_=data.factors(MODEL)
    cpi_exposures=analyze_equity_shock_mapping(ff,pricing.rolling_returns,
        inflation_shocks,train_end=TRAIN_END,min_oos=24)
    display(cpi_exposures.round(5))
    inflation_shocks.export(OUT/'inflation_release_surprises')
else:
    print('CPI announcement surprises NOT estimated: no verified actual/consensus release file')
'''),
md('''## 12. Audit / next research decision

**Decision gates:** common residual risk, incremental OOS pricing after multiple-testing adjustment, pricing stability under rolling retraining, and pre-specified economic shock exposures/predictability must each be tested separately. Report uncertainty and any model selection as exploratory. This notebook does **not** claim published weak-factor robustness, identified inflation news when consensus is absent, or improved investable TAA value.'''),
code('''print('ERP',__version__)
print('Core output:',OUT.resolve())
print('Rolling premia factors:',len(pricing.rolling_premia))
print('Pricing candidates x methods:',len(pricing.pricing_comparison))
print('FMB lambda rows:',len(pricing.fmb_lambdas))
print('GMM shrinkage status: diagnostics only; no standard J-test')
print('Policy shock mapping:', 'available' if shock_mapping is not None else 'not estimated')
print('CPI release shocks:', 'available' if inflation_shocks is not None else 'not estimated')
''')
])
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.outputs=[];cell.execution_count=None
nbformat.validate(nb)
out=ROOT/'notebooks'/'Equity_Factor_Discovery_and_Pricing_v0_12_PORTABLE.ipynb'
nbformat.write(nb,out)
print('Notebook',out,len(nb.cells),'cells',round(out.stat().st_size/1024),'KB')
