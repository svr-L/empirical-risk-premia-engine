"""Recreate published summary CSVs and figures from stored notebook outputs.
No network, estimation or external data. Values retain displayed precision.
"""
from pathlib import Path
import io,json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'results/evidence'; T=ROOT/'results/tables'; F=ROOT/'results/figures'
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)

def table(nb,cell,output):
    n=json.loads((E/nb).read_text()); html=''.join(n['cells'][cell]['outputs'][output]['data']['text/html'])
    d=pd.read_html(io.StringIO(html))[0]
    if isinstance(d.columns,pd.MultiIndex):
        d.columns=[b if str(a).startswith('Unnamed') else a for a,b in d.columns]
    return d.drop(columns=[c for c in d if str(c).startswith('Unnamed')],errors='ignore')

EQ='equity_v013_executed.ipynb'; RA='rates_v025_executed.ipynb'; STD='equity_standard_executed.ipynb'
provenance=[]
def save(name,d,nb,cell,output):
    d.to_csv(T/(name+'.csv'),index=False)
    provenance.append(dict(table=name+'.csv',notebook=nb,cell_index=cell,output_index=output,units='As displayed in source; returns and RMSE generally decimals unless source states otherwise.'))
    return d
panels={'EQUITY':1,'JK':3,'FRED_PROXY':5,'MANAGED':7,'FUNDED_RATES':9,'ETF_PROXY':12,'JOINT_XA':15}
scores=[]
for p,j in panels.items():
    x=table(EQ,33,j).assign(panel=p); scores.append(x); save('equity_dynamic_'+p.lower(),x,EQ,33,j)
a=pd.concat(scores,ignore_index=True); a.to_csv(T/'equity_dynamic_all.csv',index=False)
for name,nb,cell,output in [('equity_static_rpc',EQ,5,1),('equity_rolling_rp',EQ,17,1),('equity_constrained_rp',EQ,27,0),('equity_joint_cross_asset_decomposition',EQ,33,16),('equity_experiment_status',EQ,33,18),('equity_standard_benchmarks',STD,7,0),('equity_standard_spanning',STD,11,6),('rates_pca_variance',RA,5,3),('rates_funded_mean',RA,5,1),('rates_fama_bliss_classic',RA,9,1),('rates_forecast_scores',RA,9,3),('rates_rotation_scores',RA,11,1),('rates_rotation_spreads',RA,11,3),('rates_cost_sensitivity',RA,11,5)]:
    x=table(nb,cell,output)
    if name in ['rates_pca_variance','rates_funded_mean','rates_fama_bliss_classic']:
        # Preserve the source's unnamed index (the portfolio name).
        n=json.loads((E/nb).read_text()); x=pd.read_html(io.StringIO(''.join(n['cells'][cell]['outputs'][output]['data']['text/html'])))[0].rename(columns={'Unnamed: 0':'factor' if 'classic' not in name else 'asset'})
    if name=='equity_joint_cross_asset_decomposition':
        x=x.rename(columns={c:c.replace('_monthly','_ann') for c in x.columns if c.endswith('_monthly')})
    if name=='rates_cost_sensitivity':
        x=x.rename(columns={'cost_bps':'model','1.0':'sharpe_1bp','5.0':'sharpe_5bp','10.0':'sharpe_10bp'})
    save(name,x,nb,cell,output)
pd.DataFrame(provenance).to_csv(T/'provenance.csv',index=False)

# Source numbers behind the CV / README. Assert exact displayed values before plotting.
def get(p,m,e,s='rolling'):
    r=a.query('panel==@p and model==@m and estimator==@e and scheme==@s')
    assert len(r)==1,(p,m,e,s)
    return float(r.mean_pricing_rmse_ann.iloc[0])
base=get('EQUITY','FF','FMB'); shrink=get('EQUITY','FF','FMB_SHRINK'); rates=get('FUNDED_RATES','FF_RP12_EXTRA','FMB_SHRINK'); ref=get('FUNDED_RATES','FF','FMB_SHRINK')
assert np.allclose([base,shrink,ref,rates],[.02069,.01648,.01648,.01521])
r=pd.read_csv(T/'rates_forecast_scores.csv'); fb=r.query('model=="FB"').sort_values('horizon'); assert np.allclose(fb.OOS_R2_vs_uncond,[.0766,.1138,.1348])
metrics=pd.DataFrame([
    ['Equity: shrinkage vs plain rolling FMB',base,shrink,100*(1-shrink/base),'annualized cross-sectional RMSE of mean OOS pricing errors',176,'EQUITY'],
    ['Equity: joint residual + funded Rates addition',ref,rates,100*(1-rates/ref),'annualized cross-sectional RMSE of mean OOS pricing errors',176,'FUNDED_RATES'],
],columns=['comparison','benchmark_rmse_decimal','challenger_rmse_decimal','relative_reduction_pct','metric','n_months','panel']);metrics.to_csv(T/'headline_equity_metrics.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#334155','xtick.color':'#334155','ytick.color':'#334155','text.color':'#172b42','axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
navy='#244963'; teal='#167d82'; gray='#b0b9c2'; amber='#b26b33'
def finish(fig,filename,note):
    fig.text(.08,.025,note,fontsize=9,color='#526375',ha='left'); fig.subplots_adjust(left=.12,right=.96,bottom=.20,top=.79);fig.savefig(F/filename,dpi=180,bbox_inches='tight',pad_inches=.22);plt.close(fig)
def setup(title,subtitle):
    fig,ax=plt.subplots(figsize=(9,4.6));fig.text(.08,.94,title,size=17,weight='bold');fig.text(.08,.875,subtitle,size=10,color='#526375');ax.grid(axis='y',color='#dfe5ea',linewidth=.7);ax.set_axisbelow(True);return fig,ax
fig,ax=setup('Equity pricing: dynamic estimation & shrinkage','50 held-out portfolios  |  Jan 2012 – Aug 2026  |  176 monthly evaluation dates')
schemes=['fixed','expanding','rolling']; xx=np.arange(3); width=.23
for i,(e,label,color) in enumerate([('FMB','Fama–MacBeth',navy),('FMB_SHRINK','FMB + Gaussian shrinkage',teal),('GMM','Two-step GMM',gray)]):
    vals=[100*get('EQUITY','FF',e,s) for s in schemes];bars=ax.bar(xx+(i-1)*width,vals,width,label=label,color=color);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
ax.set_xticks(xx,['Fixed','Expanding','Rolling 72M']);ax.set_ylabel('Annualized mean pricing RMSE (%)');ax.set_ylim(0,11.7);ax.legend(frameon=False,loc='upper right',fontsize=9);finish(fig,'equity_dynamic_pricing.png','Source: executed ERP v0.13; lower is better. Mean pricing error, not monthly forecast RMSE.')
fig,ax=setup('Incremental cross-asset information','Matched 176-month FUNDED_RATES panel  |  Rolling 72-month FMB + shrinkage')
labels=['FF/AQR','FF/AQR + RP12','FF/AQR + funded Rates','FF/AQR + RP12 + Rates']; models=['FF','FF_RP12','FF_EXTRA','FF_RP12_EXTRA'];vals=[100*get('FUNDED_RATES',m,'FMB_SHRINK') for m in models]
bars=ax.bar(range(4),vals,color=[navy,gray,gray,teal],width=.60);ax.bar_label(bars,fmt='%.3f',padding=5);ax.set_xticks(range(4),['FF/AQR','+ RP12','+ Rates','+ RP12\n+ Rates']);ax.set_ylabel('Annualized mean pricing RMSE (%)');ax.set_ylim(0,2.12);finish(fig,'equity_cross_asset_pricing.png','Source: executed ERP v0.13. Joint addition: −7.7% relative RMSE; statistical significance not established.')
fig,ax=setup('Treasury premia: forecast skill by horizon','Funded Level / Slope / Curvature portfolios  |  Mean factor-level out-of-sample R²')
xx=np.arange(3); w=.23
for i,(m,label,color) in enumerate([('FB','Fama–Bliss',teal),('CP','CP-style',navy),('BASE_STATE','Macro state',gray)]):
    vals=r.query('model==@m').sort_values('horizon').OOS_R2_vs_uncond.to_numpy()*100;bars=ax.bar(xx+(i-1)*w,vals,w,color=color,label=label);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
ax.axhline(0,color='#526375',lw=.8);ax.set_xticks(xx,['3M','6M','12M']);ax.set_ylim(-46,20);ax.set_ylabel('OOS R² vs unconditional (%)');ax.legend(frameon=False,loc='lower left',fontsize=9);finish(fig,'rates_forecasting.png','Source: executed Rates v0.2.5. Repricing targets; model-specific samples, not v0.9 common-sample scores.')
fig,ax=setup('Treasury premia: forecast skill is not timing alpha','Funded monthly rotation  |  5 bp cost assumption  |  Horizon-specific strategy comparisons')
q=pd.read_csv(T/'rates_rotation_scores.csv');xx=np.arange(3)
for i,(m,label,color) in enumerate([('FB','Fama–Bliss',teal),('UNCONDITIONAL','Unconditional',navy),('CARRY_ONLY','Carry only',gray)]):
    vals=q.query('model==@m').sort_values('horizon').Sharpe.to_numpy();bars=ax.bar(xx+(i-1)*w,vals,w,color=color,label=label);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
ax.set_xticks(xx,['3M','6M','12M']);ax.set_ylabel('Reported Sharpe ratio');ax.set_ylim(0,.68);ax.legend(frameon=False,loc='upper right',fontsize=9);finish(fig,'rates_rotation.png','Source: executed Rates v0.2.5. Conditional rotation did not beat the simpler benchmarks.')
print(metrics.to_string(index=False));print('Figures:',len(list(F.glob('*.png'))))
