"""ERP 0.12.2: common-calendar monetary shock study of *realized rolling* FMPs.
No backfilling RP returns; no retrospective selection of RP12_PC2. The new
holdout is retrospective research, not an untouched confirmation sample.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .equity_v012 import _fmb_train_oos, _gmm_train_oos

@dataclass
class ShockFollowup:
    mapping: pd.DataFrame
    pricing: pd.DataFrame
    lambdas: pd.DataFrame
    premium_mapping: pd.DataFrame
    calendar: pd.DataFrame
    metadata: dict
    def export(self, folder):
        p=Path(folder);p.mkdir(parents=True,exist_ok=True)
        for name in ('mapping','pricing','lambdas','premium_mapping','calendar'):
            getattr(self,name).to_csv(p/(name+'.csv'),index=False)
        (p/'manifest.json').write_text(json.dumps(self.metadata,indent=2))
        return p

def _monthly(x):
    x=x.copy().sort_index()
    if not isinstance(x.index,pd.DatetimeIndex) or x.index.has_duplicates or not (x.index==x.index+pd.offsets.MonthEnd(0)).all():
        raise ValueError('Require unique month-end DatetimeIndex')
    if x.columns.has_duplicates:raise ValueError('Duplicate factor/asset labels')
    return x.replace([np.inf,-np.inf],np.nan)

def _fit(y,x,tr):
    z=sm.add_constant(x,has_constant='add'); ztr=z.loc[tr]
    if np.linalg.matrix_rank(ztr.to_numpy())<ztr.shape[1]:
        raise ValueError('Shock mapping rank deficient')
    fit=sm.OLS(y.loc[tr],ztr).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    pred=fit.predict(z.loc[~tr]); actual=y.loc[~tr]
    return fit,float(np.mean((actual-pred)**2))

def analyze_rolling_shocks(ff, rp, holdout, shocks, *, train_end='2011-12-31', as_of='2026-10-31', min_train=60,min_test=36, group_label='JK'):
    """All FF and rolling RP series share the same calendar per experiment.

    Contemporaneous regressions (including FF-controlled RP mapping) are
    descriptive. Lagged shocks use calendar t-1, not the prior observed row.
    Pricing uses actual dynamic FMP returns in both formation and validation;
    estimates are fixed after the dedicated formation period. FMB and GMM
    are diagnostic, not weak-identification-robust tests.
    """
    ff,rp,holdout=map(_monthly,(ff,rp,holdout)); s=_monthly(shocks.shocks if hasattr(shocks,'shocks') else shocks)
    if rp.dropna(how='any').empty or s.empty:raise ValueError('Rolling returns and shocks required')
    if set(ff)&set(rp) or (set(ff)|set(rp))&set(s) or set(holdout)&(set(ff)|set(rp)|set(s)):
        raise ValueError('Overlapping column labels')
    end=pd.Timestamp(as_of); cutoff=pd.Timestamp(train_end)
    start=rp.dropna(how='any').index.min()
    dates=pd.date_range(start,min(end,ff.index.max(),rp.index.max(),holdout.index.max(),s.index.max()),freq='ME')
    panel=pd.concat([ff,rp,holdout],axis=1).reindex(dates)
    now=s.reindex(dates);lag=s.shift(1,freq='ME').reindex(dates)
    common=panel.notna().all(axis=1)&now.notna().all(axis=1)
    lagcommon=panel.notna().all(axis=1)&lag.notna().all(axis=1)
    # Use identical dates for both contemporaneous and lagged mapping, pricing.
    valid=common&lagcommon
    audit=pd.DataFrame({'date':dates,'returns_complete':panel.notna().all(axis=1).values,
       'shock_t_complete':now.notna().all(axis=1).values,'shock_t_minus_1_complete':lag.notna().all(axis=1).values,
       'included':valid.values,'period':np.where(dates<=cutoff,'train','oos')})
    p=panel.loc[valid]; now=now.loc[valid];lag=lag.loc[valid];tr=p.index<=cutoff
    if tr.sum()<min_train or (~tr).sum()<min_test:raise ValueError(f'Common calendar insufficient: train={tr.sum()}, OOS={(~tr).sum()}')
    meta={'train_start':str(p.index[tr][0].date()),'train_end':str(p.index[tr][-1].date()),'oos_start':str(p.index[~tr][0].date()),'oos_end':str(p.index[~tr][-1].date()),'n_train':int(tr.sum()),'n_oos':int((~tr).sum()),'requested_cutoff':train_end,'excluded_months':int((~valid).sum()),'inference':'HAC and paired 12-observation block bootstrap on complete-case observations, conditional on fixed formation estimates; no training uncertainty or multiple-test adjustment; missing calendar months compress HAC lags. No Shanken or weak identification correction; GMM shrinkage no standard J-test.','economic_group':group_label,'design':'Retrospective common-sample diagnostic; OOS period already inspected in prior releases; no new independent confirmation.'}
    mapping=[]
    for name in [*ff.columns,*rp.columns]:
        y=p[name]; meanloss=float(np.mean((y.loc[~tr]-y.loc[tr].mean())**2))
        for label,x in [('contemporaneous',now),('lag1',lag)]:
            fit,loss=_fit(y,x,tr)
            row={'factor':name,'specification':label,'n_train':int(tr.sum()),'n_oos':int((~tr).sum()),'train_r2':fit.rsquared,'oos_r2_vs_train_mean':1-loss/meanloss if meanloss>0 else np.nan}
            for col in s:row.update({f'beta_{col}':fit.params[col],f't_{col}':fit.tvalues[col],f'p_{col}':fit.pvalues[col]})
            mapping.append(row)
        if name in rp:
            controls=p[list(ff.columns)]; f0,l0=_fit(y,controls,tr);f1,l1=_fit(y,pd.concat([controls,now],axis=1),tr)
            row={'factor':name,'specification':'contemporaneous_conditional_on_FF','n_train':int(tr.sum()),'n_oos':int((~tr).sum()),'train_r2':f1.rsquared,'oos_r2_vs_train_mean':1-l1/meanloss if meanloss>0 else np.nan,'oos_r2_increment_vs_FF':1-l1/l0 if l0>0 else np.nan,'train_alpha_FF_ann':12*f0.params['const'],'train_alpha_FF_economic_ann':12*f1.params['const']}
            # Ex-post OOS alpha diagnostic on identical realized months, NOT a forecast.
            for label,xx in [('FF',controls),('FF_economic',pd.concat([controls,now],axis=1))]:
                oosfit=sm.OLS(y.loc[~tr],sm.add_constant(xx.loc[~tr],has_constant='add')).fit(cov_type='HAC',cov_kwds={'maxlags':6})
                row[f'oos_expost_alpha_{label}_ann']=12*oosfit.params['const']
                row[f'oos_expost_alpha_{label}_t']=oosfit.tvalues['const']
                row[f'oos_expost_alpha_{label}_p']=oosfit.pvalues['const']
            for col in s:row.update({f'beta_{col}':f1.params[col],f't_{col}':f1.tvalues[col],f'p_{col}':f1.pvalues[col]})
            mapping.append(row)
    # Separate prespecified RP families; do not stack RP0 and RP12 duplicates.
    base=p[list(ff.columns)]; groups={'FF':base,'FF_'+group_label:pd.concat([base,now],axis=1)}
    for prefix in sorted({str(c).split('_PC')[0] for c in rp.columns}):
        r=p[[c for c in rp if str(c).split('_PC')[0]==prefix]]
        groups['FF_'+prefix]=pd.concat([base,r],axis=1);groups['FF_'+prefix+'_'+group_label]=pd.concat([base,r,now],axis=1)
    y=p[list(holdout.columns)]; yt=y.loc[tr].to_numpy();yo=y.loc[~tr].to_numpy(); rows=[]; lambdas=[];preds={}
    for name,x in groups.items():
        fp,fl,fd=_fmb_train_oos(yt,yo,x.loc[tr].to_numpy(),x.loc[~tr],hac_lags=6)
        gp,gd=_gmm_train_oos(yt,yo,x.loc[tr].to_numpy(),lags=6)
        fl['model']=name;lambdas.append(fl)
        for est,pred,stat in [('FMB',fp,fd),('GMM',gp,gd)]:
            preds[name,est]=pred;rows.append({'model':name,'estimator':est,'n_train':int(tr.sum()),'n_oos':int((~tr).sum()),**stat})
    pricing=pd.DataFrame(rows)
    # OOS mean pricing errors (not monthly forecast losses).
    # Paired circular-block bootstrap of OOS MEAN-pricing RMSE, conditional on fixed training.
    rng=np.random.default_rng(91); n=len(yo); block=12
    starts=rng.integers(0,n,size=(399,int(np.ceil(n/block))))
    inds=((starts[:,:,None]+np.arange(block))%n).reshape(399,-1)[:,:n]
    mean_draws=yo[inds].mean(axis=1)
    for i,row in pricing.iterrows():
        suffix='_'+group_label
        parent=row.model.removesuffix(suffix) if row.model.endswith(suffix) else 'FF'
        base_err=float(12*np.sqrt(np.mean((yo.mean(0)-preds[parent,row.estimator])**2)))
        pricing.loc[i,'comparison_parent']=parent
        pricing.loc[i,'rmse_improvement_ann']=base_err-row.oos_mean_pricing_rmse_ann
        draws=12*(np.sqrt(np.mean((mean_draws-preds[parent,row.estimator])**2,axis=1))-np.sqrt(np.mean((mean_draws-preds[row.model,row.estimator])**2,axis=1)))
        lo,hi=np.quantile(draws,[.025,.975])
        pricing.loc[i,'paired_mean_rmse_ci_low']=lo
        pricing.loc[i,'paired_mean_rmse_ci_high']=hi
    # Shock-only SDF, calibrated on separate holdout portfolios, no FF controls.
    # Maps shock covariance to factor expected return; NOT causal attribution.
    z=now.loc[tr].to_numpy(); z=z-z.mean(0);sd=z.std(0,ddof=1)
    if np.any(sd<1e-12):raise ValueError('Constant shock in formation')
    z=z/sd; C=yt.T@z/len(z);mu=yt.mean(0);b=np.linalg.lstsq(C,mu,rcond=None)[0]
    prem=[]
    for name in [*ff.columns,*rp.columns]:
        cov=p.loc[tr,name].to_numpy()@z/len(z); contrib=12*cov*b
        row={'factor':name,'shock_only_sdf_implied_ann':float(contrib.sum()),'formation_mean_ann':12*p.loc[tr,name].mean(),'oos_realized_mean_ann':12*p.loc[~tr,name].mean()}
        row.update({f'contribution_{c}_ann':float(v) for c,v in zip(s.columns,contrib)})
        prem.append(row)
    return ShockFollowup(pd.DataFrame(mapping),pricing,pd.concat(lambdas,ignore_index=True),pd.DataFrame(prem),audit,meta)
