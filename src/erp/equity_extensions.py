"""Equity pricing-oriented residual discovery, inference and three-pass diagnostics.

Research diagnostics, not new priced risk identification from eigenvalues alone.
Formation-only factor extraction, fixed neutral weights and held-out pricing.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .equity import EquityData
from .equity_residual import (_build_panel, _coef, _hac_alpha, _holm)


def _aligned(data, baseline):
    f, rf = data.factors(baseline.model)
    raw_d = _build_panel(data, baseline.discovery_universes, rf)
    raw_v = _build_panel(data, baseline.validation_universes, rf)
    panel = pd.concat([f, raw_d, raw_v], axis=1, join='inner').loc[:baseline.as_of].dropna()
    cols = list(f.columns); dc=list(raw_d.columns); vc=list(raw_v.columns)
    tr=panel.index<=baseline.train_end
    return panel[cols],panel[dc],panel[vc],tr


def residual_eigenvalue_null(residual: np.ndarray, *, n_pcs=5, draws=199, seed=7331):
    """Rankwise circular-shift null + max-eigenvalue FWER diagnostic.

    Each asset is independently circularly shifted, preserving its own path and
    short-run time structure, destroying contemporaneous common timing. Not a
    null for shared underlying stocks; not proof of economic pricing.
    """
    e=np.asarray(residual, float)
    if e.ndim!=2 or not np.isfinite(e).all() or draws<19:
        raise ValueError('Need finite T x N residuals and >=19 null draws')
    t,n=e.shape
    if n_pcs>min(n,t): raise ValueError('Too many PC ranks')
    e=e-e.mean(axis=0)
    obs=np.linalg.eigvalsh(e.T@e/(t-1))[::-1][:n_pcs]
    rng=np.random.default_rng(seed)
    sims=np.zeros((draws,n_pcs))
    maxs=np.zeros(draws)
    for s in range(draws):
        shifts=rng.integers(0,t,size=n)
        shuffled=e[(np.arange(t)[:,None]-shifts[None,:])%t, np.arange(n)[None,:]]
        ev=np.linalg.eigvalsh(shuffled.T@shuffled/(t-1))
        sims[s]=ev[-n_pcs:][::-1]
        maxs[s]=ev[-1]
    rows=[]
    for j,val in enumerate(obs):
        rows.append({'factor':f'RPC{j+1}', 'eigenvalue':float(val),
          'null_rank_95':float(np.quantile(sims[:,j],.95)),
          'null_rank_99':float(np.quantile(sims[:,j],.99)),
          'null_max_95_FWER':float(np.quantile(maxs,.95)),
          'p_rank_exploratory':float((1+np.sum(sims[:,j]>=val))/(draws+1)),
          'p_max_FWER_conservative':float((1+np.sum(maxs>=val))/(draws+1)),
          'passes_max_95_FWER':bool(val>np.quantile(maxs,.95))})
    return pd.DataFrame(rows).set_index('factor')


def _principal_angles(x,y):
    qx=np.linalg.qr(x)[0]; qy=np.linalg.qr(y)[0]
    s=np.linalg.svd(qx.T@qy,compute_uv=False)
    return float(np.mean(np.clip(s,0,1)**2))


def _sdf_oos(held_train,held_test,ftrain,ftest,ridge=1e-4):
    yy=held_train.to_numpy(); mu=yy.mean(axis=0)
    xx=ftrain.to_numpy()
    cov=(yy-mu).T@(xx-xx.mean(axis=0))/(len(xx)-1)
    norms=np.maximum(np.linalg.norm(cov,axis=0),1e-12)
    cs=cov/norms
    b=np.linalg.solve(cs.T@cs+ridge*np.eye(len(norms)),cs.T@mu)/norms
    predicted=cov@b
    actual=held_test.mean().to_numpy()
    return {'pricing_rmse_train_ann':float(12*np.sqrt(np.mean((mu-predicted)**2))),
            'pricing_rmse_oos_ann':float(12*np.sqrt(np.mean((actual-predicted)**2))),
            'pricing_mae_oos_ann':float(12*np.mean(np.abs(actual-predicted))),
            'n_assets':int(yy.shape[1])}


def _gx_three_pass(discovery, validation, observed, mask, *, k=5, ridge=1e-8):
    """Formation-only Giglio-Xiu-style three-pass point-estimate diagnostic.

    Pass 1 PCA on centered *asset returns*; Pass 2 cross-sectional pricing of
    latent return exposures; Pass 3 projection of observed traded factors onto
    latent factors. Not the paper's weak-factor robust asymptotic inference.
    """
    dtr=discovery.loc[mask].to_numpy()
    a=validation.loc[mask].to_numpy(); ao=validation.loc[~mask].to_numpy()
    z=observed.loc[mask].to_numpy()
    u, s, vt=np.linalg.svd(dtr-dtr.mean(axis=0),full_matrices=False)
    latent=(dtr-dtr.mean(axis=0))@vt[:k].T
    latent/=np.maximum(latent.std(axis=0,ddof=1),1e-12)
    # Cross-sectional betas on formation validation assets; no OOS inputs.
    B=_coef(np.column_stack([np.ones(len(latent)),latent]),a)[1:].T
    # Cross-sectional lambda using pooled mean returns and an intercept.
    xx=np.column_stack([np.ones(len(B)),B])
    lam=_coef(xx,a.mean(axis=0))
    covprice=float(np.std(lam[1:]))
    eta=_coef(np.column_stack([np.ones(len(latent)),latent]),z)[1:].T
    lambda_observed=eta@lam[1:]
    pred=xx@lam
    oos=ao.mean(axis=0)
    label=['pricing_intercept',*['LPC'+str(j+1) for j in range(k)]]
    observed_tbl=pd.DataFrame({'factor':observed.columns,
        'gx_threepass_lambda_ann':12*lambda_observed,
        'factor_actual_train_mean_ann':12*observed.loc[mask].mean().to_numpy(),
        'factor_actual_oos_mean_ann':12*observed.loc[~mask].mean().to_numpy()}).set_index('factor')
    diag=pd.DataFrame([{'n_latent':k,'n_discovery':discovery.shape[1],
      'n_validation':validation.shape[1], 'n_formation':int(mask.sum()),
      'formation_cs_rmse_ann':float(12*np.sqrt(np.mean((a.mean(axis=0)-pred)**2))),
      'oos_cs_rmse_ann':float(12*np.sqrt(np.mean((oos-pred)**2))),
      'lambda_dispersion_monthly':covprice}])
    return observed_tbl,diag


@dataclass
class EquityFactorExtensionsResult:
    eigen_null: pd.DataFrame
    subspace_stability: pd.DataFrame
    pricing_oriented_weights: pd.DataFrame
    pricing_oriented_premia: pd.DataFrame
    pricing_oriented_sdf: pd.DataFrame
    three_pass_observed: pd.DataFrame
    three_pass_diagnostics: pd.DataFrame
    metadata: dict

    def export(self,path):
        dest=Path(path);dest.mkdir(parents=True,exist_ok=True)
        frames={'residual_eigenvalue_null':self.eigen_null,
            'residual_subspace_stability':self.subspace_stability,
            'pricing_oriented_weights':self.pricing_oriented_weights,
            'pricing_oriented_premia':self.pricing_oriented_premia,
            'pricing_oriented_sdf':self.pricing_oriented_sdf,
            'three_pass_observed_premia':self.three_pass_observed,
            'three_pass_diagnostics':self.three_pass_diagnostics}
        for name,frame in frames.items():frame.to_csv(dest/(name+'.csv'))
        (dest/'manifest.json').write_text(json.dumps(self.metadata,indent=2,default=str))
        return dest


def analyze_equity_factor_extensions(data:EquityData, baseline, *, n_pcs:int=5,
        null_draws:int=199, null_seed:int=7331, rp_gamma:tuple[float,...]=(0.,12.,60.),
        sdf_ridge:float=1e-4, hac_lags:int=6, gx_k:int=5):
    """Post-v0.10 challengers with parameters predeclared, not tuned on OOS.

    The RP-style covariance-plus-alpha penalty is a pricing-oriented *extension*
    of residual PCA, not a full replication of Lettau-Pelger estimator.
    """
    f,d,h,tr=_aligned(data,baseline)
    ntrain=int(tr.sum());nte=int((~tr).sum())
    if ntrain<96 or nte<36:raise ValueError('Insufficient train/OOS observations')
    if n_pcs<1 or n_pcs>min(15,d.shape[1]-f.shape[1]-2):raise ValueError('invalid n_pcs')
    if any((not np.isfinite(g) or g<0) for g in rp_gamma) or len(set(rp_gamma))!=len(rp_gamma):
        raise ValueError('Gamma values must be nonnegative, distinct and predeclared')
    if gx_k<1 or gx_k>min(20,int(tr.sum())-1,d.shape[1]):raise ValueError('invalid gx_k')
    xx=np.column_stack([np.ones(ntrain),f.loc[tr].to_numpy()])
    coef=_coef(xx,d.loc[tr].to_numpy())
    e=d.loc[tr].to_numpy()-xx@coef
    cov=(e.T@e)/(ntrain-1)
    alpha=coef[0]  # formation-only TS intercept, NOT the mean of OLS residuals
    eigen_null=residual_eigenvalue_null(e,n_pcs=n_pcs,draws=null_draws,seed=null_seed)
    half=ntrain//2
    stab=[]
    for xxpart in (slice(None,half),slice(half,None)):
        x=xx[xxpart];a=d.loc[tr].to_numpy()[xxpart]
        err=a-x@_coef(x,a)
        _,_,v=np.linalg.svd(err,full_matrices=False)
        stab.append(v[:n_pcs].T)
    stability=pd.DataFrame([{'n_pcs':k,'formation_half_subspace_overlap_mean_cos2':_principal_angles(stab[0][:,:k],stab[1][:,:k]),
      'null_interpretation':'Half formation subsamples only; no guarantee of factor economic persistence'} for k in range(1,n_pcs+1)])
    constraints=np.column_stack([np.ones(d.shape[1]),coef[1:].T])
    neutral_proj=np.eye(d.shape[1])-constraints@np.linalg.pinv(constraints)
    ftr=f.loc[tr];fte=f.loc[~tr];htr=h.loc[tr];hte=h.loc[~tr]
    rows=[]; sdf=[]; wt={}; pmx=[]
    for gamma in rp_gamma:
        matrix=cov+gamma*np.outer(alpha,alpha)
        evals,vecs=np.linalg.eigh(matrix)
        ev=evals[::-1]; vv=vecs[:,::-1][:,:n_pcs]
        weights=[]
        for j in range(n_pcs):
            v=vv[:,j].copy()
            imax=int(np.argmax(np.abs(v)))
            if v[imax]<0:v=-v
            w=neutral_proj@v
            gross=np.sum(np.abs(w))
            if gross<1e-10: raise ValueError('RP candidate projected almost entirely into FF span')
            w=w/gross
            weights.append(w)
            name=f'RP{gamma:g}_PC{j+1}'
            wt[name]=pd.Series(w,index=d.columns)
            realized=pd.Series(d.to_numpy()@w,index=d.index,name=name)
            tst=_hac_alpha(realized.loc[~tr],fte,hac_lags)
            mean=_hac_alpha(realized.loc[~tr],None,hac_lags)
            exbet=_coef(np.column_stack([np.ones(nte),fte.to_numpy()]),realized.loc[~tr].to_numpy())[1:]
            rows.append({'gamma':gamma,'factor':name,'pc':j+1,
                'eigenvalue_pricing_weighted_train':float(ev[j]),
                'factor_train_alpha_ann':float(12*realized.loc[tr].mean()),
                'oos_mean_ann':float(mean['alpha_monthly']*12),
                'oos_mean_t_hac':mean['t_hac'],'oos_mean_p_hac':mean['p_hac'],
                'oos_alpha_ann':float(tst['alpha_monthly']*12),
                'oos_alpha_t_hac':tst['t_hac'],'oos_alpha_p_hac':tst['p_hac'],
                'oos_beta_max_abs':float(np.max(np.abs(exbet))),
                'train_beta_max_abs':float(np.max(np.abs(constraints.T@w))),
                'n_oos':nte})
        new=pd.DataFrame(d.to_numpy()@np.column_stack(weights),index=d.index,
                         columns=[f'RP{gamma:g}_PC{j+1}' for j in range(n_pcs)])
        for k in (0,1,n_pcs):
            chosen=f if k==0 else pd.concat([f,new.iloc[:,:k]],axis=1)
            score=_sdf_oos(htr,hte,chosen.loc[tr],chosen.loc[~tr],sdf_ridge)
            sdf.append({'gamma':gamma,'n_priced_components':k,**score})
    pmx=pd.DataFrame(rows).set_index('factor')
    # All tested methods × K: multiplicity adjusted across entire predeclared family.
    pmx['holm_adjusted_oos_alpha_p_all_tests']=_holm(pmx['oos_alpha_p_hac'])
    weights_df=pd.DataFrame(wt)
    tpo,tdiag=_gx_three_pass(d,h,f,tr,k=gx_k)
    meta={'version':'0.11.0','research':'Formation-only pricing-weighted residual PCA benchmark and three-pass diagnostic',
      'n_train_months':ntrain,'n_oos_months':nte,'train_end':str(baseline.train_end.date()),
      'null_draws':null_draws,'null_seed':null_seed,'n_pcs':n_pcs,'rp_gamma':list(rp_gamma),
      'gamma_selection':'Predeclared; no validation-based hyperparameter selection',
      'rp_pca_caveat':'RP-PCA-inspired covariance-plus-training-alpha penalty, NOT full Lettau-Pelger estimator; alpha not OLS residual mean',
      'pricing_caveat':'Incremental alpha / small SDF RMSE alone does not identify a new priced systematic factor',
      'gx_caveat':'Giglio-Xiu-style three-pass point estimate; NOT weak-factor robust inference or exact published estimator',
      'eigen_null':'Assetwise independent circular shifts; controls max eigen under no synchronous residual dependence, not underlying holdings overlap',
      'factor_weights':'Unrebalanced self-financing FMP in characteristic sorted portfolios, gross 1, FF beta neutral in train only',
      'sampling':'Fixed formation/OOS; all preprocessing and candidate sign decisions done in formation; revised data not PIT'}
    return EquityFactorExtensionsResult(eigen_null,stability,weights_df,pmx,pd.DataFrame(sdf),tpo,tdiag,meta)
