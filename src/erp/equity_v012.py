"""ERP 0.12: rolling *priced-residual-risk challengers* and comparable OOS pricing.

The RP construction is an explicit augmented-mean objective after partialling out
FF/AQR; it is NOT the complete Lettau--Pelger published asymptotic estimator.
No future returns or prices enter a portfolio weight before the holding month.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .equity import EquityData
from .equity_extensions import _aligned
from .equity_residual import _coef, _hac_alpha


def _nw_mean(x, lags=6):
    a=np.asarray(x,dtype=float)
    a=a[np.isfinite(a)]
    if len(a)<12:return (float('nan'),float('nan'),float('nan'))
    fit=sm.OLS(a,np.ones((len(a),1))).fit(cov_type='HAC',cov_kwds={'maxlags':min(lags,len(a)-2)})
    return float(fit.params[0]),float(fit.tvalues[0]),float(fit.pvalues[0])


def _factor_weights(y, f, *, n_pcs, gamma, anchor=None):
    """Training-only augmentation of *partially regressed returns*.

    Write Y_t = alpha + beta F_t + e_t. After partialling out beta F_t,
    Ytilde_t=alpha+e_t. Its second moment is Cov(e)+alpha alpha', with a
    gamma-weighted mean penalty M=Cov(e)+gamma alpha alpha'. Eigenvectors are
    exactly the right singular vectors of [E/sqrt(T-1);sqrt(gamma)*alpha']. 
    This uses the priced mean moment explicitly, rather than the (zero) mean
    of OLS residuals. It is a conditional/residual adaptation of RP-PCA.
    """
    n,p=y.shape; nf=f.shape[1]
    if n<=max(40,5*(nf+1)) or p<=nf+n_pcs+2:raise ValueError('Insufficient formation data')
    x=np.column_stack([np.ones(n),f]); coef=_coef(x,y)
    alpha=coef[0]; err=y-x@coef
    aug=np.vstack([err/np.sqrt(n-1),np.sqrt(gamma)*alpha.reshape(1,-1)])
    # Eigenspace of augmented covariance, factor extracted from true returns
    m=aug.T@aug
    eig,vec=np.linalg.eigh(m); vec=vec[:,::-1][:,:n_pcs].copy()
    con=np.column_stack([np.ones(p),coef[1:].T])
    neutral=np.eye(p)-con@np.linalg.pinv(con)
    weights=[]
    for j in range(n_pcs):
        v=neutral@vec[:,j]
        if np.sum(np.abs(v))<1e-10:raise ValueError('No FF-neutral factor direction')
        v/=np.abs(v).sum()
        # Anchor sign to PREVIOUS *formation*, never returns of holding period.
        if anchor is not None and float(v@anchor[:,j])<0:v=-v
        elif anchor is None and v[np.argmax(np.abs(v))]<0:v=-v
        weights.append(v)
    return np.column_stack(weights),eig[::-1][:n_pcs],con


def _fmb_train_oos(ytrain, ytest, ftrain, ftest, *, hac_lags=6):
    """Conventional two-pass with formation betas fixed and time-series CS prices.

    Pricing across validation assets; beta estimated only on formation. Not
    Shanken-adjusted, measurement-error-corrected, nor overlapping-asset robust.
    """
    T,N=ytrain.shape;K=ftrain.shape[1]
    b=_coef(np.column_stack([np.ones(T),ftrain]),ytrain)[1:].T
    design=np.column_stack([np.ones(N),b]);rank=int(np.linalg.matrix_rank(design))
    cond=float(np.linalg.cond(design/np.maximum(design.std(axis=0),1e-12))) # intercept gets scale 1e-12; condition meaningless
    scaled=np.column_stack([np.ones(N),b/np.maximum(b.std(axis=0),1e-10)])
    cond=float(np.linalg.cond(scaled))
    lam=np.linalg.lstsq(design,ytrain.T,rcond=None)[0].T
    lmean=lam.mean(axis=0)
    pm=design@lmean
    truth=ytest.mean(axis=0)
    rows=[]
    for j,name in enumerate(['intercept',*list(ftest.columns)]):
        mu,t,pv=_nw_mean(lam[:,j],hac_lags)
        rows.append({'factor':name,'lambda_training_ann':12*mu,'t_hac':t,'p_hac':pv,
                     'n_train_months':len(lam),'beta_partial_dispersion':float(np.std(b[:,j-1])) if j else float('nan')})
    summary={'method':'Fama-MacBeth two-pass, fixed formation betas',
             'n_assets':N,'n_factors':K,'rank':rank,'condition_scaled':cond,
             'oos_mean_pricing_rmse_ann':float(12*np.sqrt(np.mean((truth-pm)**2)))}
    return pm,pd.DataFrame(rows),summary


def _hac_moment_cov(m, lags=6):
    x=np.asarray(m,float);n=x.shape[0]
    x=x-x.mean(axis=0)
    omega=x.T@x/n
    for ell in range(1,min(lags,n-2)+1):
        w=1-ell/(min(lags,n-2)+1)
        c=x[ell:].T@x[:-ell]/n
        omega+=w*(c+c.T)
    return (omega+omega.T)/2


def _gmm_train_oos(ytrain, ytest, ftrain, *, ridge=1e-4, lags=6):
    """SDF moment condition E[R(1-b'F_centered)] = 0; two-step ridge GMM.

    HAC weight shrunk toward diagonal for positive definiteness; do NOT
    interpret usual chi-square J-test asymptotics as valid with this shrinkage.
    """
    yy=np.asarray(ytrain,float);ff=np.asarray(ftrain,float)
    n,N=yy.shape;K=ff.shape[1]
    z=ff-ff.mean(axis=0)
    sd=np.maximum(z.std(axis=0,ddof=1),1e-9)
    z=z/sd
    C=yy.T@z/n
    mu=yy.mean(axis=0)
    scale=max(float(np.mean(mu**2)),1e-12)
    ridge_eff=ridge*max(float(np.trace(C.T@C)/K),1e-10)
    b1=np.linalg.solve(C.T@C+ridge_eff*np.eye(K),C.T@mu)
    m=yy*(1-z@b1)[:,None]
    omega=_hac_moment_cov(m,lags)
    shrink=.5
    omega=(1-shrink)*omega+shrink*(np.trace(omega)/N)*np.eye(N)
    W=np.linalg.pinv(omega,rcond=1e-9)
    norm=float(np.trace(C.T@W@C)/K)
    b2=np.linalg.solve(C.T@W@C+ridge*max(norm,1e-10)*np.eye(K),C.T@W@mu)
    pred=C@b2
    truth=np.asarray(ytest,float).mean(axis=0)
    out={'method':'two-step HAC-weighted shrinkage GMM SDF',
         'n_assets':N,'n_factors':K,'weight_shrinkage':shrink,'ridge':ridge,
         'oos_mean_pricing_rmse_ann':float(12*np.sqrt(np.mean((truth-pred)**2))),
         'formation_moment_rmse_ann':float(12*np.sqrt(np.mean((mu-pred)**2))),
         'weight_condition':float(np.linalg.cond(omega)),
         'gmm_j_inference':'NOT_VALIDATED: shrinkage + overlapping characteristic portfolios'}
    return pred,out


def _paired_block_bootstrap(loss_base,loss_alt,*,block=12,draws=399,seed=91):
    """Paired chronological-month resampling of MSE difference, never resample assets."""
    b=np.asarray(loss_base,float);a=np.asarray(loss_alt,float)
    if b.shape!=a.shape or b.ndim!=1 or len(b)<24:raise ValueError('paired losses require >=24 equal monthly observations')
    difference=b-a; n=len(b);rng=np.random.default_rng(seed)
    realized=float(difference.mean());sims=np.zeros(draws)
    for j in range(draws):
        inds=[]
        while len(inds)<n:
            start=int(rng.integers(0,n))
            inds.extend((start+np.arange(block))%n)
        sims[j]=difference[np.asarray(inds[:n],int)].mean()
    lo,hi=np.quantile(sims,[.025,.975])
    return {'mse_improvement_monthly':realized,'block_bootstrap_ci_low':float(lo),
            'block_bootstrap_ci_high':float(hi),'ci_excludes_zero':bool(lo>0 or hi<0),
            'draws':draws,'block_months':block}


@dataclass
class EquityV012Result:
    rolling_weights:pd.DataFrame
    rolling_returns:pd.DataFrame
    rolling_diagnostics:pd.DataFrame
    rolling_premia:pd.DataFrame
    pricing_comparison:pd.DataFrame
    fmb_lambdas:pd.DataFrame
    bootstrap_comparison:pd.DataFrame
    metadata:dict

    def export(self,root):
        out=Path(root);out.mkdir(parents=True,exist_ok=True)
        for key in ['rolling_weights','rolling_returns','rolling_diagnostics','rolling_premia',
                    'pricing_comparison','fmb_lambdas','bootstrap_comparison']:
            getattr(self,key).to_csv(out/(key+'.csv'))
        (out/'manifest.json').write_text(json.dumps(self.metadata,indent=2,default=str))
        return out


def analyze_equity_v012(data:EquityData,baseline,*,formation_window=120,
          rebalance_months=12,n_pcs=3,gammas=(0.,12.),gmm_ridge=1e-4,
          bootstrap_draws=399,seed=91,hac_lags=6):
    """Rolling RP challengers, same-sample pricing horse race.

    Pricing horse race uses *fixed 2005 formation* (first available OOS month)
    with factor returns generated by frozen formation weights. Rolling factor
    returns are a SEPARATE dynamic-portfolio analysis, not fed back into the
    fixed-formation horse race to avoid training/feature mismatch.
    """
    f,d,h,tr=_aligned(data,baseline)
    d=d.astype(float);h=h.astype(float);f=f.astype(float)
    start=int(tr.sum());N=len(f)
    if start<formation_window or N-start<36:raise ValueError('insufficient formation/OOS months')
    if n_pcs>min(10,d.shape[1]-f.shape[1]-2):raise ValueError('too many factors')
    if any(g<0 for g in gammas) or len(set(gammas))!=len(gammas):raise ValueError('invalid gammas')
    # -- Rolling weights: each holding month t uses weights formed through t-1.
    factors=[f'RP{g:g}_PC{j+1}' for g in gammas for j in range(n_pcs)]
    rollret=pd.DataFrame(index=f.index[start:],columns=factors,dtype=float)
    wt_rows=[];diag=[]; prev={g:None for g in gammas};prev_w={}
    for pos in range(start,N,rebalance_months):
        dt_train=f.index[pos-formation_window:pos]
        dt_hold=f.index[pos:min(pos+rebalance_months,N)]
        y=d.loc[dt_train].to_numpy();x=f.loc[dt_train].to_numpy()
        for gamma in gammas:
            old=prev[gamma]
            W,ev,C=_factor_weights(y,x,n_pcs=n_pcs,gamma=gamma,anchor=old)
            if old is None:
                stability=float('nan')
            else:
                qold=np.linalg.qr(old)[0];qnew=np.linalg.qr(W)[0]
                stability=float(np.mean(np.clip(np.linalg.svd(qold.T@qnew,compute_uv=False),0,1)**2))
            prev[gamma]=W
            for j in range(n_pcs):
                name=f'RP{gamma:g}_PC{j+1}';w=W[:,j]
                rollret.loc[dt_hold,name]=d.loc[dt_hold].to_numpy()@w
                change=float(np.abs(w-prev_w[name]).sum()) if name in prev_w else np.nan
                prev_w[name]=w.copy()
                diag.append({'formed_as_of':str(dt_train[-1].date()),'holding_from':str(dt_hold[0].date()),
                    'holding_through':str(dt_hold[-1].date()),'factor':name,'gamma':gamma,
                    'eigenvalue':float(ev[j]),'training_beta_max_abs':float(np.max(np.abs(C.T@w))),
                    'net_dollar':float(w.sum()),'gross':float(np.abs(w).sum()),
                    'l1_turnover_on_rebalance':change,
                    'rolling_subspace_overlap_mean_cos2':stability,
                    'direction_cosine_with_previous_formation':
                        float(w@old[:,j]/(np.linalg.norm(w)*np.linalg.norm(old[:,j])))
                        if old is not None else np.nan})
                wt_rows.extend({'formed_as_of':dt_train[-1],'factor':name,'asset':asset,'weight':float(val)}
                    for asset,val in zip(d.columns,w))
    prem=[]
    for name in factors:
        s=rollret[name].dropna();mean,t,pv=_nw_mean(s,hac_lags)
        alp=_hac_alpha(s,f.loc[s.index],hac_lags)
        prem.append({'factor':name,'oos_mean_ann':12*mean,'oos_mean_t_hac':t,'oos_mean_p_hac':pv,
            'oos_alpha_ann':12*alp['alpha_monthly'],'oos_alpha_t_hac':alp['t_hac'],
            'oos_alpha_p_hac':alp['p_hac'],'oos_months':len(s)})
    # -- Common formation: freeze portfolios and compare *all* models OOS identically.
    ytrain=d.loc[tr].to_numpy();xtrain=f.loc[tr].to_numpy()
    ftr=f.loc[tr]; fte=f.loc[~tr];htr=h.loc[tr];hte=h.loc[~tr]
    price_rows=[];lambdas=[];predictions={}; truth=hte.to_numpy()
    # Prespecified 3 candidates only: baseline, PCA K, augmented-mean RP K.
    candidates=[('FF_AQR',None)]
    for gamma in gammas:
        W,_,_=_factor_weights(ytrain,xtrain,n_pcs=n_pcs,gamma=gamma)
        gtr=pd.DataFrame(ytrain@W,index=ftr.index,columns=[f'RP{gamma:g}_PC{i+1}' for i in range(n_pcs)])
        gte=pd.DataFrame(d.loc[~tr].to_numpy()@W,index=fte.index,columns=gtr.columns)
        candidates.append((f'FF_AQR_RP{gamma:g}',(gtr,gte)))
    for name,additional in candidates:
        if additional is None:
            train=ftr;test=fte
        else:
            train=pd.concat([ftr,additional[0]],axis=1)
            test=pd.concat([fte,additional[1]],axis=1)
        fp,fl,fd=_fmb_train_oos(htr.to_numpy(),hte.to_numpy(),train.to_numpy(),test,hac_lags=hac_lags)
        gp,gd=_gmm_train_oos(htr.to_numpy(),hte.to_numpy(),train.to_numpy(),ridge=gmm_ridge,lags=hac_lags)
        for method,pred,stat in [('FMB',fp,fd),('GMM',gp,gd)]:
            predictions[(name,method)]=pred
            price_rows.append({'model':name,'estimator':method,**stat,
                'train_end':str(ftr.index[-1].date()),'oos_start':str(fte.index[0].date()),
                'oos_end':str(fte.index[-1].date()),'oos_months':len(hte)})
        fl['model']=name;lambdas.append(fl)
    comparisons=[]
    for estimator in ['FMB','GMM']:
        base=predictions[('FF_AQR',estimator)]
        loss_b=np.mean((truth-base[None,:])**2,axis=1)
        for name,additional in candidates[1:]:
            pred=predictions[(name,estimator)]
            loss_a=np.mean((truth-pred[None,:])**2,axis=1)
            comparisons.append({'model':name,'estimator':estimator,
                **_paired_block_bootstrap(loss_b,loss_a,draws=bootstrap_draws,seed=seed)})
    meta={'version':'0.12.0','model':baseline.model,'as_of':str(baseline.as_of.date()),
          'train_end':str(baseline.train_end.date()),'rolling_window':formation_window,
          'rebalance_months':rebalance_months,'n_pcs':n_pcs,'gammas_predeclared':list(gammas),
          'discovery_portfolios':list(d.columns),'heldout_portfolios':list(h.columns),
          'formation_months':start,'oos_months':len(hte),'pricing_train_holdout_only':True,
          'rp_limit':'augmented-mean RP-PCA challenger after FF partialling; not full published asymptotics',
          'fmb_limit':'fixed formation time-series beta, CS prices month by month; no Shanken correction',
          'gmm_limit':'HAC-weighted shrunk two-step SDF GMM; no valid J-statistic provided',
          'overlap':'different sorted portfolios can contain same underlying stocks',
          'pricing_horse_race':'fixed FMP weights trained before OOS; common holdout assets and dates',
          'rolling_evaluation':'dynamic FMP, methodologically separate from fixed formation pricing',
          'publication_vintage':'historical returns and FRED revisions not point-in-time'}
    return EquityV012Result(pd.DataFrame(wt_rows),rollret,pd.DataFrame(diag),
          pd.DataFrame(prem).set_index('factor'),pd.DataFrame(price_rows),
          pd.concat(lambdas,ignore_index=True),pd.DataFrame(comparisons),meta)
