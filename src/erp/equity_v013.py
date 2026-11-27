"""Dynamic mean pricing, shrinkage, constrained RP-PCA and three-pass estimates.
All forecasts use calendar dates strictly before realization. Research, not PIT.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment
from .equity_v012 import _hac_moment_cov
from .equity_shock_followup import _monthly
from .equity_extensions import _aligned


def rp_basis(y, gamma=0., constraints=None, k=3):
    """Lettau-Pelger balanced-panel matrix E[rr'] + gamma mu mu'.
    gamma=-1 is centered PCA. Optional constraints restrict the objective ex ante.
    """
    y=np.asarray(y,float); T,N=y.shape
    if gamma < -1 or k<1: raise ValueError('gamma >= -1 and positive k required')
    Q=np.eye(N) if constraints is None else null_space(np.asarray(constraints).T)
    if Q.shape[1]<k: raise ValueError('Insufficient neutral space')
    X=y@Q; mu=X.mean(0); mat=X.T@X/T+gamma*np.outer(mu,mu)
    ev,v=np.linalg.eigh(mat); return Q@v[:,::-1][:,:k],ev[::-1][:k]


def constrained_rp_rolling(data, baseline, *, window=120, step=12, k=3, gammas=(-1.,11.)):
    """Fixed neutral subspace each formation; gross-one actual FMP returns.
    -1 and 11 are matched mean weights 0 and 12 under LP convention.
    """
    f,d,h,tr=_aligned(data,baseline); start=int(tr.sum()); ret={}; rows=[]; weights=[]
    if start<window: raise ValueError('RP formation history insufficient')
    for gamma in gammas:
        prev=None; names=[f'CRP{gamma:g}_PC{i+1}' for i in range(k)]
        for t in range(start,len(d),step):
            y=d.iloc[t-window:t].to_numpy(); z=f.iloc[t-window:t].to_numpy()
            beta=np.linalg.lstsq(np.column_stack([np.ones(len(y)),z]),y,rcond=None)[0][1:]
            C=np.column_stack([np.ones(y.shape[1]),beta.T]); V,ev=rp_basis(y,gamma,C,k)
            if prev is not None:
                a,b=linear_sum_assignment(-np.abs(prev.T@V)); V=V[:,b[np.argsort(a)]]; ev=ev[b[np.argsort(a)]]
                V*=np.where(np.sum(prev*V,axis=0)<0,-1,1)
                stability=np.mean(np.linalg.svd(prev.T@V,compute_uv=False)**2)
            else:
                V*=np.where(V[np.argmax(np.abs(V),axis=0),np.arange(k)]<0,-1,1);stability=np.nan
            W=V/np.abs(V).sum(0)
            hold=d.iloc[t:t+step]; frame=pd.DataFrame(hold.to_numpy()@W,index=hold.index,columns=names)
            ret.setdefault(gamma,[]).append(frame)
            for j,name in enumerate(names):
                rows.append(dict(formation=d.index[t-1],holding_from=hold.index[0],factor=name,eigenvalue=ev[j],subspace_stability=stability,max_constraint_error=np.abs(C.T@W[:,j]).max(),turnover=np.abs(W[:,j]-prev[:,j]/np.abs(prev[:,j]).sum()).sum() if prev is not None else np.nan))
                weights.append(pd.Series(W[:,j],index=d.columns,name=(d.index[t-1],name)))
            prev=V
    return pd.concat([pd.concat(v) for v in ret.values()],axis=1),pd.DataFrame(rows),pd.DataFrame(weights).rename_axis(index=['formation','factor'])


def _posterior(mean, covariance, prior_sd):
    """Gaussian plug-in posterior N(0,D), known HAC mean covariance estimate."""
    D=np.diag(np.asarray(prior_sd)**2); V=(covariance+covariance.T)/2
    e,q=np.linalg.eigh(V); V=(q*np.maximum(e,0))@q.T
    return D@np.linalg.solve(D+V,mean),D-D@np.linalg.solve(D+V,D)


def _fmb(y,f,shrink=False):
    n=len(y); x=np.column_stack([np.ones(n),f]); beta=np.linalg.lstsq(x,y,rcond=None)[0][1:].T
    A=np.column_stack([np.ones(len(beta)),beta]); lam=np.linalg.lstsq(A,y.T,rcond=None)[0].T
    mu=lam.mean(0); V=_hac_moment_cov(lam,6)/n
    if shrink:
        # Weak intercept prior; 0.25 monthly Sharpe-unit prior for factor prices.
        sd=np.r_[.10,np.maximum(.25*np.std(f,axis=0,ddof=1),1e-8)]
        mu,V=_posterior(mu,V,sd)
    return A@mu,mu[1:],np.sqrt(np.maximum(np.diag(V)[1:],0)),float(np.linalg.cond(np.column_stack([np.ones(len(beta)),beta/np.maximum(beta.std(0),1e-12)])))


def _sdf(y,f):
    n,N=y.shape; sd=f.std(0,ddof=1);z=(f-f.mean(0))/np.maximum(sd,1e-12); C=y.T@z/n;mu=y.mean(0);K=f.shape[1]
    ridge=1e-4*max(np.trace(C.T@C)/K,1e-12); b=np.linalg.solve(C.T@C+ridge*np.eye(K),C.T@mu)
    omega=_hac_moment_cov(y*(1-z@b)[:,None],6);omega=.5*omega+.5*np.trace(omega)/N*np.eye(N); W=np.linalg.pinv(omega,rcond=1e-9)
    H=C.T@W@C; b=np.linalg.solve(H+1e-4*max(np.trace(H)/K,1e-12)*np.eye(K),C.T@W@mu)
    return C@b,(f-f.mean(0)).T@z@b/n,np.full(K,np.nan),float(np.linalg.cond(H))


def gx_three_pass(discovery, test_assets, observed, k):
    """Centered PCA, no-intercept cross-sectional gamma, observed projection.
    Point estimate of observed prices, separate heldout asset mean prediction.
    """
    d=np.asarray(discovery,float); y=np.asarray(test_assets,float); g=np.asarray(observed,float)
    n=len(d); X=d-d.mean(0)
    # Small Gram eigenproblem, common time factors normalized V'V/T=I.
    cov=X.T@X/n; e,v=np.linalg.eigh(cov); vals=e[::-1][:k]
    if k>min(n-1,d.shape[1]) or np.any(vals<=1e-12):raise ValueError('Insufficient latent rank')
    V=X@v[:,::-1][:,:k]/np.sqrt(vals)
    B=y.T@V/n; gamma=np.linalg.lstsq(B,y.mean(0),rcond=None)[0]
    eta=g.T@V/n; fitted=V@eta.T
    r2=1-((g-g.mean(0)-fitted)**2).sum(0)/np.maximum(((g-g.mean(0))**2).sum(0),1e-20)
    return B@gamma,eta@gamma,r2,float(np.linalg.cond(B))


def gx_bootstrap(d,y,f,*,k=8,draws=199,block=12,seed=91):
    """Refit all three passes on synchronized circular block samples.
    Exploratory intervals; not the paper's weak-identification asymptotics.
    """
    d,y,f=map(np.asarray,(d,y,f));n=len(d);rng=np.random.default_rng(seed);samples=[]
    p,mu,r2,cond=gx_three_pass(d,y,f,k)
    for _ in range(draws):
        ix=((rng.integers(n,size=int(np.ceil(n/block)))[:,None]+np.arange(block))%n).ravel()[:n]
        try:samples.append(gx_three_pass(d[ix],y[ix],f[ix],k)[1])
        except (ValueError,np.linalg.LinAlgError):continue
    if len(samples)<.8*draws:raise ValueError('Too many rank-deficient bootstrap samples')
    lo,hi=np.quantile(samples,[.025,.975],axis=0)
    return pd.DataFrame(dict(lambda_ann=12*mu,ci_low_ann=12*lo,ci_high_ann=12*hi,projection_r2=r2,condition=cond,successful_draws=len(samples)))


def _group(ff,rp,extras):
    groups={'FF':list(ff)}
    for prefix in sorted({str(c).split('_PC')[0] for c in rp}):
        cols=list(ff)+[c for c in rp if str(c).split('_PC')[0]==prefix];groups['FF_'+prefix]=cols
    if len(extras):
        groups.update({name+'_EXTRA':cols+list(extras) for name,cols in list(groups.items()) if name in ('FF','FF_RP12')})
    return groups


@dataclass
class DynamicPremia:
    forecasts:pd.DataFrame
    prices:pd.DataFrame
    scores:pd.DataFrame
    factor_scores:pd.DataFrame
    bootstrap:pd.DataFrame
    gx:pd.DataFrame
    hedges:pd.DataFrame
    calendar:pd.DataFrame
    metadata:dict
    def export(self,root):
        p=Path(root);p.mkdir(parents=True,exist_ok=True)
        for name in ('forecasts','prices','scores','factor_scores','bootstrap','gx','hedges','calendar'):
            getattr(self,name).to_csv(p/(name+'.csv'),index=False)
        (p/'manifest.json').write_text(json.dumps(self.metadata,indent=2,default=str));return p


def analyze_dynamic_premia(ff,rp,holdout,discovery,*,extras=None,extra_traded=False,as_of='2026-10-31',window=72,min_train=60,step=3,k=8,draws=199,seed=91, include_gx=True):
    """Common sample per panel, same dates for fixed/expanding/rolling estimates.
    Fixed is calibrated at first valid forecast origin (not the old 2011 sample).
    Quarterly refits, monthly predictions; no realized factor at t in predictions.
    Extra factors are orthogonalized to FF on each formation, preserving means.
    """
    ff,rp,y,d=map(_monthly,(ff,rp,holdout,discovery)); extra=_monthly(extras) if extras is not None else pd.DataFrame(index=ff.index)
    groups=_group(ff.columns,rp.columns,extra.columns)
    p=pd.concat([ff,rp,extra,y,d],axis=1).loc[:as_of];
    if p.columns.has_duplicates:raise ValueError('Duplicate panel labels')
    start=max(z.dropna(how='any').index.min() for z in (ff,rp,y,d,extra) if z.shape[1]);end=min(pd.Timestamp(as_of),*(z.index.max() for z in (ff,rp,y,d,extra) if z.shape[1]))
    cal=pd.date_range(start,end,freq='ME');p=p.reindex(cal);ok=p.notna().all(1);p=p.loc[ok]
    if len(p)<min_train+24:raise ValueError('Insufficient common history')
    origins=[i for i in range(len(p)) if (p.index[i]-pd.DateOffset(months=window)).to_period('M')>=cal[0].to_period('M') and (p.index[:i]>p.index[i]-pd.offsets.MonthEnd(window+1)).sum()>=min_train]
    if not origins or len(p)-origins[0]<24:raise ValueError('Insufficient common rolling history')
    first=origins[0];yy=p[list(y)].to_numpy();dd=p[list(d)].to_numpy(); F=p[list(ff)+list(rp)+list(extra)]; frows=[];prows=[];gxrows=[];hedgerows=[];rng=np.random.default_rng(seed)
    # Refit on quarterly observed origins; rolling window is calendar months, not count.
    fixed={}; refit_cache={}
    for i in range(first,len(p),step):
        target=p.index[i]; history={'fixed':np.arange(first),'expanding':np.arange(i),'rolling':np.flatnonzero((p.index[:i]>=target-pd.offsets.MonthEnd(window)))}
        if len(history['rolling'])<min_train:raise ValueError('Rolling common sample fell below minimum')
        hold=range(i,min(i+step,len(p)))
        for scheme,ix in history.items():
            cachekey=(scheme,0 if scheme=='fixed' else i)
            if cachekey not in refit_cache:
                tr=F.iloc[ix].copy(); beta_extra=None
                if len(extra.columns):
                    bx=np.linalg.lstsq(np.column_stack([np.ones(len(ix)),tr[list(ff)].to_numpy()]),tr[list(extra)].to_numpy(),rcond=None)[0][1:]
                    tr.loc[:,list(extra)]-=tr[list(ff)].to_numpy()@bx
                    beta_extra=bx
                    # Audit fitted hedges; no intercept subtraction, covariance neutral in formation.
                    for j,c in enumerate(extra.columns):
                        corr=tr[list(ff)].corrwith(tr[c]).abs().max()
                        row=dict(scheme=scheme,formation=p.index[ix[-1]],factor=c,n_train=len(ix),max_abs_training_correlation=corr)
                        row.update({f'hedge_beta_{v}':x for v,x in zip(ff.columns,bx[:,j])})
                        hedgerows.append(row)
                fit={};means=yy[ix].mean(0)
                for name,cols in groups.items():
                    z=tr[cols].to_numpy();fit[name]={}
                    for est,fun in [('FMB',lambda:_fmb(yy[ix],z)),('FMB_SHRINK',lambda:_fmb(yy[ix],z,True)),('GMM',lambda:_sdf(yy[ix],z))]:
                        fit[name][est]=fun()
                    if name=='FF' and include_gx:
                        gz=tr.to_numpy()
                        gp,gm,gr,gc=gx_three_pass(dd[ix],yy[ix],gz,k)
                        for c,r,m in zip(tr.columns,gr,gm):gxrows.append(dict(formation=p.index[ix[-1]],scheme=scheme,factor=c,projection_r2=r,price_monthly=m,k=k,condition=gc))
                        fit['GX_LATENT']={'GX':(gp,gm,np.full(len(tr.columns),np.nan),gc)}
                    # Direct traded means: never apply a mean-return forecast to nontraded macro shocks.
                    traded=[c for c in cols if c not in extra or extra_traded]
                    zs=tr[traded].to_numpy(); mu=zs.mean(0);V=_hac_moment_cov(zs,6)/len(ix)
                    post,PV=_posterior(mu,V,np.maximum(.25*zs.std(0,ddof=1),1e-8))
                    fit[name]['DIRECT_MEAN']=(None,mu,np.sqrt(np.maximum(np.diag(V),0)),np.nan)
                    fit[name]['DIRECT_SHRINK']=(None,post,np.sqrt(np.maximum(np.diag(PV),0)),np.nan)
                refit_cache[cachekey]=(fit,means,beta_extra,ix)
            fit,means,bx,ix=refit_cache[cachekey]
            for j in hold:
                date=p.index[j];actual=F.iloc[j].copy()
                if bx is not None:actual.loc[list(extra)]-=actual[list(ff)].to_numpy()@bx
                for name in fit:
                    cols=groups[name] if name in groups else list(F.columns)
                    for est,(pred,mu,se,cond) in fit[name].items():
                        direct=est.startswith('DIRECT'); cs=[c for c in cols if c not in extra or extra_traded] if direct else cols
                        for c,m,s in zip(cs,mu,se):prows.append(dict(date=date,formation=p.index[ix[-1]],scheme=scheme,model=name,estimator=est,factor=c,price_monthly=m,se_monthly=s,actual_return=actual[c] if c not in extra or extra_traded else np.nan,condition=cond,n_train=len(ix)))
                        if pred is not None:
                            for a,m,r in zip(y.columns,pred,yy[j]):frows.append(dict(date=date,formation=p.index[ix[-1]],scheme=scheme,model=name,estimator=est,asset=a,forecast=m,actual=r))
                for a,m,r in zip(y.columns,means,yy[j]):frows.append(dict(date=date,formation=p.index[ix[-1]],scheme=scheme,model='HIST_ASSET_MEAN',estimator='MEAN',asset=a,forecast=m,actual=r))
    forecasts=pd.DataFrame(frows);prices=pd.DataFrame(prows);scores=[];boots=[]
    key=['scheme','model','estimator']; piv={}
    for keys,g in forecasts.groupby(key):
        residual=g.assign(error=g.actual-g.forecast).pivot(index='date',columns='asset',values='error').sort_index(); e=residual.to_numpy();piv[keys]=e
        scores.append(dict(zip(key,keys),n_months=len(e),mean_pricing_rmse_ann=12*np.sqrt(np.mean(e.mean(0)**2)),monthly_forecast_rmse=np.sqrt(np.mean(e**2)),mean_error_ann=12*e.mean()))
    # Paired OOS bootstrap of mean pricing RMSE (forecast paths held fixed).
    n=len(piv[next(iter(piv))]); idx=((rng.integers(n,size=(draws,int(np.ceil(n/12))))[:,:,None]+np.arange(12))%n).reshape(draws,-1)[:,:n]
    for keys,e in piv.items():
        scheme,name,est=keys; parent=name.removesuffix('_EXTRA') if name.endswith('_EXTRA') else 'FF'; bk=(scheme,parent,est)
        if bk==keys or bk not in piv:continue
        b=piv[bk]; dif=12*(np.sqrt(np.mean(b[idx].mean(1)**2,axis=1))-np.sqrt(np.mean(e[idx].mean(1)**2,axis=1)));lo,hi=np.quantile(dif,[.025,.975])
        boots.append(dict(zip(key,keys),parent=parent,improvement_ann=12*(np.sqrt(np.mean(b.mean(0)**2))-np.sqrt(np.mean(e.mean(0)**2))),ci_low=lo,ci_high=hi))
    for keys,e in piv.items():
        scheme,name,est=keys
        parents=[]
        if scheme!='fixed':parents.append(('fixed',name,est))
        if est=='FMB_SHRINK':parents.append((scheme,name,'FMB'))
        if est!='MEAN':parents.append((scheme,'HIST_ASSET_MEAN','MEAN'))
        for bk in parents:
            if bk not in piv:continue
            b=piv[bk];dif=12*(np.sqrt(np.mean(b[idx].mean(1)**2,axis=1))-np.sqrt(np.mean(e[idx].mean(1)**2,axis=1)));lo,hi=np.quantile(dif,[.025,.975])
            boots.append(dict(zip(key,keys),parent='/'.join(bk),improvement_ann=12*(np.sqrt(np.mean(b.mean(0)**2))-np.sqrt(np.mean(e.mean(0)**2))),ci_low=lo,ci_high=hi))
    fs=[]
    for keys,g in prices.dropna(subset=['actual_return']).groupby(key+['factor']):
        err=g.actual_return-g.price_monthly; fs.append(dict(zip(key+['factor'],keys),n_months=len(g),forecast_mean_ann=12*g.price_monthly.mean(),realized_mean_ann=12*g.actual_return.mean(),monthly_rmse=np.sqrt(np.mean(err**2)),bias_ann=12*err.mean()))
    return DynamicPremia(forecasts,prices,pd.DataFrame(scores),pd.DataFrame(fs),pd.DataFrame(boots),pd.DataFrame(gxrows),pd.DataFrame(hedgerows),pd.DataFrame({'date':cal,'included':ok.values,'oos':cal>=p.index[first]}),dict(first_oos=str(p.index[first]),last_oos=str(p.index[-1]),window_calendar_months=window,step_observations=step,min_train=min_train,gx_k=k,extra_traded=extra_traded,extra_factors=list(extra),prior='zero mean, SD 0.25 times monthly factor volatility; intercept SD 0.10; plug-in HAC covariance',limitations='Retrospective OOS; overlapping assets, revised histories, no weak-identification robust inference. Bootstrap conditional on forecast paths, compressed complete-case observations. FMB/GMM/GX price estimates are not automatically traded means. Nontraded extra inputs must be known for historical estimation, not forecasts.'))


def economic_return_decomposition(targets,drivers,*,as_of='2026-10-31',window=72,min_train=60,step=3):
    """Decompose projected factor expected return into traded driver means + intercept.
    Raw (not FF-orthogonal) drivers explain original FF/RP; intercept retained.
    Contemporaneous attribution uses frozen coefficients, not a return forecast.
    Separate expected-mean forecast uses only historical driver means.
    """
    targets,drivers=map(_monthly,(targets,drivers)); p=pd.concat([targets,drivers],axis=1).loc[:as_of].dropna()
    if p.columns.has_duplicates:raise ValueError('Duplicate names')
    rows=[]; first=None
    for i,t in enumerate(p.index):
        ix=np.flatnonzero((p.index<t)&(p.index>=t-pd.offsets.MonthEnd(window)))
        if len(ix)<min_train or t-pd.offsets.MonthEnd(window)<p.index[0]:continue
        if first is None:first=i
        if (i-first)%step:continue
        x=p.iloc[ix][list(drivers)].to_numpy();y=p.iloc[ix][list(targets)].to_numpy();coef=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0];mu=x.mean(0)
        for date in p.index[i:i+step]:
            realized=p.loc[date,list(drivers)].to_numpy();act=p.loc[date,list(targets)].to_numpy()
            for j,c in enumerate(targets):
                contrib=coef[1:,j]*mu; row=dict(date=date,formation=p.index[ix[-1]],factor=c,intercept_monthly=coef[0,j],driver_implied_monthly=contrib.sum(),total_expected_monthly=coef[0,j]+contrib.sum(),actual_monthly=act[j],contemporaneous_fit_monthly=coef[0,j]+realized@coef[1:,j])
                row.update({f'contribution_{v}':a for v,a in zip(drivers,contrib)});rows.append(row)
    if not rows:raise ValueError('Insufficient cross-asset mapping history')
    return pd.DataFrame(rows)


def gx_initial_inference(ff,rp,holdout,discovery,*,k_values=(5,8,12),train_end='2011-12-31',as_of='2026-10-31',draws=199):
    """Predeclared latent dimensions on a common initial formation sample; refit all passes."""
    p=pd.concat([ff,rp,holdout,discovery],axis=1).loc[:min(pd.Timestamp(train_end),pd.Timestamp(as_of))].dropna()
    if len(p)<60:raise ValueError('GX requires 60 common formation months')
    names=list(ff)+list(rp);out=[]
    for k in k_values:
        q=gx_bootstrap(p[list(discovery)],p[list(holdout)],p[names],k=k,draws=draws)
        q['factor']=names;q['k']=k;q['n_train']=len(p);q['train_end']=p.index.max();out.append(q)
    return pd.concat(out,ignore_index=True)
