"""Equity implementation: CAPM/FF3/Carhart4/FF5/FF6 and published AQR BAB/QMJ.

Reproduces original v0.1 FF-family SMB selection and its lagged-beta FMB design.
All estimates are in monthly decimal excess returns; expanded test universe is
purposefully overlapping and is NOT treated as independent observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from .core import FactorDefinition, PremiaSnapshot
from .data import load_french_monthly, load_aqr_monthly

@dataclass(frozen=True)
class EquityModel:
    name: str
    family: str
    factors: tuple[str, ...]

MODELS = OrderedDict([
    ("CAPM", EquityModel("CAPM", "FF3", ("MKT",))),
    ("FF3", EquityModel("FF3", "FF3", ("MKT", "SMB", "HML"))),
    ("CARHART4", EquityModel("CARHART4", "FF3", ("MKT", "SMB", "HML", "MOM"))),
    ("FF5", EquityModel("FF5", "FF5", ("MKT", "SMB", "HML", "RMW", "CMA"))),
    ("FF6", EquityModel("FF6", "FF5", ("MKT", "SMB", "HML", "RMW", "CMA", "MOM"))),
    ("FF6_BAB", EquityModel("FF6_BAB", "FF5", ("MKT", "SMB", "HML", "RMW", "CMA", "MOM", "BAB"))),
    ("FF6_QMJ", EquityModel("FF6_QMJ", "FF5", ("MKT", "SMB", "HML", "RMW", "CMA", "MOM", "QMJ"))),
    ("FF6_BAB_QMJ", EquityModel("FF6_BAB_QMJ", "FF5", ("MKT", "SMB", "HML", "RMW", "CMA", "MOM", "BAB", "QMJ"))),
    ("FF3_BAB_QMJ", EquityModel("FF3_BAB_QMJ", "FF3", ("MKT", "SMB", "HML", "BAB", "QMJ"))),
])

@dataclass
class EquityData:
    ff3: pd.DataFrame  # MKT, SMB, HML, RF
    ff5: pd.DataFrame  # MKT, SMB, HML, RMW, CMA, RF
    mom: pd.Series     # MOM
    universes: dict[str, pd.DataFrame]  # raw total simple monthly returns, decimals
    aqr: pd.DataFrame | None = None      # BAB/QMJ long-short monthly decimal returns

    @classmethod
    def from_public_sources(cls, cache_dir="data/cache", refresh=False, aqr=True,
                            universe_keys=("SIZE_BM_25", "INDUSTRY_30", "SIZE_OP_25", "SIZE_INV_25", "SIZE_MOM_25"),
                            aqr_paths: dict | None = None) -> "EquityData":
        ff3=load_french_monthly("FF3",cache_dir,refresh)
        ff5=load_french_monthly("FF5",cache_dir,refresh)
        mm=load_french_monthly("MOM",cache_dir,refresh)
        momc=next((c for c in mm if c.strip().upper() in ("MOM","UMD")), None)
        if momc is None:
            raise ValueError(f"French MOM column not found: {list(mm)}")
        assets={key:load_french_monthly(key,cache_dir,refresh) for key in universe_keys}
        aqr_df=None
        if aqr:
            paths=aqr_paths or {}
            try:
                aqr_df=pd.concat([load_aqr_monthly(f,cache_dir,refresh,local_path=paths.get(f)) for f in ("BAB","QMJ")],axis=1)
            except Exception as e:
                # no synthetic substitute; standard models still usable but caller must see failure
                raise RuntimeError("Published AQR BAB/QMJ could not be loaded. Set aqr=False for French-only models or supply aqr_paths with original AQR workbooks. "+str(e)) from e
        return cls(ff3=ff3,ff5=ff5,mom=mm[momc].rename("MOM"),universes=assets,aqr=aqr_df)

    def factors(self, model:str) -> tuple[pd.DataFrame,pd.Series]:
        spec=MODELS[model]
        family=self.ff3 if spec.family=="FF3" else self.ff5
        frame=family.copy()
        if "MOM" in spec.factors:
            frame=frame.join(self.mom,how="inner")
        extras=[f for f in spec.factors if f in ("BAB","QMJ")]
        if extras:
            if self.aqr is None or any(c not in self.aqr for c in extras):
                raise ValueError(f"Model {model} needs real AQR factors {extras}; data unavailable")
            frame=frame.join(self.aqr[extras],how="inner")
        cols=list(spec.factors)
        if any(f not in frame for f in cols) or "RF" not in frame:
            raise ValueError(f"Missing factors for {model}: {cols}")
        return frame[cols].dropna(),frame["RF"].dropna()

    def excess_returns(self, universe:str, rf:pd.Series) -> pd.DataFrame:
        if universe.startswith("POOLED"):
            names=[x for x in self.universes if x != "POOLED"]
            # Prefix avoids identical column identifiers, but economic constituents still overlap.
            assets=pd.concat([self.universes[k].add_prefix(k+"__") for k in names],axis=1)
        else:
            assets=self.universes[universe]
        aligned=assets.join(rf.rename("RF"),how="inner")
        return aligned[assets.columns].subtract(aligned["RF"],axis=0)


def _monthly(df):
    out=df.copy()
    out.index=pd.DatetimeIndex(out.index)+pd.offsets.MonthEnd(0)
    return out.sort_index()


def _ols_matrix(X:np.ndarray,Y:np.ndarray):
    coefs=np.linalg.lstsq(X,Y,rcond=None)[0]
    return coefs, Y-X@coefs


def _alpha_metrics(y:pd.DataFrame,f:pd.DataFrame,hac_lags:int=6):
    common=y.join(f,how="inner").dropna()
    yy=common[y.columns]; ff=common[f.columns]
    X=sm.add_constant(ff,has_constant="add")
    rows=[]; betas=[]; alpha=[]; residuals=pd.DataFrame(index=common.index,columns=y.columns,dtype=float)
    for a in yy.columns:
        fit=sm.OLS(yy[a],X).fit(cov_type="HAC",cov_kwds={"maxlags":min(hac_lags,len(X)-2)})
        rows.append({"asset":a,"alpha_monthly":float(fit.params["const"]),"alpha_ann":12*float(fit.params["const"]),
                     "alpha_t_hac":float(fit.tvalues["const"]),"adj_r2":float(fit.rsquared_adj),
                     "resid_ann_vol":float(fit.resid.std(ddof=1)*np.sqrt(12)),"nobs":int(fit.nobs)})
        betas.append(pd.Series(fit.params[f.columns],name=a))
        alpha.append((a,float(fit.params["const"])))
        residuals.loc[fit.resid.index,a]=fit.resid
    stats_df=pd.DataFrame(rows).set_index("asset")
    return (stats_df,pd.DataFrame(betas),pd.Series(dict(alpha),name="alpha_monthly"),residuals,yy,ff)


def static_cross_sectional_pricing(returns:pd.DataFrame, betas:pd.DataFrame):
    """Cross-sectional mean excess return pricing; independent of the FMB estimator.
    With free intercept, poorly dispersed market beta can yield a large lambda_0.
    """
    yy=returns.mean().reindex(betas.index).astype(float)
    X=sm.add_constant(betas,has_constant="add")
    fit=sm.OLS(yy,X).fit()
    lam=fit.params.rename(index={"const":"intercept"})
    errors=fit.resid.rename("pricing_error_monthly")
    metrics={"cs_r2":float(fit.rsquared),
             "pricing_rmse_ann":float(12*np.sqrt(np.mean(fit.resid**2))),
             "pricing_mean_abs_ann":float(12*np.abs(fit.resid).mean()),
             "cs_condition_raw":float(np.linalg.cond(X.to_numpy()))}
    return lam,errors,metrics


def grs_test(y:pd.DataFrame, factors:pd.DataFrame, alpha:pd.Series,residuals:pd.DataFrame):
    """Standard GRS F test; requires T > N+K and invertible residual covariance."""
    T,N=y.shape; K=factors.shape[1]
    if T<=N+K or N==0:
        return {"grs_F":np.nan,"grs_p":np.nan,"grs_note":"insufficient T relative to N+K"}
    sig=residuals.cov().to_numpy(); omega=factors.cov().to_numpy()
    if np.linalg.matrix_rank(sig)<N or np.linalg.matrix_rank(omega)<K:
        return {"grs_F":np.nan,"grs_p":np.nan,"grs_note":"singular covariance"}
    a=alpha.reindex(y.columns).to_numpy()
    m=factors.mean().to_numpy()
    numerator=(T-N-K)/N*(a@np.linalg.solve(sig,a))
    denominator=1.+m@np.linalg.solve(omega,m)
    f=float(numerator/denominator)
    return {"grs_F":f,"grs_p":float(stats.f.sf(f,N,T-N-K)),"grs_note":"standard GRS F; no covariance shrinkage"}


def identification_diagnostics(beta:pd.DataFrame, with_intercept:bool=True) -> pd.DataFrame:
    """Factor-specific dispersion, *partial* dispersion and auxiliary R²; scale-aware condition number.
    Low partial dispersion means factor has little independent leverage to identify lambda.
    """
    b=beta.astype(float).dropna()
    centered=b-b.mean()
    raw_mat=np.column_stack([np.ones(len(b)),b.to_numpy()]) if with_intercept else b.to_numpy()
    rank=int(np.linalg.matrix_rank(raw_mat))
    standard=np.column_stack([np.ones(len(b)), centered.to_numpy()/centered.std(ddof=1).clip(lower=1e-12).to_numpy()]) if with_intercept else centered.to_numpy()/centered.std(ddof=1).clip(lower=1e-12).to_numpy()
    condition=float(np.linalg.cond(standard))
    rows=[]
    for f in b:
        y=b[f].to_numpy()
        others=b.drop(columns=f)
        X=np.column_stack([np.ones(len(b)),others.to_numpy()])
        err=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
        v=float(np.var(y,ddof=1))
        pv=float(np.var(err,ddof=1))
        rows.append({"factor":f,"n_assets":len(b),"beta_std":float(np.std(y,ddof=1)),
                     "beta_q05":float(np.quantile(y,.05)),"beta_q95":float(np.quantile(y,.95)),
                     "partial_beta_std":float(np.sqrt(max(0,pv))),
                     "auxiliary_r2":float(1-pv/v) if v>1e-20 else np.nan,
                     "cs_rank":rank,"cs_n_columns":raw_mat.shape[1],"cs_condition_scaled":condition})
    return pd.DataFrame(rows).set_index("factor")


def fmb_rolling(y:pd.DataFrame, factors:pd.DataFrame, window:int=60, min_assets:int=15, hac_lags:int=6):
    """At each month t, beta uses only t-window..t-1; CS monthly excess returns at t.
    Uses joint least squares for balanced panels, assetwise fallback on unbalanced panels.
    """
    common=y.join(factors,how="inner")
    assets=list(y.columns); names=list(factors.columns)
    if len(common)<=window+6:
        raise ValueError("FMB needs more dates than beta window + 6")
    cross=[]; beta_snapshots=[]; oos=[]
    for i in range(window,len(common)):
        dt=common.index[i]
        history=common.iloc[i-window:i]
        xx=history[names]
        if xx.isna().any().any():
            continue
        X=np.column_stack([np.ones(len(history)),xx.to_numpy()])
        Y=history[assets]
        if np.linalg.matrix_rank(X)<len(names)+1:
            continue
        if Y.notna().all().all():
            b,_=_ols_matrix(X,Y.to_numpy())
            beta=pd.DataFrame(b[1:,:].T,index=assets,columns=names)
        else:
            row=[]
            for asset in assets:
                valid=Y[asset].notna()
                if valid.sum()<max(len(names)+8,window//2):
                    row.append([np.nan]*len(names))
                else:
                    row.append(np.linalg.lstsq(X[valid],Y.loc[valid,asset],rcond=None)[0][1:])
            beta=pd.DataFrame(row,index=assets,columns=names)
        current=common.iloc[i]
        yi=current[assets].astype(float)
        ok=yi.notna() & beta.notna().all(axis=1)
        if ok.sum()<max(min_assets,len(names)+4):
            continue
        bb=beta.loc[ok]; yy=yi.loc[ok]
        x_cs=np.column_stack([np.ones(len(bb)),bb.to_numpy()])
        if np.linalg.matrix_rank(x_cs)<len(names)+1:
            continue
        coefs=np.linalg.lstsq(x_cs,yy.to_numpy(),rcond=None)[0]
        cross.append(pd.Series(coefs,index=["intercept",*names],name=dt))
        bf=bb.to_numpy()@current[names].to_numpy(dtype=float)
        oos.append(pd.DataFrame({"Date":dt,"asset":bb.index,
                                 "factor_component":bf,"realized_excess_return":yy.to_numpy(),
                                 "unexplained_including_alpha":yy.to_numpy()-bf}).set_index(["Date","asset"]))
        beta_snapshots.append((dt,beta))
    if not cross:
        raise ValueError("No rank-identified Fama-MacBeth cross-section; inspect factor / test-asset design")
    lambdas=pd.DataFrame(cross)
    rows=[]
    for factor in lambdas:
        a=lambdas[factor].dropna()
        fit=sm.OLS(a.to_numpy(),np.ones((len(a),1))).fit(cov_type='HAC',cov_kwds={'maxlags':min(hac_lags,len(a)-1)})
        rows.append({'factor':factor,'lambda_monthly':float(fit.params[0]),'lambda_ann_arithmetic':float(12*fit.params[0]),
                     't_hac':float(fit.tvalues[0]),'n_months':len(a)})
    summary=pd.DataFrame(rows).set_index('factor')
    return lambdas,summary, pd.concat(oos).sort_index(), beta_snapshots


class EquityPremiaEngine:
    def __init__(self,data:EquityData): self.data=data

    def fit(self,model:str="FF6",universe:str="SIZE_BM_25",as_of=None,
            beta_window:int=60,hac_lags:int=6,run_fmb:bool=True)->PremiaSnapshot:
        if model not in MODELS:
            raise ValueError(f"Model {model} not in {list(MODELS)}")
        f,rf=self.data.factors(model)
        r=self.data.excess_returns(universe,rf)
        cutoff=pd.Timestamp(as_of) if as_of is not None else min(f.index.max(),r.index.max())
        cutoff=cutoff  # Strict date cutoff: do not include month-end after a mid-month as_of
        f=f.loc[:cutoff]; r=r.loc[:cutoff]
        both=r.join(f,how="inner").dropna()
        if len(both)<beta_window+12:
            raise ValueError(f"Insufficient common data for {model}/{universe}: {len(both)} months")
        r=both[r.columns]; f=both[f.columns]
        s,b,a,e,rr,ff=_alpha_metrics(r,f,hac_lags)
        g=grs_test(rr,ff,a,e)
        cs_lambdas,cs_errors,cs_metrics=static_cross_sectional_pricing(rr,b)
        fmb_l=pd.DataFrame(); fmb_s=pd.DataFrame(); oos=pd.DataFrame()
        identification=identification_diagnostics(b)
        metadata={"sample_start":str(r.index.min().date()),"sample_end":str(r.index.max().date()),
                  "n_months":len(r),"n_assets":len(r.columns),"n_factors":len(f.columns),
                  "smb_family":MODELS[model].family,"beta_window":beta_window,"hac_lags":hac_lags,
                  "overlapping_constituents":universe.startswith("POOLED"),
                  "grs":g,"static_cross_sectional_pricing":cs_metrics,
                  "factor_premia_estimator":"historical_traded_factor_means",
                  "data_vintage":"downloaded series; not true historical publication vintage"}
        if run_fmb:
            fmb_l,fmb_s,oos,history=fmb_rolling(r,f,beta_window,hac_lags=hac_lags)
            # For FMB diagnostics use ex ante rolling betas from last usable cross section.
            last_beta=history[-1][1].dropna()
            identification=identification_diagnostics(last_beta)
            metadata['last_fmb_beta_date']=str(history[-1][0].date())
            metadata['fmb_identification_note']='FMB inference is HAC on lambda_t; Shanken/error-in-beta correction not implemented.'
        s.attrs.clear()
        definitions=[]
        for fac in f:
            provider="AQR" if fac in ("BAB","QMJ") else "Kenneth French"
            family=MODELS[model].family if provider=="Kenneth French" else "AQR"
            definitions.append(FactorDefinition(name=fac,family=family,source=provider,
                construction=("FF3-family SMB" if fac=="SMB" and family=="FF3" else
                              "FF5-family SMB" if fac=="SMB" and family=="FF5" else
                              "published long-short factor" if provider=="AQR" else "published factor portfolio"),
                traded=True,convention="MKT is market excess return; others are self-financing factor returns"))
        result=PremiaSnapshot(asset_class="equity",model=model,universe=universe,as_of=r.index.max(),
                factors=f,asset_excess_returns=r,betas=b[f.columns],alpha=a,residuals=e,
                factor_premia=f.mean().rename('mean_monthly'),fit_diagnostics=s,metadata=metadata,
                fmb_lambdas=fmb_l,fmb_diagnostics=fmb_s.join(identification,how="outer"),
                oos_attribution=oos,factor_definitions=definitions,
                static_cs_premia=cs_lambdas,static_pricing_errors=cs_errors)
        return result.validate()

    def spanning_test(self,*,target:str,against_model:str="FF6",as_of=None,hac_lags:int=6):
        """HAC alpha of a published target factor against a benchmark factor span.
        The same factor name is not allowed as both target and explanatory factor.
        """
        if target in MODELS[against_model].factors:
            raise ValueError('Target factor already belongs to benchmark model')
        f,_=self.data.factors(against_model)
        if target in ("BAB","QMJ"):
            if self.data.aqr is None or target not in self.data.aqr:
                raise ValueError(f'Published AQR {target} is unavailable')
            y=self.data.aqr[target]
        elif target=="MOM":
            y=self.data.mom
        else:
            src=self.data.ff3 if MODELS[against_model].family=="FF3" else self.data.ff5
            if target not in src:raise ValueError(f'Unknown factor {target}')
            y=src[target]
        d=f.join(y.rename('target'),how='inner').dropna()
        if as_of is not None:d=d.loc[:pd.Timestamp(as_of)]
        if len(d)<30:raise ValueError('Insufficient factor spanning history')
        fit=sm.OLS(d['target'],sm.add_constant(d[f.columns],has_constant='add')).fit(
            cov_type='HAC',cov_kwds={'maxlags':min(hac_lags,len(d)-2)})
        return {'target':target,'against_model':against_model,'n_months':len(d),
                'alpha_monthly':float(fit.params['const']),'alpha_ann':float(fit.params['const']*12),
                'alpha_t_hac':float(fit.tvalues['const']),'r2':float(fit.rsquared)}

    def compare_models(self,models:list[str],universe:str="SIZE_BM_25",as_of=None,
                       beta_window:int=60,run_fmb:bool=False)->pd.DataFrame:
        # common dates across models are explicitly enforced, preventing BAB/QMJ truncation bias.
        series={name:self.data.factors(name)[0] for name in models}
        common=pd.DatetimeIndex(sorted(set.intersection(*(set(s.index) for s in series.values()))))
        if as_of is not None: common=common[common<=pd.Timestamp(as_of)]
        if len(common)<beta_window+12: raise ValueError('Too few common dates')
        reports=[]; oos_by_model={}
        for model in models:
            snap=self.fit(model,universe,as_of=common.max(),beta_window=beta_window,run_fmb=run_fmb)
            # TS alpha all fit on precisely the same common months.
            intersect=common.intersection(snap.asset_excess_returns.index)
            if len(intersect)!=len(common):
                raise ValueError('Model-specific missing dates prevent strict common sample')
            _,_,a,res,r,f=_alpha_metrics(snap.asset_excess_returns.loc[common],snap.factors.loc[common])
            _,_,oos,_=fmb_rolling(r,f,window=beta_window)
            oos_by_model[model]=oos
            reports.append({'model':model,'months':len(common),'n_assets':r.shape[1],'n_factors':f.shape[1],
                'alpha_rmse_ann':float(12*np.sqrt(np.mean(a**2))),
                'alpha_abs_mean_ann':float(12*np.abs(a).mean()),
                'factor_explained_r2':float(1-res.pow(2).sum().sum()/((r-r.mean()).pow(2).sum().sum())),
                'sample_start':str(common.min().date()),'sample_end':str(common.max().date())})
        # Same month+asset pairs across *all* models: honest common OOS comparison.
        common_observations=set.intersection(*(set(frame.index) for frame in oos_by_model.values()))
        if not common_observations:raise ValueError('No common cross-model OOS observations')
        shared=pd.MultiIndex.from_tuples(sorted(common_observations),names=['Date','asset'])
        for row in reports:
            o=oos_by_model[row['model']].loc[shared]
            target=o['realized_excess_return']
            predicted=o['factor_component']
            diff=target-predicted
            denom=target.groupby(level='asset').transform('mean')
            sst=float(((target-denom)**2).sum())
            row['oos_factor_rmse_monthly']=float(np.sqrt(np.mean(diff**2)))
            row['oos_factor_explained_r2']=float(1-(diff**2).sum()/sst) if sst>1e-18 else np.nan
            row['oos_common_observations']=len(shared)
            row['oos_start']=str(shared.get_level_values('Date').min().date())
            row['oos_end']=str(shared.get_level_values('Date').max().date())
        return pd.DataFrame(reports).set_index('model')


def run_equity_research(data:EquityData,model="FF6",universe="POOLED_130",as_of=None,**kwargs):
    return EquityPremiaEngine(data).fit(model,universe,as_of,**kwargs)
