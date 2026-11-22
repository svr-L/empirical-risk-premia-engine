"""Funded Treasury curve factors ported from Rates v0.8.

Preserves: monthly fitted zero curve, lagged EWMA raw-yield vol,
static standardized PCA (training ends 2002), one-sigma yield shocks,
exact gross-normalized funded mimicking portfolios with unrestricted net risky
weight and explicit cash financing. Factor returns use weights known at t-1.

The v0.8 FB/CP timing grid is available through erp.rates_research,
which consumes the fitted engine's formation-date factor weights and risk states.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .core import FactorDefinition, PremiaSnapshot
from .data import load_fedfunds, load_kw_zero_curve
from .equity import _alpha_metrics, fmb_rolling, identification_diagnostics

FACTOR_NAMES = ("Level", "Slope", "Curvature")

@dataclass
class RatesData:
    zero_curve: pd.DataFrame  # 1Y,...,10Y zero-coupon yields in DECIMAL, month-end
    fedfunds: pd.Series     # FEDFUNDS in DECIMAL PER YEAR, month-end

    @classmethod
    def from_fred(cls, cache_dir="data/cache", refresh=False, curve_source="auto",
                  local_curve_path=None, local_fedfunds_path=None, funding_source="mirror"):
        """Back-compatible API; preferred curve source is official Board Kim-Wright CSV.

        Does NOT replace zero-coupon yields with par yields or use funding proxies.
        """
        z = load_kw_zero_curve(cache_dir=cache_dir, refresh=refresh,
                               source=curve_source, local_path=local_curve_path) / 100.
        ff = load_fedfunds(cache_dir=cache_dir, refresh=refresh,
                           local_path=local_fedfunds_path,source=funding_source).resample('ME').last() / 100.
        if z.empty or ff.dropna().empty:
            raise ValueError('Zero curve or exact FEDFUNDS history is empty')
        print(f'Kim-Wright curve ready: {len(z)} months, {z.index.min().date()} to {z.index.max().date()}')
        print(f'FEDFUNDS ready: {int(ff.notna().sum())} months, latest {ff.dropna().index.max().date()}')
        return cls(z, ff)


def interpolate_yield(curve:pd.Series,maturity:float)->float:
    tenors=np.array([float(x.rstrip('Y')) for x in curve.index])
    if maturity < tenors.min()-1e-10 or maturity>tenors.max()+1e-10:
        raise ValueError('Maturity outside zero-curve grid; do not extrapolate')
    return float(np.interp(maturity,tenors,curve.to_numpy(dtype=float)))


def zc_return_decomposition(z:pd.DataFrame, maturities=range(2,11)):
    total=[]; carry=[]
    for i in range(len(z)-1):
        prev=z.iloc[i]; new=z.iloc[i+1]
        rr={}; cr={}
        for tau in maturities:
            rem=tau-1./12.
            p0=np.exp(-tau*prev[f'{tau}Y'])
            cr[f'ZC{tau}Y']=float(np.exp(-rem*interpolate_yield(prev,rem))/p0-1)
            rr[f'ZC{tau}Y']=float(np.exp(-rem*interpolate_yield(new,rem))/p0-1)
        total.append(pd.Series(rr,name=z.index[i+1]))
        carry.append(pd.Series(cr,name=z.index[i]))
    return pd.DataFrame(total),pd.DataFrame(carry)


def lagged_ewma(dy:pd.DataFrame,lam=.97,warmup=36):
    v=dy.to_numpy(dtype=float)
    cov=np.cov(v[:warmup],rowvar=False,ddof=1)
    out={}
    for i in range(warmup,len(dy)):
        dt=dy.index[i];out[dt]=cov.copy()   # time t known before observing shock t
        if np.isfinite(v[i]).all():
            cov=lam*cov+(1-lam)*np.outer(v[i],v[i])
    return out


@dataclass
class StaticCurvePCA:
    loadings:pd.DataFrame
    eigenvalues:pd.Series
    mean:pd.Series
    explained_variance_ratio:pd.Series

    @classmethod
    def fit(cls,z:pd.DataFrame,n=3):
        x=z.to_numpy()-z.mean().to_numpy()
        val,vec=np.linalg.eigh(np.cov(x,rowvar=False,ddof=1))
        inds=np.argsort(val)[::-1];val=val[inds];vec=vec[:,inds][:,:n]
        if vec[:,0].sum()<0:vec[:,0]*=-1
        if n>1 and vec[:,1]@np.linspace(-1,1,len(z.columns))<0: vec[:,1]*=-1
        if n>2:
            shape=1-2*np.linspace(-1,1,len(z.columns))**2
            shape-=shape.mean()
            if vec[:,2]@shape<0:vec[:,2]*=-1
        names=FACTOR_NAMES[:n]
        return cls(pd.DataFrame(vec,index=z.columns,columns=names),pd.Series(val[:n],index=names),
                   z.mean(),pd.Series(val[:n]/val.sum(),index=names))

    def factor_shocks(self,raw_cov:np.ndarray):
        vol=np.sqrt(np.clip(np.diag(raw_cov),0,None))
        shock=(vol[:,None]*self.loadings.to_numpy())*np.sqrt(self.eigenvalues.to_numpy())[None,:]
        return pd.DataFrame(shock,index=self.loadings.index,columns=self.loadings.columns)


def exact_funded_factor_weights(B:pd.DataFrame,H:pd.DataFrame,target_factor:str,gross=1.):
    """Matches v0.8 min w'Hw /2 subject to B_standardized.T w = e_k; cash=-sum(w)."""
    scale=np.sqrt((B**2).mean(axis=0))
    if (scale<=0).any():raise ValueError('zero factor exposure')
    normalized=B.divide(scale,axis=1)
    C=normalized.to_numpy()
    idx=list(normalized.columns).index(target_factor)
    q=np.zeros(C.shape[1]);q[idx]=1.
    HH=H.reindex(index=B.index,columns=B.index).to_numpy()
    invC=np.linalg.solve(HH,C)
    middle=C.T@invC
    w_raw=invC@np.linalg.pinv(middle)@q
    raw_gross=np.abs(w_raw).sum()
    if raw_gross<=1e-14:raise ValueError('degenerate portfolio')
    w=pd.Series(w_raw*gross/raw_gross,index=B.index,name=target_factor)
    exposure=normalized.T@w
    deviation=float(exposure.drop(target_factor).abs().max())
    diagnostics={'purity':float(abs(exposure[target_factor])/exposure.abs().sum()),
                 'max_cross_constraint_violation':deviation,
                 'risky_net_weight':float(w.sum()),'cash_weight':float(-w.sum()),
                 'gross_risky_weight':float(w.abs().sum()),'raw_gross':float(raw_gross)}
    return w,diagnostics


def trailing_risk_cov(asset_excess:pd.DataFrame,dt,assets,lookback=60,minobs=36,diag_shrink=.25,ridge=1e-8):
    hist=asset_excess.loc[:dt,assets].tail(lookback).dropna(how='any')
    if len(hist)<minobs:
        return pd.DataFrame(np.eye(len(assets)),index=assets,columns=assets)
    s=hist.cov().to_numpy();diag=np.diag(np.diag(s))
    H=(1-diag_shrink)*s+diag_shrink*diag
    H+=ridge*max(float(np.diag(H).mean()),1e-12)*np.eye(len(assets))
    return pd.DataFrame(H,index=assets,columns=assets)


class RatesPremiaEngine:
    def __init__(self,data:RatesData):
        self.data=data
        self.latest_weights=None
        self.latest_analytic_exposures=None
        self.latest_mimic_diagnostics=None
        self.factor_carry=None
        self.factor_curve_component=None
        self.pca=None
        self.zero_curve_sample=None
        self.asset_total_returns=None
        self.asset_excess_returns=None
        self.cash_returns=None
        self.factor_returns=None
        self.mimic_weights_panel=None
        self.ewma_covariance_path=None
        self.standardized_yield_changes=None
        self.factor_scores_1sigma=None

    def fit(self,model="PCA3_FUNDED",universe="ZC2Y_10Y",as_of=None,
            pca_train_end="2002-12-31",warmup=36,run_fmb=False,**kwargs)->PremiaSnapshot:
        if model != 'PCA3_FUNDED':raise ValueError('Rates supports PCA3_FUNDED; v0.8 FB/CP research notebook retained separately')
        z=self.data.zero_curve.copy().sort_index()
        z.index=pd.DatetimeIndex(z.index)+pd.offsets.MonthEnd(0)
        if as_of is not None: z=z.loc[:pd.Timestamp(as_of)]
        z=z[[f'{x}Y' for x in range(1,11)]].dropna()
        if len(z)<100:raise ValueError('Rates requires enough history for EWMA/PCA/mimic construction')
        cash_rate=self.data.fedfunds.copy()
        cash_rate.index=pd.DatetimeIndex(cash_rate.index)+pd.offsets.MonthEnd(0)
        last_funding=cash_rate.dropna().index.max()
        # Prevent silently carrying forward a stale funding print into months
        # that were never observed. Preserve only genuine overlap.
        if z.index.max()>last_funding:
            import warnings
            warnings.warn(f'FEDFUNDS ends {last_funding.date()}; truncating the Rates run '
                          f'from {z.index.max().date()} to {last_funding.date()}. '
                          'No future funding rates were imputed.',RuntimeWarning)
            z=z.loc[:last_funding]
        exact_cash=cash_rate.reindex(z.index)
        if exact_cash.isna().any():
            missing=exact_cash.index[exact_cash.isna()].strftime('%Y-%m').tolist()
            raise ValueError(f'FEDFUNDS missing {len(missing)} months inside the curve sample: {missing[:12]}')
        cash_m=(exact_cash.shift(1)/12.).rename('cash')
        total,carry=zc_return_decomposition(z)
        excess=total.sub(cash_m.reindex(total.index),axis=0)
        dy=z.diff().dropna()
        cov=lagged_ewma(dy,warmup=warmup)
        std=[]
        for dt,S in cov.items():
            vol=np.sqrt(np.clip(np.diag(S),1e-12,None))
            std.append(pd.Series(dy.loc[dt].to_numpy()/vol,index=dy.columns,name=dt))
        zz=pd.DataFrame(std)
        train=zz.loc[:pca_train_end].dropna()
        if len(train)<36:raise ValueError('PCA training insufficient; select a later pca_train_end')
        pca=StaticCurvePCA.fit(train)
        self.pca=pca
        self.standardized_yield_changes=zz
        self.ewma_covariance_path=cov
        self.factor_scores_1sigma=(zz.subtract(pca.mean,axis=1).to_numpy() @ pca.loadings.to_numpy()) / np.sqrt(pca.eigenvalues.to_numpy())
        self.factor_scores_1sigma=pd.DataFrame(self.factor_scores_1sigma,index=zz.index,columns=list(FACTOR_NAMES))
        names=list(FACTOR_NAMES)
        # mimic weights are only used after the initial PCA calibration window, avoiding look-ahead.
        factor_returns={x:{} for x in names};factor_carry={x:{} for x in names}
        weight_rows=[];diag_rows=[];analytic_latest=None
        for i in range(1,len(z)):
            dt0=z.index[i-1];dt1=z.index[i]
            if dt0 not in cov or dt0<pd.Timestamp(pca_train_end):continue
            shocks=pca.factor_shocks(cov[dt0])
            B=pd.DataFrame({f:pd.Series({f'ZC{m}Y': -m*shocks.loc[f'{m}Y',f] for m in range(2,11)}) for f in names})
            assets=B.index
            H=trailing_risk_cov(excess,dt0,assets)
            current_weights={}
            current_diagnostics={}
            for f in names:
                w,d=exact_funded_factor_weights(B,H,f)
                current_weights[f]=w
                current_diagnostics[f]=d
                if dt1 in excess.index and excess.loc[dt1,assets].notna().all():
                    r=float(w@excess.loc[dt1,assets])
                    prev_funding=float(exact_cash.loc[dt0])/12.
                    cr=float(w@(carry.loc[dt0,assets]-prev_funding))
                    factor_returns[f][dt1]=r
                    factor_carry[f][dt1]=cr
                diag_rows.append({'Date':dt0,'factor':f,**d})
                weight_rows.append({'Date':dt0,'factor':f,**w.to_dict()})
            analytic_latest=B
        result_f=pd.DataFrame({f:pd.Series(v) for f,v in factor_returns.items()}).sort_index().dropna()
        fc=pd.DataFrame({f:pd.Series(v) for f,v in factor_carry.items()}).sort_index().reindex(result_f.index)
        self.factor_carry=fc
        self.factor_returns=result_f
        self.zero_curve_sample=z
        self.asset_total_returns=total
        self.asset_excess_returns=excess
        self.cash_returns=cash_m
        self.factor_curve_component=result_f-fc
        weights=pd.DataFrame(weight_rows).set_index(['Date','factor']).sort_index()
        self.mimic_weights_panel={f:weights.xs(f,level='factor') for f in names}
        mimic_diag=pd.DataFrame(diag_rows).set_index(['Date','factor']).sort_index()
        self.latest_weights=weights.loc[weights.index.get_level_values(0).max()]
        self.latest_mimic_diagnostics=mimic_diag.loc[mimic_diag.index.get_level_values(0).max()]
        self.latest_analytic_exposures=analytic_latest
        # Current-date weights are observable at the latest curve snapshot,
        # even though their future t->t+1 realized return cannot be observed.
        current_date=z.index[-1]
        if current_date in cov and current_date>=pd.Timestamp(pca_train_end):
            shocks=pca.factor_shocks(cov[current_date])
            Bb=pd.DataFrame({f:pd.Series({f'ZC{m}Y':-m*shocks.loc[f'{m}Y',f] for m in range(2,11)}) for f in names})
            HH=trailing_risk_cov(excess,current_date,Bb.index)
            ww={};dd={}
            for fac in names:
                ww[fac], dd[fac]=exact_funded_factor_weights(Bb,HH,fac)
            self.latest_weights=pd.DataFrame(ww).T
            self.latest_mimic_diagnostics=pd.DataFrame(dd).T
            self.latest_analytic_exposures=Bb
        if result_f.empty:raise ValueError('No funded factor returns created')
        # All factor prices are realized returns on self-financing funded mimicking portfolios.
        joined=excess.join(result_f,how='inner').dropna()
        r=joined[excess.columns];f=joined[names]
        s,b,a,e,rr,ff=_alpha_metrics(r,f,6)
        meta={'data_source':'Federal Reserve Board Kim-Wright fitted zeros and FEDFUNDS (documented source)' , 'last_funding_month': str(last_funding.date()),
              'pca_fit_end':pca_train_end,'pca_variance':pca.explained_variance_ratio.to_dict(),
              'n_months':len(f),'n_assets':len(r.columns),
              'funding':'cash funding = minus risky weight sum',
              'mimic_max_cross_violation':float(mimic_diag['max_cross_constraint_violation'].max()),
              'factor_mean_interpretation':'trailing unconditional funded factor mean, not FB/CP conditional forecast',
              'last_mimic_formation_date':str(current_date.date()),
              'data_vintage':'FRED currently revised values; not historical release vintages'}
        snap=PremiaSnapshot(asset_class='rates',model=model,universe=universe,as_of=f.index.max(),
                 factors=f,asset_excess_returns=r,betas=b[names],alpha=a,residuals=e,
                 factor_premia=f.mean(),fit_diagnostics=s,metadata=meta,
                 factor_definitions=[FactorDefinition(name=x,family='static standardized PCA',
                 source='Federal Reserve fitted zeros via FRED',construction='funded, exactly cross-neutral 2Y-10Y portfolio; time-varying weights',
                 convention='positive Level shock corresponds to yields rising') for x in names])
        return snap.validate()

    def export_auxiliary(self,dest):
        from pathlib import Path
        p=Path(dest);p.mkdir(parents=True,exist_ok=True)
        if self.latest_weights is not None:self.latest_weights.to_csv(p/'rates_latest_funded_weights.csv')
        if self.latest_analytic_exposures is not None:self.latest_analytic_exposures.to_csv(p/'rates_latest_analytic_shock_betas.csv')
        if self.latest_mimic_diagnostics is not None:self.latest_mimic_diagnostics.to_csv(p/'rates_latest_mimic_constraints.csv')
        if self.factor_carry is not None:self.factor_carry.to_csv(p/'rates_funded_carry.csv')
        if self.factor_curve_component is not None:self.factor_curve_component.to_csv(p/'rates_realized_curve_component.csv')
        return p


def run_rates_research(data:RatesData,**kwargs):
    return RatesPremiaEngine(data).fit(**kwargs)
