"""Rates v0.9: common-sample OOS research & direct vs decomposed targets.

No alteration to v0.8 forecasting, portfolio construction, or macro transforms.
All metrics use matching dates across selected models, benchmark and factor.
Historical forecast-origin targets require H months of realized subsequent returns;
this is a retrospective evaluation, not an operational prediction for as_of.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

from .rates_research import (
    FACTOR_NAMES,MODEL_FAMILIES,FB_FORWARD_MATURITIES,CP_FORWARD_MATURITIES,
    build_forward_rate_panel,build_horizon_unconditional_forecast,
    walk_forward_empirical_premia_model,performance_metrics,
    RatesEmpiricalBondPremiaEngine,
)


def hac_mean_test(values, lags:int) -> dict:
    """Two-sided HAC mean test, robust to overlapping H-period labels."""
    z=pd.Series(values,dtype=float).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<3:
        return {'n':len(z),'mean':np.nan,'t_hac':np.nan,'p_hac':np.nan,'hac_se':np.nan}
    fit=sm.OLS(z.to_numpy(),np.ones((len(z),1))).fit(
        cov_type='HAC',cov_kwds={'maxlags':min(max(0,int(lags)),len(z)-1)})
    return {'n':len(z),'mean':float(fit.params[0]),'t_hac':float(fit.tvalues[0]),
            'p_hac':float(fit.pvalues[0]),'hac_se':float(fit.bse[0])}


def circular_block_bootstrap_mean(values,block_length:int,replicates:int=399,seed:int=202609):
    """Paired circular moving-block bootstrap on a time-ordered statistic.

    Each observation is one date (factor means are aggregated first), so that
    correlated factor losses are never resampled independently. These are
    percentile CIs for the mean, not a claim of exact small-sample coverage.
    """
    x=np.asarray(pd.Series(values).dropna(),dtype=float)
    n=len(x)
    if n<12 or replicates<=0:
        return {'ci_low':np.nan,'ci_high':np.nan,'bootstrap_p_two_sided':np.nan,'block_length':np.nan}
    L=min(max(1,int(block_length)),n)
    b=int(np.ceil(n/L));rng=np.random.default_rng(seed)
    means=np.empty(replicates,dtype=float)
    offsets=np.arange(L)
    for k in range(replicates):
        starts=rng.integers(0,n,size=b)
        inds=((starts[:,None]+offsets[None,:])%n).ravel()[:n]
        means[k]=x[inds].mean()
    ci=np.percentile(means,[2.5,97.5])
    # Centered bootstrap null distribution; conservative two-sided approximate p.
    observed=abs(x.mean());centered=means-means.mean()
    p=float((1+np.count_nonzero(abs(centered)>=observed))/(replicates+1))
    return {'ci_low':float(ci[0]),'ci_high':float(ci[1]),
            'bootstrap_p_two_sided':p,'block_length':L}


def _p(m):
    """Return forecast table with dates x factors; one table for each method."""
    if m is None or m.empty:return pd.DataFrame()
    if 'total_mean_h' not in m:raise ValueError('Forecast missing total_mean_h')
    return m['total_mean_h'].unstack('factor').reindex(columns=FACTOR_NAMES)


def _predictor_panel(rates):
    z=rates.zero_curve_sample
    forwards=build_forward_rate_panel(z)
    fb=pd.DataFrame({f'FB{n}':forwards[f'f{n}']-forwards['f1'] for n in FB_FORWARD_MATURITIES},index=z.index)
    cp=forwards[[f'f{n}' for n in CP_FORWARD_MATURITIES]].copy()
    return fb,cp


def build_direct_total_forecasts(rates,research,macro_raw,models=('FB','CP'),min_train=60,ridge=1.0):
    """Fit identical v0.8 learner to H-month *total excess* factor returns.

    Reuses exact v0.8 walk-forward function with carry=0, so it estimates
    E_t[R^excess_(t,t+H)] directly rather than E_t[curve] + known carry_t,H.
    Does NOT use future realized cash as an input; only labels realized by
    forecast date - H are admitted by the legacy trainer.
    """
    from .rates_research import build_v08_state_panel
    models=tuple(models)
    if not set(models).issubset(MODEL_FAMILIES):raise ValueError('Unknown direct model family')
    if any('STATE' in m for m in models) and macro_raw is None:
        raise ValueError('Direct STATE models need original macro_raw')
    fb,cp=_predictor_panel(rates)
    state=build_v08_state_panel(rates,macro_raw) if any('STATE' in m for m in models) else None
    out={};coef={}
    for h,targ in research.factor_targets.items():
        if targ['total'].empty:continue
        zeros=pd.DataFrame(0.,index=targ['total'].index,columns=FACTOR_NAMES)
        # CP first stage remains the same as in v0.8. Its observed target is
        # average 2Y-5Y realized H-month log excess returns, computed before.
        from .rates_research import build_horizon_zero_coupon_data
        z=rates.zero_curve_sample
        cash=rates.cash_returns.reindex(z.index)
        funding=rates.data.fedfunds.copy();funding.index=pd.DatetimeIndex(funding.index)+pd.offsets.MonthEnd(0)
        fund_month=funding.reindex(z.index)/12.
        assets=build_horizon_zero_coupon_data(z,list(range(2,11)),cash,fund_month,h)
        avg=assets['log_excess'][['ZC2Y','ZC3Y','ZC4Y','ZC5Y']].mean(axis=1)
        for model in models:
            forecast,c=walk_forward_empirical_premia_model(
                horizon_months=h,model_family=model,curve_target_h=targ['total'],
                carry_h=zeros,avg_asset_excess_h=avg,state_panel=state,
                fb_panel=fb,cp_forward_panel=cp,master_index=z.index,
                min_train=min_train,ridge=ridge)
            if forecast is not None and not forecast.empty:
                # Column naming kept for legacy generic learner but *here*
                # curve_mean_h is a direct forecast of TOTAL excess return.
                out[(h,'DIRECT_'+model)]=forecast
            if c is not None:coef[(h,'DIRECT_'+model)]=c
    return out,coef


def cp_first_stage_diagnostics(rates,research,step_months=12,ridges=(1e-10,.01,.1,1.),min_train=60):
    """Prospective annual checkpoint sensitivity of CP first-stage ridge.

    This is a *diagnostic*, not a selection of ridge based on full-sample
    forecasting outcomes. Forward features are standardized using only the
    training sample and labels originate at most at t-H.
    Coefficients and ridge penalties are in standardized feature units;
    therefore do not compare numeric magnitudes to v0.8 raw CP ridge.
    """
    from .rates_research import (build_horizon_zero_coupon_data,realization_cutoff_date)
    z=rates.zero_curve_sample
    fwd=build_forward_rate_panel(z)[[f'f{n}' for n in CP_FORWARD_MATURITIES]]
    cash=rates.cash_returns.reindex(z.index)
    funding=rates.data.fedfunds.copy();funding.index=pd.DatetimeIndex(funding.index)+pd.offsets.MonthEnd(0)
    monthly_funding=funding.reindex(z.index)/12.
    rows=[]
    for h in sorted(research.factor_targets):
        available=research.forecasts.get((h,'CP'))
        if available is None or available.empty:continue
        dates=available.index.get_level_values('forecast_date').unique().sort_values()[::max(1,int(step_months))]
        asset=build_horizon_zero_coupon_data(z,[2,3,4,5],cash,monthly_funding,h)
        target=asset['log_excess'][['ZC2Y','ZC3Y','ZC4Y','ZC5Y']].mean(axis=1)
        for t in dates:
            cutoff=realization_cutoff_date(t,h,z.index)
            if cutoff is None:continue
            tframe=pd.concat([target.rename('y'),fwd],axis=1).loc[:cutoff].dropna()
            if len(tframe)<min_train:continue
            X=tframe[fwd.columns].to_numpy(float)
            mu=X.mean(axis=0)
            sigma=np.maximum(X.std(axis=0,ddof=1),1e-12)
            X=(X-mu)/sigma
            y=tframe['y'].to_numpy(float)
            xcurrent=(fwd.loc[t].to_numpy(float)-mu)/sigma
            if not np.isfinite(xcurrent).all():continue
            Xz=np.column_stack([np.ones(len(X)),X]);x0=np.r_[1.,xcurrent]
            cond=np.linalg.cond(X) if np.linalg.matrix_rank(X)==X.shape[1] else np.inf
            for alpha in ridges:
                penalty=np.eye(Xz.shape[1])*alpha; penalty[0,0]=0
                b=np.linalg.pinv(Xz.T@Xz+penalty)@Xz.T@y
                yreal=float(target.loc[t]) if t in target.index and pd.notna(target.loc[t]) else np.nan
                rows.append({'horizon':h,'forecast_date':t,'ridge_standardized':alpha,
                    'cond_standardized_forward_X':cond,'n_train':len(y),
                    'predicted_mean_log_excess':float(x0@b),'realized_mean_log_excess':yreal,
                    'slope_norm_standardized':float(np.linalg.norm(b[1:])),
                    **{f'coef_z_{name}':float(b[i+1]) for i,name in enumerate(fwd.columns)}})
    return pd.DataFrame(rows)

def aligned_forecast_evaluation(research,extra_forecasts=None,bootstrap_reps:int=399,seed:int=202609):
    """Matched H-month forecast scoring for v0.8 models + optional direct models.

    All reported models, both historical baselines, targets and *all factors*
    share exactly the same forecast-origin dates within each horizon.
    Both total-return and curve-repricing errors are measured on that sample.
    Method comparison against the same baseline is ALWAYS the total H-month
    excess-return forecast; this makes DIRECT and decomposed fairly comparable.
    """
    rows=[];individual=[];loss_by_date=[];samples={}
    extra_forecasts=extra_forecasts or {}
    hs=sorted(research.factor_targets)
    for h in hs:
        target=research.factor_targets[h]
        if not {'total','curve','carry'}.issubset(target):continue
        full={m:_p(panel) for (hh,m),panel in {**research.forecasts,**extra_forecasts}.items() if hh==h}
        if not full:continue
        full['UNCONDITIONAL']=_p(research.unconditional_forecasts.get(h))
        full['CARRY_ONLY']=_p(research.carry_only_forecasts.get(h))
        # Need dates for *every* model, both benchmarks, and each factor
        real=target['total'].reindex(columns=FACTOR_NAMES)
        curve=target['curve'].reindex(columns=FACTOR_NAMES)
        common=real.dropna().index.intersection(curve.dropna().index)
        for m,frame in full.items():
            common=common.intersection(frame.dropna().index)
        common=common.sort_values()
        if len(common)<3:continue
        dates=pd.DatetimeIndex(common)
        samples[h]=dates
        ref=full['UNCONDITIONAL'].loc[dates]
        obs=real.loc[dates]
        base_loss=(obs-ref)**2
        for model,frame in full.items():
            f=frame.loc[dates]
            mse=(obs-f)**2
            baseline_sse=float(base_loss.to_numpy().sum())
            sse=float(mse.to_numpy().sum())
            r2=1-sse/baseline_sse if baseline_sse>0 else np.nan
            dt_loss=(base_loss-mse).mean(axis=1)
            tests=hac_mean_test(dt_loss,max(h-1,6))
            boot=circular_block_bootstrap_mean(dt_loss,max(12,2*h),bootstrap_reps,seed=seed+h)
            diff=f.sub(obs)
            r={'horizon':h,'model':model,'sample_start':str(dates[0].date()),
               'sample_end':str(dates[-1].date()),'n_dates':len(dates),
               'n_asset_months':int(len(dates)*len(FACTOR_NAMES)),
               'pooled_oos_r2_total_vs_uncond':r2,
               'rmse_total':float(np.sqrt(mse.to_numpy().mean())),
               'mae_total':float(np.abs(diff.to_numpy()).mean()),
               'mean_loss_gain_vs_uncond':tests['mean'],
               'mean_loss_gain_t_hac':tests['t_hac'],
               'mean_loss_gain_p_hac':tests['p_hac'],
               'loss_gain_ci_low_block':boot['ci_low'],
               'loss_gain_ci_high_block':boot['ci_high'],
               'loss_gain_p_block':boot['bootstrap_p_two_sided'],
               'block_length':boot['block_length'],
               'hac_lags':max(h-1,6)}
            rows.append(r)
            z=pd.DataFrame({'observed':obs.stack(),'forecast':f.stack(),
                'baseline':ref.stack()})
            z.index.names=['forecast_date','factor']
            z['model']=model;z['horizon']=h
            # complete panel for audit, includes benchmark reference for each date
            loss_by_date.append(pd.DataFrame({'horizon':h,'model':model,
              'forecast_date':dates,'mean_loss_gain':dt_loss.to_numpy()}))
            for factor in FACTOR_NAMES:
                y=obs[factor];yh=f[factor]; b=ref[factor]
                b_sse=float(((y-b)**2).sum());e_sse=float(((y-yh)**2).sum())
                individual.append({'horizon':h,'model':model,'factor':factor,'n_dates':len(dates),
                    'oos_r2_total_vs_uncond':1-e_sse/b_sse if b_sse>0 else np.nan,
                    'rmse_total':float(np.sqrt(((y-yh)**2).mean())),
                    'forecast_realized_corr':float(y.corr(yh)) if (y.std()>1e-14 and yh.std()>1e-14) else np.nan})
    table=pd.DataFrame(rows)
    by_factor=pd.DataFrame(individual)
    by_date=pd.concat(loss_by_date,ignore_index=True) if loss_by_date else pd.DataFrame()
    return table,by_factor,by_date,samples


def aligned_rotation_evaluation(research,rates,extra_backtests=None,bootstrap_reps=399,seed=202609):
    """Portfolio horse-race with *the same realized monthly dates* per H.

    No splicing/mismatched sample Sharpe. Excess Sharpe uses same observed
    funding cash path; realized backtests have 1-month execution and costs.
    """
    bt={**research.backtests,**(extra_backtests or {})}
    rows=[];tests=[];diagnostics=[];samples={}
    cash=rates.cash_returns
    for h in sorted({x[0] for x in bt}):
        models={m:x for (hh,m),x in bt.items() if hh==h}
        if not {'UNCONDITIONAL','CARRY_ONLY'}.issubset(models):continue
        dates=None
        for m,b in models.items():
            r=b['returns'].dropna()
            dates=r.index if dates is None else dates.intersection(r.index)
        dates=dates.intersection(cash.dropna().index).sort_values()
        if len(dates)<3:continue
        samples[h]=dates
        for model,b in models.items():
            r=b['returns'].reindex(dates)
            metrics=performance_metrics(r,cash=cash,turnover=b['turnover'])
            rows.append({'horizon':h,'model':model,'n_months':len(dates),'start':str(dates[0].date()),
              'end':str(dates[-1].date()),**metrics.to_dict()})
            gross=b['overlay_pnl'].reindex(dates)
            turnover=b['turnover'].reindex(dates)
            # Costs inferred exactly using cash + overlay P&L - realized net returns
            observed_cost=(cash.reindex(dates)+gross-r)
            fa=b['factor_alloc']; uw=b['overlay_weights']
            diagnostics.append({'horizon':h,'model':model,
               'mean_gross_overlay_monthly':float(gross.mean()),
               'mean_turnover_monthly':float(turnover.mean()),
               'mean_paid_cost_monthly':float(observed_cost.mean()),
               'mean_net_excess_monthly':float((r-cash.reindex(dates)).mean()),
               'avg_factor_gross_at_formation':float(fa.abs().sum(axis=1).mean()) if len(fa) else np.nan,
               'avg_underlying_gross_at_formation':float(uw.abs().sum(axis=1).mean()) if len(uw) else np.nan,
               'fraction_flat_factor_months':float((fa.abs().sum(axis=1)<1e-12).mean()) if len(fa) else np.nan})
            for reference in ('UNCONDITIONAL','CARRY_ONLY'):
                if model==reference:continue
                rref=models[reference]['returns'].reindex(dates)
                spread=r-rref
                stat=hac_mean_test(spread,lags=max(12,h))
                boot=circular_block_bootstrap_mean(spread,block_length=max(12,h),replicates=bootstrap_reps,seed=seed+h)
                tests.append({'horizon':h,'model':model,'benchmark':reference,'n_months':len(dates),
                    'annualized_mean_spread':stat['mean']*12,'spread_t_hac':stat['t_hac'],
                    'spread_p_hac':stat['p_hac'],
                    'annualized_ci_low_block':boot['ci_low']*12,
                    'annualized_ci_high_block':boot['ci_high']*12,
                    'p_block':boot['bootstrap_p_two_sided']})
    return pd.DataFrame(rows),pd.DataFrame(tests),pd.DataFrame(diagnostics),samples


@dataclass
class RatesV09Result:
    predictive: pd.DataFrame
    predictive_by_factor: pd.DataFrame
    predictive_losses_by_date: pd.DataFrame
    rotation: pd.DataFrame
    rotation_spreads: pd.DataFrame
    signal_execution: pd.DataFrame
    common_samples: pd.DataFrame
    direct_forecasts: dict
    cp_first_stage: pd.DataFrame
    metadata: dict

    def export(self,directory):
        folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
        for name in ('predictive','predictive_by_factor','predictive_losses_by_date',
            'rotation','rotation_spreads','signal_execution','common_samples','cp_first_stage'):
            getattr(self,name).to_csv(folder/f'rates_v09_{name}.csv',index=False)
        for (h,model),frame in self.direct_forecasts.items():
            frame.to_csv(folder/f'rates_v09_direct_forecast_{h:02d}m_{model}.csv')
        (folder/'rates_v09_manifest.json').write_text(json.dumps(self.metadata,indent=2,default=str))
        return folder


def analyze_rates_v09(rates,research,macro_raw=None,direct_models=('FB','CP'),
                      min_train=60,ridge=1.,cost_bps=5.,bootstrap_reps=399,seed=202609):
    """Conduct rigorously aligned analysis of existing v0.8 research result."""
    direct,c=build_direct_total_forecasts(rates,research,macro_raw,
        models=direct_models,min_train=min_train,ridge=ridge)
    extra_bt={}
    runner=RatesEmpiricalBondPremiaEngine(rates)
    for key,panel in direct.items():
        extra_bt[key]=runner._backtest(panel,cost_bps)
    forecast,by_factor,losses,forecast_samples=aligned_forecast_evaluation(
        research,direct,bootstrap_reps=bootstrap_reps,seed=seed)
    rotation,spreads,signal,rotation_samples=aligned_rotation_evaluation(
        research,rates,extra_bt,bootstrap_reps=bootstrap_reps,seed=seed)
    cp_diag=cp_first_stage_diagnostics(rates,research,step_months=12,min_train=min_train)
    s=[]
    for h in sorted(set(forecast_samples)|set(rotation_samples)):
        f=forecast_samples.get(h,pd.DatetimeIndex([]));r=rotation_samples.get(h,pd.DatetimeIndex([]))
        s.append({'horizon':h,'forecast_origin_start':str(f.min().date()) if len(f) else None,
            'forecast_origin_end':str(f.max().date()) if len(f) else None,'n_forecast_origins':len(f),
            'realized_strategy_start':str(r.min().date()) if len(r) else None,
            'realized_strategy_end':str(r.max().date()) if len(r) else None,'n_strategy_months':len(r)})
    meta={'research_version':'0.9.0','reference_legacy':'Rates v0.8 / ERP v0.2.5',
       'underlying_factors':'funded PCA L/S/C (risk exposures, not proof each priced)',
       'direct_models':list(direct_models),'direct_model_description':'same v0.8 learner; response is TOTAL H-month excess return, carry set to zero in trainer',
       'all_models_common_by_horizon':True,'inference':'HAC lags >= H-1 on H-month forecast LOSS differences, paired moving-block bootstrap on per-date factor-average loss gain',
       'bootstrap_reps':bootstrap_reps,'bootstrap_seed':seed,
       'strategy':'existing v0.8 signed mean-confidence and monthly next-period execution, identical limits/costs',
       'timestamp_discipline':'labels for formation t trained only from origins s<=t-H; no live forecasts of last incomplete H months',
       'vintage_limit':'revised macro time series are NOT historical release-point-in-time; this is not publication-vintage OOS.',
       'negative_result_discipline':'small/negative results not evidence of no risk premium; inference is exploratory across many model comparisons',
       'data_as_of':str(research.snapshot.as_of)}
    return RatesV09Result(forecast,by_factor,losses,rotation,spreads,signal,pd.DataFrame(s),direct,cp_diag,meta)
