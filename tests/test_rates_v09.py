"""v0.9 matched OOS comparability, inference and direct-target controls."""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from erp.rates_v09 import (hac_mean_test,circular_block_bootstrap_mean,
    aligned_forecast_evaluation,aligned_rotation_evaluation,analyze_rates_v09)
from erp import RatesData,RatesPremiaEngine,RatesEmpiricalBondPremiaEngine


def panel(idx, x):
    rows=[]
    for d in idx:
        for factor in ('Level','Slope','Curvature'):
            rows.append((d,factor,x))
    out=pd.DataFrame(rows,columns=['forecast_date','factor','total_mean_h']).set_index(['forecast_date','factor'])
    return out


def test_matched_sample_never_compares_mismatched_forecast_dates():
    idx=pd.date_range('2010-01-31',periods=42,freq='ME')
    obs=pd.DataFrame(.02,index=idx,columns=['Level','Slope','Curvature'])
    orig=SimpleNamespace(factor_targets={3:{'total':obs,'curve':obs,'carry':obs*0}},
        forecasts={(3,'FB'):panel(idx,.02), (3,'CP'):panel(idx[9:],.019)},
        unconditional_forecasts={3:panel(idx,.0)},carry_only_forecasts={3:panel(idx,.0)})
    out,detail,loss,samples=aligned_forecast_evaluation(orig,bootstrap_reps=99)
    assert set(out.model)=={'FB','CP','UNCONDITIONAL','CARRY_ONLY'}
    assert set(out.n_dates)=={33}
    assert samples[3].min()==idx[9]
    assert float(out.set_index('model').loc['FB','pooled_oos_r2_total_vs_uncond'])==pytest.approx(1.)
    assert len(loss)==4*33
    assert len(detail)==4*3


def test_block_resampling_and_hac_no_artificial_significance():
    x=np.zeros(28)
    out=circular_block_bootstrap_mean(x,block_length=12,replicates=79)
    assert out['ci_high']==pytest.approx(0.)
    assert out['bootstrap_p_two_sided']==1.0
    z=hac_mean_test(np.ones(35)*.005,6)
    assert z['n']==35


def test_direct_target_matches_known_horizon_timing():
    rng=np.random.default_rng(16)
    n=232; idx=pd.date_range('1991-01-31',periods=n,freq='ME')
    base=.04+np.cumsum(rng.normal(0,.0003,n));term=np.linspace(0,1,10)
    yields=pd.DataFrame({f'{i}Y':base+.001*i+term[i-1]*np.cumsum(rng.normal(0,.0001,n)) for i in range(1,11)},index=idx)
    funding=pd.Series(.025,index=idx)
    rates=RatesPremiaEngine(RatesData(yields,funding))
    snap=rates.fit(as_of=idx[-1],pca_train_end='2002-12-31')
    research=RatesEmpiricalBondPremiaEngine(rates).fit(snap,horizons=(3,),models=('FB','CP'),min_train=12,cost_sensitivity=(),require_full_grid=False)
    assert research.forecasts
    analysis=analyze_rates_v09(rates,research,direct_models=('FB',),min_train=12,
        bootstrap_reps=19,cost_bps=5.)
    assert (3,'DIRECT_FB') in analysis.direct_forecasts
    assert set(analysis.predictive['model'])=={'FB','CP','DIRECT_FB','UNCONDITIONAL','CARRY_ONLY'}
    assert analysis.predictive['n_dates'].nunique()==1
    assert set(analysis.rotation['model'])=={'FB','CP','DIRECT_FB','UNCONDITIONAL','CARRY_ONLY'}
    assert analysis.rotation['n_months'].nunique()==1
    assert analysis.common_samples.loc[0,'n_forecast_origins']>0
    assert analysis.common_samples.loc[0,'n_strategy_months']>0
    assert analysis.metadata['vintage_limit'].startswith('revised macro')
