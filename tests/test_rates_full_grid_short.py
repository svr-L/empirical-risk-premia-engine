"""Five original v0.8 families, three horizons, two benchmarks: small synthetic full-grid smoke."""
import numpy as np
import pandas as pd
from erp import RatesData,RatesPremiaEngine,RatesEmpiricalBondPremiaEngine


def test_all_15_model_horizon_pairs():
    rng=np.random.default_rng(87)
    idx=pd.date_range('1990-01-31',periods=208,freq='ME');n=len(idx)
    lev=.05+np.cumsum(rng.normal(0,.0005,n))
    slp=.006+np.cumsum(rng.normal(0,.00015,n))
    curv=np.cumsum(rng.normal(0,.00005,n))
    curve=pd.DataFrame({f'{m}Y':lev+.001*m+slp*(m-5)/10+curv*(1-2*((m-5)/5)**2) for m in range(1,11)},index=idx)
    ff=pd.Series(.02+.006*np.sin(np.arange(n)/30),index=idx)
    macro=pd.DataFrame({'CPI':200*np.exp(.002*np.arange(n))*(1+.001*np.sin(np.arange(n)/15)),
         'INDPRO':90*np.exp(.001*np.arange(n))*(1+.015*np.sin(np.arange(n)/17)),
         'BREAKEVEN10Y':2+.15*np.sin(np.arange(n)/20)},index=idx)
    funded=RatesPremiaEngine(RatesData(curve,ff))
    snap=funded.fit(as_of=idx[-1],pca_train_end='2002-12-31')
    res=RatesEmpiricalBondPremiaEngine(funded).fit(snap,macro_raw=macro,
         horizons=(3,6,12),min_train=15,cost_sensitivity=(5,),require_full_grid=True)
    assert len(res.forecasts)==15
    assert len(res.backtests)==21
    assert len(res.fb_12m_diagnostics)==4
    assert len(res.cost_sensitivity)==15
    assert len(res.strategy_results)==21
    assert all(len(x)>0 for x in res.forecasts.values())
    assert res.strategy_results['CAGR'].notna().all()
