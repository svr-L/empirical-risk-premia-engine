"""All observations in this module are synthetic and NOT empirical results."""
import numpy as np
import pandas as pd
import pytest
from erp.equity_v012 import (analyze_equity_v012,_factor_weights,
                              _fmb_train_oos,_gmm_train_oos,_paired_block_bootstrap)
from erp.equity_shocks import (_jk_parse,load_jk_monetary_shocks,
                               load_inflation_release_surprises,analyze_equity_shock_mapping)
from erp.equity_residual import analyze_equity_residuals
from erp.equity import EquityData
from test_equity_residual import make_panel
from io import BytesIO

@pytest.fixture(scope='module')
def synthetic():
    data,macro=make_panel(seed=19)
    base=analyze_equity_residuals(data,train_end='2001-12-31',as_of='2010-12-31',n_pcs=3)
    return data,base

@pytest.fixture(scope='module')
def run(synthetic):
    return analyze_equity_v012(*synthetic,formation_window=120,rebalance_months=12,
        n_pcs=2,gammas=(0.,12.),bootstrap_draws=99)

def test_baseline_fixed_common_oos_and_exports(run,tmp_path):
    assert len(run.pricing_comparison)==6 # 3 models x 2 estimators
    assert (run.pricing_comparison.oos_months==108).all()
    assert set(run.pricing_comparison.estimator)=={'FMB','GMM'}
    assert len(run.bootstrap_comparison)==4
    assert not run.rolling_returns.isna().any().any()
    out=run.export(tmp_path)
    assert (out/'pricing_comparison.csv').exists()
    assert (out/'rolling_diagnostics.csv').exists()


def test_rolling_weights_train_neutral_and_pre_oos_dates(run):
    d=run.rolling_diagnostics
    assert d.training_beta_max_abs.abs().max()<1e-8
    assert d.net_dollar.abs().max()<1e-8
    assert np.allclose(d.gross,1)
    assert (pd.to_datetime(d.formed_as_of)<pd.to_datetime(d.holding_from)).all()
    assert (d.l1_turnover_on_rebalance.dropna()>=0).all()


def test_rolling_future_mutation_cannot_change_first_weights(synthetic,run):
    data,base=synthetic
    mutated={k:v.copy() for k,v in data.universes.items()}
    for v in mutated.values():v.loc['2002-01-01':]+=np.linspace(0,20,len(v.loc['2002-01-01':]))[:,None]
    other=EquityData(data.ff3,data.ff5,data.mom,mutated,data.aqr)
    new=analyze_equity_v012(other,base,formation_window=120,rebalance_months=12,
        n_pcs=2,gammas=(0.,12.),bootstrap_draws=99)
    d1=run.rolling_weights.loc[run.rolling_weights.formed_as_of==pd.Timestamp('2001-12-31')].reset_index(drop=True)
    d2=new.rolling_weights.loc[new.rolling_weights.formed_as_of==pd.Timestamp('2001-12-31')].reset_index(drop=True)
    pd.testing.assert_frame_equal(d1,d2)


def test_gamma_0_equals_ordinary_pca_eigenspace():
    g=np.random.default_rng(9);f=g.normal(size=(140,3));y=f@g.normal(size=(3,22))+g.normal(size=(140,22))
    w,ev,c=_factor_weights(y,f,n_pcs=3,gamma=0)
    assert np.allclose(w.sum(axis=0),0,atol=1e-8)
    assert np.max(abs(c.T@w))<1e-8
    assert np.all(np.isfinite(ev))


def test_bootstrap_direction_and_common_calendar():
    g=np.random.default_rng(91)
    base=np.ones(100)*2+g.normal(size=100)*.1
    alt=base-.3
    x=_paired_block_bootstrap(base,alt,draws=199)
    assert x['mse_improvement_monthly']==pytest.approx(.3)
    assert x['ci_excludes_zero']
    with pytest.raises(ValueError):_paired_block_bootstrap(np.ones(30),np.ones(29))


def _jk_synth(n=170):
    d=pd.date_range('1990-01-31',periods=n,freq='ME')
    g=np.random.default_rng(3);mp=g.normal(0,.1,n);cb=g.normal(0,.09,n)
    return pd.DataFrame({'year':d.year,'month':d.month,'pc1_hf':mp+cb,
        'SP500_hf':g.normal(size=n),'MP_pm':mp,'CBI_pm':cb,
        'MP_median':mp,'CBI_median':cb})


def test_jk_exact_identity_and_parser():
    df=_jk_synth()
    out=_jk_parse(df.to_csv(index=False).encode())
    assert len(out)==170
    assert 'SP500_hf' not in out # high-frequency EQUITY response is not a monetary shock
    assert np.allclose(out.MP_median+out.CBI_median,out.pc1_hf)
    bad=df.copy();bad.loc[5,'pc1_hf']+=.4
    with pytest.raises(ValueError,match='identity'):_jk_parse(bad.to_csv(index=False).encode())


def test_jk_loader_cache_and_no_lookahead(tmp_path):
    path=tmp_path/'jk.csv';path.write_text(_jk_synth().to_csv(index=False))
    a=load_jk_monetary_shocks(cache_dir=tmp_path/'cache',path=path,as_of='2001-12-31')
    b=load_jk_monetary_shocks(cache_dir=tmp_path/'cache',as_of='2001-12-31')
    pd.testing.assert_frame_equal(a.shocks,b.shocks)
    assert a.shocks.index.max()==pd.Timestamp('2001-12-31')


def test_inflation_uses_release_date_not_cpi_reference_month(tmp_path):
    path=tmp_path/'inflation.csv'
    pd.DataFrame({'release_date':['2000-02-15','2000-03-15'],
      'actual':[.3,.7],'consensus':[.2,.4],'consensus_source':['survey A','survey A']}).to_csv(path,index=False)
    z=load_inflation_release_surprises(path)
    assert z.shocks.loc['2000-02-29','CPI_RELEASE_SURPRISE']==pytest.approx(.1)
    assert z.shocks.index.min()==pd.Timestamp('2000-02-29')


def test_shock_mapping_lags_contiguous_series():
    df=_jk_synth(252);shocks=_jk_parse(df.to_csv(index=False).encode())
    from erp.equity_shocks import EconomicShockPanel
    sh=EconomicShockPanel(shocks,{'source':'synthetic test'})
    rng=np.random.default_rng(11);d=shocks.index
    ff=pd.DataFrame({'MKT': .5*shocks.MP_median-.2*shocks.CBI_median+rng.normal(0,.02,len(d))},index=d)
    pc=pd.DataFrame({'RPC1':rng.normal(size=len(d))*.05},index=d)
    t=analyze_equity_shock_mapping(ff,pc,sh,train_end='2001-12-31')
    assert set(t.factor)=={'MKT','RPC1'}
    assert t.loc[(t.factor=='MKT') & (t.lag==0),'oos_r2_vs_train_mean'].iloc[0]>.4
    assert t.loc[t.lag==1,'interpretation'].eq('forecast using prior month event shocks').all()

def test_cpi_release_mapping_does_not_shift_sparse_rows_across_months(tmp_path):
    from erp.equity_shocks import EconomicShockPanel, _one_shock_mapping
    d=pd.date_range('1990-01-31',periods=240,freq='ME')
    sparse=d[::2]
    shock=pd.DataFrame({'CPI_RELEASE_SURPRISE':np.ones(len(sparse))},index=sparse)
    prices=pd.Series(np.random.default_rng(0).normal(size=len(d)),index=d)
    z=pd.concat([prices,shock.shift(1,freq='ME')],axis=1,join='inner').dropna()
    assert (z.index==sparse+pd.offsets.MonthEnd(1)).all()
    mapping=analyze_equity_shock_mapping(pd.DataFrame({'MKT':prices}),
              pd.DataFrame(index=d),EconomicShockPanel(shock,{}),train_end='2000-12-31')
    assert set(mapping.shock_group)=={'CPI_actual_minus_prerelease_consensus'}

def test_identified_policy_prices_fit_common_holdout_calendar():
    from erp.equity_shocks import estimate_jk_shock_prices,EconomicShockPanel
    date=pd.date_range('1990-01-31',periods=252,freq='ME')
    rng=np.random.default_rng(150)
    ff=pd.DataFrame(rng.normal(0,.02,(252,8)),index=date,columns=['MKT','SMB','HML','RMW','CMA','MOM','BAB','QMJ'])
    mp=rng.normal(0,.06,252);cbi=rng.normal(0,.05,252)
    shock=EconomicShockPanel(pd.DataFrame({'MP_median':mp,'CBI_median':cbi},index=date),{})
    beta=rng.normal(0,.6,(50,8));bshock=rng.normal(0,.5,(50,2))
    hold=pd.DataFrame(ff.to_numpy()@beta.T+
       np.column_stack([mp,cbi])@bshock.T+rng.normal(0,.01,(252,50)),
       index=date,columns=[str(i) for i in range(50)])
    diag,lam=estimate_jk_shock_prices(hold,ff,shock,train_end='2001-12-31')
    assert len(diag)==2
    assert (diag.oos_months==108).all()
    assert {'MP_median','CBI_median'}.issubset(set(lam.factor))
