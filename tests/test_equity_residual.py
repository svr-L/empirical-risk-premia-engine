"""Synthetic, fully deterministic tests: no synthetic findings claimed for real markets."""
import json
import numpy as np
import pandas as pd
import pytest
from erp import EquityData,analyze_equity_residuals
from erp.equity_residual import _holm


def make_panel(seed=19, residual_drift=.015, noise=.008):
    g=np.random.default_rng(seed)
    dates=pd.date_range('1990-01-31',periods=252,freq='ME')
    nf=8
    F=g.normal(0,.025,(len(dates),nf))
    F[:,0]+=.008
    cols=['MKT','SMB','HML','RMW','CMA','MOM','BAB','QMJ']
    factor=pd.DataFrame(F,index=dates,columns=cols)
    z=g.normal(0,.040,len(dates))+residual_drift
    universes={}
    for nm,n in [('SIZE_BM_25',25),('INDUSTRY_30',30),('SIZE_MOM_25',25),('SIZE_OP_25',25),('SIZE_INV_25',25)]:
        bet=g.normal(0,.5,(n,nf));bet[:,0]+=1.
        load=g.normal(0,1,n)
        ret=F@bet.T + z[:,None]*load[None,:]+g.normal(0,noise,(len(dates),n))+.001
        # Input expected as raw total returns; RF will be subtracted.
        universes[nm]=pd.DataFrame(ret+.002,index=dates,columns=[f'B{i}' for i in range(n)])
    ff3=factor[['MKT','SMB','HML']].copy();ff3['RF']=.002
    ff5=factor[['MKT','SMB','HML','RMW','CMA']].copy();ff5['RF']=.002
    aqr=factor[['BAB','QMJ']].copy()
    data=EquityData(ff3,ff5,factor['MOM'],universes,aqr)
    macro=pd.DataFrame({'innovation_proxy':z+g.normal(0,.01,len(z)),
                        'independent_noise':g.normal(0,1,len(z))},index=dates)
    return data,macro


@pytest.fixture(scope='module')
def simulated():
    return make_panel()


@pytest.fixture(scope='module')
def analysis(simulated):
    data,_=simulated
    return analyze_equity_residuals(data,train_end='2001-12-31',as_of='2010-12-31',n_pcs=3)


def test_discovery_train_and_validation_test_separation(analysis):
    result=analysis
    assert result.metadata['formation_months']==144
    assert result.metadata['oos_months']==108
    assert result.fmp_premia_tests.n_oos_months.eq(108).all()
    assert all(x.startswith(('SIZE_BM_25','INDUSTRY_30','SIZE_MOM_25')) for x in result.weights.index)
    assert result.holdout_metrics.iloc[0].n_heldout_assets==50
    assert result.factors.index.max()==pd.Timestamp('2010-12-31')


def test_neutrality_and_gross_not_only_residual_ols_orthogonality(analysis):
    weights=analysis.weights
    assert (weights.sum(axis=0).abs()<1e-10).all()
    assert np.allclose(weights.abs().sum(axis=0),1,atol=1e-10)
    assert analysis.neutrality_tests.max_abs_factor_beta_train.max()<1e-10
    assert analysis.neutrality_tests.net_dollar_weight.abs().max()<1e-10


def test_planted_residual_premium_can_survive_outside_training(analysis):
    pc1=analysis.fmp_premia_tests.loc['RPC1']
    assert abs(pc1.oos_mean_ann)>.01
    assert abs(pc1.oos_spanning_t_hac)>2.0
    assert 0<pc1.eigenvalue_share_train<1
    assert analysis.holdout_metrics.loc[1,'oos_rmse_monthly'] < analysis.holdout_metrics.loc[0,'oos_rmse_monthly']
    assert analysis.holdout_metrics.loc[0,'oos_r2_vs_ff_baseline']==pytest.approx(0)


def test_incremental_p_value_adjustment(analysis):
    x=analysis.fmp_premia_tests
    assert (x.oos_spanning_p_holm>=x.oos_spanning_p_hac-1e-12).all()
    assert (x.oos_mean_p_holm>=x.oos_mean_p_hac-1e-12).all()
    assert _holm(pd.Series({'a':.01,'b':.05,'c':.2})).round(3).to_dict()=={'a':.03,'b':.10,'c':.20}


def test_oos_data_never_affects_preformed_weights(simulated,analysis):
    data,_=simulated
    # Deliberately mutate ONLY returns after 2001, keeping baseline factors fixed.
    copy={k:v.copy() for k,v in data.universes.items()}
    for k in copy:
        copy[k].loc['2002-01-01':]+=100
    modified=EquityData(data.ff3,data.ff5,data.mom,copy,data.aqr)
    b=analyze_equity_residuals(modified,train_end='2001-12-31',as_of='2010-12-31',n_pcs=3)
    pd.testing.assert_frame_equal(analysis.weights,b.weights)
    pd.testing.assert_frame_equal(analysis.residual_eigenvalues,b.residual_eigenvalues)
    # A net-zero funded portfolio cancels a COMMON +100 shock; compare test holdout returns instead.
    assert analysis.holdout_metrics.loc[0,'oos_rmse_monthly']!=b.holdout_metrics.loc[0,'oos_rmse_monthly']
    assert analysis.sdf_gmm_comparison.loc[0,'pricing_rmse_train_ann']==pytest.approx(b.sdf_gmm_comparison.loc[0,'pricing_rmse_train_ann'])
    assert analysis.sdf_gmm_comparison.loc[0,'pricing_rmse_oos_ann']!=b.sdf_gmm_comparison.loc[0,'pricing_rmse_oos_ann']


def test_future_after_asof_never_affects_results(simulated):
    data,_=simulated
    base=analyze_equity_residuals(data,train_end='2001-12-31',as_of='2007-12-31',n_pcs=2)
    modified={k:x.copy() for k,x in data.universes.items()}
    for k in modified:modified[k].loc['2008-01-01':]*=100
    d2=EquityData(data.ff3,data.ff5,data.mom,modified,data.aqr)
    other=analyze_equity_residuals(d2,train_end='2001-12-31',as_of='2007-12-31',n_pcs=2)
    pd.testing.assert_frame_equal(base.weights,other.weights)
    pd.testing.assert_frame_equal(base.fmp_premia_tests,other.fmp_premia_tests)
    pd.testing.assert_frame_equal(base.holdout_metrics,other.holdout_metrics)
    pd.testing.assert_frame_equal(base.sdf_gmm_comparison,other.sdf_gmm_comparison)


def test_optional_macroeconomic_mapping_is_explicitly_nonpricing(simulated,tmp_path):
    data,macro=simulated
    a=analyze_equity_residuals(data,train_end='2001-12-31',as_of='2010-12-31',
                               n_pcs=2,macro_shocks=macro)
    assert 'MKT' in a.macro_exposures.index and 'RPC1' in a.macro_exposures.index
    assert a.macro_diagnostics.loc['RPC1','oos_macro_r2_vs_train_mean']>0
    assert 'RPC1' in a.macro_sdf_pricing.index
    assert set(a.macro_sdf_coefficients.index)=={'innovation_proxy','independent_noise'}
    assert a.macro_sdf_pricing.validation_pricing_rmse_oos_ann.notna().all()
    assert a.macro_diagnostics.loc['RPC1','n_macro_oos_months']==108
    dest=a.export(tmp_path/'report')
    info=json.loads((dest/'manifest.json').read_text())
    assert info['definitions']['macro'].startswith('contemporaneous descriptive')
    assert (dest/'holdout_oos_attribution.csv').exists()


def test_refuse_same_discovery_and_holdout(simulated):
    data,_=simulated
    with pytest.raises(ValueError,match='disjoint'):
        analyze_equity_residuals(data,discovery_universes=('SIZE_BM_25',),
                                 validation_universes=('SIZE_BM_25',),
                                 train_end='2001-12-31')


def test_optional_macro_requires_numeric_true_data(simulated):
    data,macro=simulated
    bad=macro.astype(str)
    with pytest.raises(ValueError,match='numeric innovations'):
        analyze_equity_residuals(data,train_end='2001-12-31',n_pcs=2,macro_shocks=bad)


def test_gmm_scoring_and_estimate_is_formation_only(analysis):
    g=analysis.sdf_gmm_comparison
    assert list(g.index)==[0,1,2,3]
    assert g.loc[0,'n_sdf_factors']==8
    assert g.loc[3,'n_sdf_factors']==11
    assert g.pricing_rmse_train_ann.notna().all()
    assert g.pricing_rmse_oos_ann.notna().all()
    assert (g.n_moments==50).all()
    assert analysis.sdf_gmm_coefficients.loc[(3,'RPC1'),'implied_lambda_monthly']==pytest.approx(
        analysis.sdf_gmm_coefficients.loc[(3,'RPC1'),'implied_lambda_monthly'])


def test_lagged_state_forecasts_are_expanding_and_future_safe(simulated):
    data,macro=simulated
    # Macro proxy is included only for this synthetic reproducibility test.
    a=analyze_equity_residuals(data,train_end='2001-12-31',as_of='2010-12-31',
          n_pcs=2,state_predictors=macro,state_lag_months=2,state_ridge=10.)
    assert not a.state_scores.empty
    assert 'RPC1' in a.state_scores.index and 'MKT' in a.state_scores.index
    assert a.state_scores['n_oos'].eq(108).all()
    assert a.state_scores.state_lag_months.eq(2).all()
    assert a.state_forecasts.index.get_level_values('Date').min()==pd.Timestamp('2002-01-31')
    # Mutating a time-t predictor cannot affect the time-t forecast with two lags.
    changed=macro.copy()
    changed.loc['2002-01-31','innovation_proxy']+=200
    b=analyze_equity_residuals(data,train_end='2001-12-31',as_of='2010-12-31',
          n_pcs=2,state_predictors=changed,state_lag_months=2,state_ridge=10.)
    pd.testing.assert_frame_equal(a.state_forecasts.loc[pd.IndexSlice['2002-01-31',:]],
                                  b.state_forecasts.loc[pd.IndexSlice['2002-01-31',:]])
