import numpy as np
import pandas as pd
import pytest
from erp import (analyze_equity_residuals,analyze_equity_factor_extensions,
                 residual_eigenvalue_null,make_equity_macro_panel,load_equity_fred_macro)
from test_equity_residual import make_panel

@pytest.fixture(scope='module')
def synthetic():
    data,_=make_panel(seed=19)
    baseline=analyze_equity_residuals(data,train_end='2001-12-31',as_of='2010-12-31',n_pcs=3)
    return data,baseline

@pytest.fixture(scope='module')
def ext(synthetic):
    return analyze_equity_factor_extensions(*synthetic,n_pcs=3,null_draws=39,rp_gamma=(0.,12.,60.),gx_k=3)

def test_eigenvalue_null_common_signal_and_independent_noise():
    rng=np.random.default_rng(19)
    e=rng.normal(size=(150,20))
    common=rng.normal(size=150)
    e[:,:10]+=common[:,None]*1.5
    scores=residual_eigenvalue_null(e,n_pcs=3,draws=79,seed=99)
    assert scores.loc['RPC1','passes_max_95_FWER']
    assert scores.loc['RPC1','eigenvalue']>scores.loc['RPC1','null_max_95_FWER']
    assert (scores.p_rank_exploratory>=1/80-1e-10).all()

def test_extension_exports_and_model_comparison(ext,tmp_path):
    assert set(ext.pricing_oriented_premia.gamma)=={0.,12.,60.}
    assert len(ext.pricing_oriented_premia)==9
    assert len(ext.pricing_oriented_sdf)==9
    assert len(ext.three_pass_observed)==8
    assert ext.pricing_oriented_premia.train_beta_max_abs.max()<1e-9
    assert np.allclose(ext.pricing_oriented_weights.sum(axis=0),0,atol=1e-9)
    assert np.allclose(ext.pricing_oriented_weights.abs().sum(axis=0),1,atol=1e-10)
    assert (ext.pricing_oriented_premia.holm_adjusted_oos_alpha_p_all_tests>=ext.pricing_oriented_premia.oos_alpha_p_hac-1e-12).all()
    out=ext.export(tmp_path/'ex')
    assert (out/'residual_eigenvalue_null.csv').is_file()
    assert (out/'pricing_oriented_sdf.csv').is_file()
    assert (out/'three_pass_observed_premia.csv').is_file()

def test_extension_has_no_future_weight_lookahead(synthetic,ext):
    data,baseline=synthetic
    d2={k:v.copy() for k,v in data.universes.items()}
    for x in d2.values():x.loc['2002-01-01':]*=10
    from erp import EquityData
    modified=EquityData(data.ff3,data.ff5,data.mom,d2,data.aqr)
    other=analyze_equity_factor_extensions(modified,baseline,n_pcs=3,null_draws=39,rp_gamma=(0.,12.,60.),gx_k=3)
    pd.testing.assert_frame_equal(ext.pricing_oriented_weights,other.pricing_oriented_weights)
    pd.testing.assert_frame_equal(ext.eigen_null,other.eigen_null)
    assert not ext.pricing_oriented_premia.oos_mean_ann.equals(other.pricing_oriented_premia.oos_mean_ann)


def _fred_panel():
    rng=np.random.default_rng(4)
    dt=pd.date_range('1990-01-31',periods=240,freq='ME')
    base=np.arange(len(dt))
    return pd.DataFrame({'CPIAUCSL':100+base*.2, 'INDPRO':90+base*.1,
         'DGS10':3.+rng.normal(0,.1,len(dt)), 'DGS2':2.+rng.normal(0,.1,len(dt)),
         'BAA10Y':2.1+rng.normal(0,.1,len(dt)), 'VIXCLS':20.+rng.normal(0,.1,len(dt))},index=dt)

def test_macro_states_calendar_lags_and_no_imputation():
    r=_fred_panel();r.loc['1999-02-28','CPIAUCSL']=np.nan
    panel=make_equity_macro_panel(r)
    assert panel.states.loc['1999-02-28','CPI_INFLATION_YOY_PCT']!=panel.states.loc['1999-02-28','CPI_INFLATION_YOY_PCT']
    assert np.isnan(panel.states.loc['2000-02-29','CPI_INFLATION_YOY_PCT'])
    assert 'CHANGE_BAA_SPREAD_PP' in panel.changes
    assert panel.provenance['changes_are_identified_shocks'] is False

def test_download_fred_single_combined_mock_and_cache(tmp_path):
    r=_fred_panel(); f=r.reset_index().rename(columns={'index':'DATE'})
    content=f.to_csv(index=False).encode()
    class Response:
        def __init__(self,content):self.content=content
        def raise_for_status(self):pass
    class Session:
        def __init__(self):self.calls=0
        def get(self,*args,**kwargs):
            self.calls+=1;return Response(content)
    sess=Session()
    out=load_equity_fred_macro(cache_dir=tmp_path,session=sess,as_of='2005-12-31')
    assert sess.calls==1
    assert len(out.states.dropna())>120
    offline=load_equity_fred_macro(cache_dir=tmp_path,as_of='2005-12-31')
    pd.testing.assert_frame_equal(offline.states,out.states,check_freq=False)

def test_gamma_zero_recovers_original_pca(synthetic,ext):
    _,baseline=synthetic
    w1=baseline.weights.to_numpy()
    w2=ext.pricing_oriented_weights[[f'RP0_PC{i+1}' for i in range(3)]].to_numpy()
    assert np.allclose(w1,w2,atol=1e-10)
