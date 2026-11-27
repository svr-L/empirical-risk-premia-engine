import numpy as np
import pandas as pd
from erp.equity_v013 import rp_basis,gx_three_pass,gx_bootstrap,analyze_dynamic_premia,_posterior,economic_return_decomposition
from erp.cross_asset import etf_factor_returns,ETF_TICKERS

def sample(n=126):
    rng=np.random.default_rng(13); ix=pd.date_range('2006-01-31',periods=n,freq='ME'); z=rng.normal(.003,.03,(n,4)); F=pd.DataFrame(z[:,:2],index=ix,columns=['F1','F2']);rp=pd.DataFrame(z[:,2:3],index=ix,columns=['RP12_PC1']); ex=pd.DataFrame(z[:,3:],index=ix,columns=['XA']);d=pd.DataFrame(z@rng.normal(size=(4,14))+rng.normal(0,.005,(n,14)),index=ix,columns=['d'+str(i) for i in range(14)]); y=pd.DataFrame(z@rng.normal(size=(4,8))+rng.normal(0,.005,(n,8)),index=ix,columns=['y'+str(i) for i in range(8)]);return F,rp,y,d,ex

def test_rp_exact_and_neutral():
    rng=np.random.default_rng(8);r=rng.normal(size=(90,10)); v,e=rp_basis(r,-1,k=3);assert np.allclose(e,np.linalg.eigvalsh(np.cov(r,rowvar=False,bias=True))[::-1][:3]); C=np.column_stack([np.ones(10),rng.normal(size=10)]);w,e=rp_basis(r,11,C,k=3);assert np.max(np.abs(C.T@w))<1e-10;assert np.allclose(w.T@w,np.eye(3))

def test_gx_rotation_and_scale():
    f,r,y,d,e=sample();g=pd.concat([f,r,e],axis=1);_,mu,r2,_=gx_three_pass(d,y,g,4); _,mu2,r22,_=gx_three_pass(d,y,g*3,4);assert np.allclose(mu*3,mu2);assert np.allclose(r2,r22); assert gx_bootstrap(d,y,g,k=4,draws=19).shape[0]==4

def test_posterior():
    m,v=_posterior(np.array([1.,2.]),np.diag([1.,4.]),np.ones(2)); assert np.allclose(m,[.5,.4]);assert np.allclose(np.diag(v),[.5,.8])

def test_dynamic_dates_and_no_leakage(tmp_path):
    f,r,y,d,e=sample(); kw=dict(extras=e,extra_traded=True,window=60,min_train=55,step=12,k=4,draws=19); a=analyze_dynamic_premia(f,r,y,d,**kw); assert (a.forecasts.formation<a.forecasts.date).all(); assert a.scores.n_months.nunique()==1; assert set(a.prices.estimator)>={'FMB','FMB_SHRINK','GMM','GX','DIRECT_MEAN','DIRECT_SHRINK'};assert np.isfinite(a.scores.mean_pricing_rmse_ann).all(); cutoff=a.forecasts.date.min();y.loc[cutoff:]*=10;b=analyze_dynamic_premia(f,r,y,d,**kw); aa=a.forecasts[a.forecasts.date==cutoff];bb=b.forecasts[b.forecasts.date==cutoff];np.testing.assert_allclose(aa.forecast,bb.forecast);a.export(tmp_path);assert (tmp_path/'manifest.json').exists()

def test_nontraded_no_direct_forecast():
    f,r,y,d,e=sample(); a=analyze_dynamic_premia(f,r,y,d,extras=e,window=60,min_train=55,step=12,k=4,draws=19);assert not ((a.prices.factor=='XA') & a.prices.estimator.str.startswith('DIRECT')).any();assert a.prices.loc[a.prices.factor=='XA','actual_return'].isna().all()

def test_traded_returns_no_fill():
    ix=pd.date_range('2020-01-31',periods=6,freq='ME');p=pd.DataFrame({c:np.arange(100,106.) for c in ETF_TICKERS},index=ix);p.loc[ix[2],'HYG']=np.nan;rf=pd.Series(.001,index=ix);out=etf_factor_returns(p,rf,as_of=str(ix[-1].date()));assert np.isclose(out.loc[ix[1],'XA_DURATION'],.009);assert pd.isna(out.loc[ix[2],'XA_CREDIT']) and pd.isna(out.loc[ix[3],'XA_CREDIT'])

def test_economic_contribution_identity():
    f,r,y,d,e=sample(); a=economic_return_decomposition(f,e,window=60,min_train=55);np.testing.assert_allclose(a.intercept_monthly+a.contribution_XA,a.total_expected_monthly);assert (a.formation<a.date).all()

def test_managed_lag_no_future():
    from erp.cross_asset import managed_factor_returns
    f,r,y,d,e=sample();f.columns=['MKT','HML'];s=pd.DataFrame({'CPI_INFLATION_YOY_PCT':np.arange(len(f)),'TERM_SPREAD_10Y2Y_PP':np.sin(np.arange(len(f)))},index=f.index);a=managed_factor_returns(f,s);s.iloc[90:]*=20;b=managed_factor_returns(f,s);pd.testing.assert_frame_equal(a.iloc[:92],b.iloc[:92])

def test_constrained_rolling_and_old_baseline():
    from test_equity_residual import make_panel
    from erp import analyze_equity_residuals,constrained_rp_rolling
    data,_=make_panel(seed=19);base=analyze_equity_residuals(data,train_end='2001-12-31',as_of='2010-12-31',n_pcs=3)
    a,diag,w=constrained_rp_rolling(data,base,k=2);assert len(a)==108;assert not a.isna().any().any();assert diag.max_constraint_error.max()<1e-8;assert (diag.formation<diag.holding_from).all();assert np.allclose(w.abs().sum(1),1)

def test_auditable_extra_hedges():
    f,r,y,d,e=sample();a=analyze_dynamic_premia(f,r,y,d,extras=e,extra_traded=True,window=60,min_train=55,step=12,k=4,draws=19)
    assert a.hedges.max_abs_training_correlation.max()<1e-10
    assert set(a.hedges.factor)=={'XA'}
