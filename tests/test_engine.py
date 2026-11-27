import numpy as np
import pandas as pd
import pytest
from erp import EquityData, EquityPremiaEngine, RatesData, RatesPremiaEngine
from erp.core import load_snapshot, export_snapshot
from erp.equity import MODELS, fmb_rolling, identification_diagnostics
from erp.rates import exact_funded_factor_weights, zc_return_decomposition
from erp.data import parse_french_monthly, parse_aqr_monthly_xlsx
from openpyxl import Workbook
from io import BytesIO

@pytest.fixture
def equity_fixture():
    rng=np.random.default_rng(812)
    dates=pd.date_range('2000-01-31',periods=170,freq='ME')
    base=rng.normal(.004,.035,(len(dates),9))
    ff3=pd.DataFrame(base[:,0:3],index=dates,columns=['MKT','SMB','HML']);ff3['RF']=.002
    ff5=pd.DataFrame(np.column_stack([base[:,0],base[:,1]+.002,base[:,2],base[:,3:5]]),index=dates,columns=['MKT','SMB','HML','RMW','CMA']);ff5['RF']=.002
    mom=pd.Series(base[:,5],index=dates,name='MOM')
    aqr=pd.DataFrame(base[:,6:8],index=dates,columns=['BAB','QMJ'])
    b=rng.normal(0,1,(30,8));b[:,0]+=1
    factor=np.column_stack([ff5[x].to_numpy() for x in ['MKT','SMB','HML','RMW','CMA']]+[mom.to_numpy(),aqr['BAB'].to_numpy(),aqr['QMJ'].to_numpy()])
    r=pd.DataFrame(factor@b.T+rng.normal(0,.02,(len(dates),30))+.002,index=dates,columns=[f'asset{i}' for i in range(30)])
    return EquityData(ff3=ff3,ff5=ff5,mom=mom,aqr=aqr,universes={'SIZE_BM_25':r.iloc[:,:25], 'INDUSTRY_30':r})


def test_family_smb_not_silently_merged(equity_fixture):
    f3,_=equity_fixture.factors('FF3'); f5,_=equity_fixture.factors('FF5')
    assert (f3['SMB']-f5['SMB']).abs().mean()>.001
    assert list(equity_fixture.factors('FF6_BAB_QMJ')[0].columns)==list(MODELS['FF6_BAB_QMJ'].factors)


def test_equity_v02_fit_as_of_fmb_and_exports(equity_fixture,tmp_path):
    eng=EquityPremiaEngine(equity_fixture)
    as_of=equity_fixture.ff5.index[-15]
    snap=eng.fit('FF6_BAB_QMJ','INDUSTRY_30',as_of=as_of,beta_window=36,run_fmb=True)
    assert snap.as_of<=as_of
    assert snap.fmb_lambdas.index.max()<=as_of
    assert set(snap.fmb_diagnostics.columns)>={'beta_std','partial_beta_std','cs_condition_scaled','t_hac'}
    assert snap.oos_attribution.index.get_level_values('Date').max()<=as_of
    assert set(snap.as_alpha_referee_payload())>= {'benchmark_residual_returns','factor_exposures'}
    exported=export_snapshot(snap,tmp_path/'snapshot')
    manifest,tables=load_snapshot(exported)
    assert manifest['point_in_time_note'].find('NOT historical vintages')>=0
    assert len(tables['factor_returns'])==len(snap.factors)
    assert 'BAB' in tables['betas'].columns
    assert (snap.betas.columns==snap.factors.columns).all()


def test_common_sample(equity_fixture):
    out=EquityPremiaEngine(equity_fixture).compare_models(['FF3','FF5','FF6','FF6_QMJ'],'INDUSTRY_30',beta_window=36)
    assert out['months'].nunique()==1
    assert out['sample_start'].nunique()==1 and out['sample_end'].nunique()==1


def test_no_future_changes_older_alpha(equity_fixture):
    cut=equity_fixture.ff5.index[-20]
    a=EquityPremiaEngine(equity_fixture).fit('FF6','INDUSTRY_30',as_of=cut,beta_window=36,run_fmb=True)
    sub=EquityData(equity_fixture.ff3.loc[:cut],equity_fixture.ff5.loc[:cut],equity_fixture.mom.loc[:cut],
                   {'INDUSTRY_30':equity_fixture.universes['INDUSTRY_30'].loc[:cut]},equity_fixture.aqr.loc[:cut])
    b=EquityPremiaEngine(sub).fit('FF6','INDUSTRY_30',beta_window=36,run_fmb=True)
    pd.testing.assert_frame_equal(a.fmb_lambdas,b.fmb_lambdas)


def test_identification_not_just_asset_count():
    g=np.random.default_rng(3)
    x=g.normal(0,.7,100)
    B=pd.DataFrame({'MKT':1+g.normal(0,.0001,100),'SMB':x,'HML':x+g.normal(0,.00001,100)})
    diag=identification_diagnostics(B)
    assert diag.loc['MKT','beta_std']<.001
    assert diag.loc['HML','partial_beta_std']<.001
    assert diag.loc['HML','auxiliary_r2']>.999


def test_funded_constraints_match_original_math():
    rng=np.random.default_rng(7)
    B=pd.DataFrame(rng.normal(0,.2,(9,3)),index=[f'ZC{i}Y' for i in range(2,11)],columns=['Level','Slope','Curvature'])
    xx=rng.normal(0,1,(9,9));H=pd.DataFrame(xx.T@xx+np.eye(9),index=B.index,columns=B.index)
    for f in B.columns:
        w,d=exact_funded_factor_weights(B,H,f)
        x=B.divide(np.sqrt((B**2).mean(axis=0)),axis=1).T@w
        assert x.drop(f).abs().max()<1e-10
        assert abs(w.sum()+d['cash_weight'])<1e-12
        assert abs(w.abs().sum()-1)<1e-12


def test_rates_static_pca_funded_pipeline(tmp_path):
    rng=np.random.default_rng(77)
    dates=pd.date_range('1990-01-31',periods=210,freq='ME')
    load=np.column_stack([np.ones(10),np.linspace(-1,1,10),1-2*np.linspace(-1,1,10)**2])
    dy=rng.normal(0,.001,(210,3))@load.T
    z=.03+np.linspace(0,.01,10)+np.cumsum(dy,axis=0)
    curve=pd.DataFrame(z,index=dates,columns=[f'{i}Y' for i in range(1,11)])
    data=RatesData(curve,pd.Series(.025,index=dates))
    e=RatesPremiaEngine(data)
    snap=e.fit(pca_train_end='1998-12-31',warmup=36)
    assert len(snap.factors)>20
    assert snap.metadata['mimic_max_cross_violation']<1e-7
    assert set(snap.betas)=={'Level','Slope','Curvature'}
    assert np.allclose(e.latest_mimic_diagnostics['cash_weight']+e.latest_mimic_diagnostics['risky_net_weight'],0)
    export_snapshot(snap,tmp_path/'rates')
    e.export_auxiliary(tmp_path/'rates')
    assert (tmp_path/'rates'/'rates_latest_funded_weights.csv').exists()


def test_french_parser_and_aqr_monthly_xlsx():
    text='Some preamble\n, Mkt-RF,SMB,HML,RF\n200001,1.0,2.0,3.0,0.2\n200002,2.0,-99.99,4.0,0.3\nAnnual Factors: below\n2000,1,2,3,4'
    f=parse_french_monthly(text)
    assert f.iloc[0]['MKT']==pytest.approx(.01)
    assert pd.isna(f.iloc[1]['SMB'])
    wb=Workbook();ws=wb.active;ws.title='BAB Factors'
    for i in range(18):ws.append(['note'])
    ws.append(['DATE','USA','UK'])
    ws.append(['1/31/2020',.02,.01]);ws.append(['2/29/2020',-.03,.02])
    buff=BytesIO();wb.save(buff)
    b=parse_aqr_monthly_xlsx(buff.getvalue(),'BAB')
    assert len(b)==2 and b.iloc[0]==pytest.approx(.02)
