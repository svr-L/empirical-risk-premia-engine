import numpy as np
import pandas as pd
import pytest
from erp.equity_shock_followup import analyze_rolling_shocks

def panel():
    rng=np.random.default_rng(22); dates=pd.date_range('2005-12-31',periods=180,freq='ME')
    s=pd.DataFrame(rng.normal(size=(180,2)),index=dates,columns=['MP_median','CBI_median'])
    f=pd.DataFrame(rng.normal(0,.025,(180,3)),index=dates,columns=['MKT','HML','MOM'])
    r=pd.DataFrame(.005*s.to_numpy()+rng.normal(0,.01,(180,2)),index=dates,columns=['RP0_PC1','RP12_PC2']).iloc[1:]
    y=pd.DataFrame(np.column_stack([f,s])@rng.normal(0,.015,(5,30))+rng.normal(0,.02,(180,30)),index=dates,columns=[f'a{i}' for i in range(30)])
    s.iloc[25]=np.nan
    return f,r,y,s

def test_rolling_mapping_pricing_common_calendar(tmp_path):
    f,r,y,s=panel();out=analyze_rolling_shocks(f,r,y,s,train_end='2011-12-31')
    assert len(out.mapping)==12 # 5*2 + two FF-controlled RP rows
    assert len(out.pricing)==12
    assert set(out.mapping.factor)==set(f)|set(r)
    assert out.mapping.n_train.nunique()==out.pricing.n_train.nunique()==1
    assert out.mapping.n_train.iloc[0]==70 # 72 months minus missing current and next
    assert not out.calendar.loc[out.calendar.date.isin([s.index[25],s.index[26]]),'included'].any()
    assert np.isfinite(out.pricing.oos_mean_pricing_rmse_ann).all()
    assert np.allclose(out.premium_mapping.shock_only_sdf_implied_ann,out.premium_mapping.contribution_MP_median_ann+out.premium_mapping.contribution_CBI_median_ann)
    out.export(tmp_path);assert (tmp_path/'calendar.csv').exists()

def test_future_mutation_does_not_change_formation_estimates():
    f,r,y,s=panel();a=analyze_rolling_shocks(f,r,y,s)
    for x in (f,r,y,s):x.loc['2012':]*=2
    b=analyze_rolling_shocks(f,r,y,s)
    pd.testing.assert_frame_equal(a.lambdas,b.lambdas)
    np.testing.assert_allclose(a.premium_mapping.shock_only_sdf_implied_ann,b.premium_mapping.shock_only_sdf_implied_ann)
    np.testing.assert_allclose(a.mapping.beta_MP_median,b.mapping.beta_MP_median)

def test_no_backfill_before_rolling_start():
    f,r,y,s=panel()
    with pytest.raises(ValueError,match='insufficient'):analyze_rolling_shocks(f,r,y,s,train_end='2005-12-31')

def test_fred_autodiscovery_ignores_wrong_zip(tmp_path):
    from erp.equity_macro import FRED_IDS,find_complete_equity_fred_file
    import zipfile,os
    frame=pd.DataFrame({'DATE':['2020-01-31'],**{c:[1.] for c in FRED_IDS}})
    good=tmp_path/'fredgraph (5).zip'
    with zipfile.ZipFile(good,'w') as z:z.writestr('fredgraph.csv',frame.to_csv(index=False))
    bad=tmp_path/'fredgraph (6).csv';frame[['DATE','CPIAUCSL']].to_csv(bad,index=False)
    os.utime(bad,(good.stat().st_mtime+10,good.stat().st_mtime+10))
    path,rejected=find_complete_equity_fred_file([tmp_path]);assert path==good and len(rejected)==1
    with pytest.raises(ValueError,match='Missing series'):find_complete_equity_fred_file([tmp_path],explicit=bad)
