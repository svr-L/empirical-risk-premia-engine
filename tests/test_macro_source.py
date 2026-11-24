from io import BytesIO
from pathlib import Path
import zipfile
import pandas as pd
import pytest
import requests
from erp.macro import load_macro_panel, load_macro_series_files, _from_bytes


def original_series_csv(start='2003-01-01', end='2026-10-01'):
    dates=pd.date_range(start,end,freq='MS')
    n=len(dates)
    # Genuine-format **synthetic test fixtures**, never bundled as market observations.
    return pd.DataFrame({'observation_date':dates.strftime('%Y-%m-%d'),
                         'CPIAUCSL':[100*(1.002)**i for i in range(n)],
                         'INDPRO':[80*(1.001)**i for i in range(n)],
                         'T10YIE':[2.0+0.1*(i%3) for i in range(n)]})


def test_wide_fred_csv_and_units(tmp_path):
    p=tmp_path/'orig.csv'
    original_series_csv().to_csv(p,index=False)
    out=load_macro_panel(source='local',path=p,cache_dir=tmp_path/'cache',as_of='2026-08-31')
    assert list(out.columns)==['CPI','INDPRO','BREAKEVEN10Y']
    assert out.index.max()==pd.Timestamp('2026-08-31')
    assert out.loc['2026-08-31','BREAKEVEN10Y']>1
    cached=load_macro_panel(source='auto',cache_dir=tmp_path/'cache',as_of='2026-10-31')
    assert cached.index.max()==pd.Timestamp('2026-10-31')


def test_fred_single_zip_and_daily_last(tmp_path):
    p=tmp_path/'fred.zip'
    wide=original_series_csv()
    daily=wide.copy()
    extra=daily.tail(1).copy()
    extra['observation_date']='2026-10-30'
    extra['T10YIE']=2.98
    daily=pd.concat([daily,extra],ignore_index=True)
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('README.txt','FRED data')
        z.writestr('data/fredgraph.csv',daily.to_csv(index=False))
    out=load_macro_panel(source='local',path=p,cache_dir=tmp_path/'cache',as_of='2026-10-31')
    assert out.loc['2026-10-31','BREAKEVEN10Y']==2.98


def test_three_series_zip(tmp_path):
    frame=original_series_csv()
    p=tmp_path/'multiple.zip'
    with zipfile.ZipFile(p,'w') as z:
        for col in ['CPIAUCSL','INDPRO','T10YIE']:
            z.writestr(f'{col}.csv',frame[['observation_date',col]].to_csv(index=False))
    out=load_macro_panel(source='local',path=p,cache_dir=tmp_path/'cache')
    assert len(out.dropna())>200
    assert set(out)=={'CPI','INDPRO','BREAKEVEN10Y'}


def test_three_separate_files(tmp_path):
    df=original_series_csv()
    paths={}
    for sid in ['CPIAUCSL','INDPRO','T10YIE']:
        q=tmp_path/f'{sid}.csv'
        df[['observation_date',sid]].to_csv(q,index=False)
        paths[sid]=q
    out=load_macro_series_files(paths)
    assert out.index.min()==pd.Timestamp('2003-01-31')


def test_missing_series_is_a_hard_error(tmp_path):
    df=original_series_csv().drop(columns=['T10YIE'])
    p=tmp_path/'bad.csv';df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Missing original macro series'):
        load_macro_panel(source='local',path=p,cache_dir=tmp_path/'cache')


def test_fred_failure_one_request_clear_error(tmp_path):
    class Failing:
        calls=0
        def get(self,*args,**kwargs):
            self.calls+=1
            raise requests.exceptions.ReadTimeout('blocked from Colab')
    sess=Failing()
    with pytest.raises(ConnectionError,match='Download the original CSV or ZIP'):
        load_macro_panel(source='auto',cache_dir=tmp_path/'cache',session=sess)
    assert sess.calls==1


def test_gaps_are_not_forward_filled(tmp_path):
    df=original_series_csv()
    df.loc[df['observation_date']=='2025-06-01','INDPRO']=None
    p=tmp_path/'gap.csv';df.to_csv(p,index=False)
    out=load_macro_panel(source='local',path=p,cache_dir=tmp_path/'cache')
    assert pd.isna(out.loc['2025-06-30','INDPRO'])
    assert out.loc['2025-07-31','INDPRO']>0
