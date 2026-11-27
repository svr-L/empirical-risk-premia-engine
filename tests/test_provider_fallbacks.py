import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from erp.data import parse_kw_board_csv, load_kw_zero_curve, parse_fred_text, fred_csv_series
from erp.rates import RatesData


def board_bytes(n=100):
    header = 'DATE,'+','.join(f'THREEFY{i:02d}00.B' for i in range(1,11))
    dates=pd.bdate_range('2023-01-02',periods=n)
    rows=['# provenance: 3 factor KW rates',header]
    for k,dt in enumerate(dates):
        rows.append(dt.strftime('%d-%m-%Y')+','+','.join(f'{2.5+.1*i+.001*k:.4f}' for i in range(1,11)))
    return ('\n'.join(rows)).encode()


def test_board_parser_full_grid():
    x=parse_kw_board_csv(board_bytes())
    assert x.shape==(100,10)
    assert list(x.columns)==[f'{i}Y' for i in range(1,11)]
    assert x.index[0]==pd.Timestamp('2023-01-02')
    assert x['1Y'].iloc[0]==pytest.approx(2.6)


def test_board_csv_local_zero_yields(tmp_path):
    path=tmp_path/'feds200533.csv';path.write_bytes(board_bytes())
    z=load_kw_zero_curve(local_path=path)
    assert z.shape[1]==10
    assert z.index.is_month_end.all()
    assert (z.max()<10).all()


def test_rates_explicit_original_local_sources(tmp_path):
    curve=tmp_path/'curve.csv';curve.write_bytes(board_bytes())
    daily=pd.bdate_range('2022-01-03',periods=330)
    ff=tmp_path/'funding.csv'
    ff.write_text('observation_date,FEDFUNDS\n'+'\n'.join(d.strftime('%Y-%m-%d')+',2.5' for d in daily))
    output=RatesData.from_fred(cache_dir=tmp_path,local_curve_path=curve,local_fedfunds_path=ff)
    assert output.zero_curve.shape[1]==10
    assert output.zero_curve.max().max()<.1
    assert output.fedfunds.dropna().iloc[-1]==pytest.approx(.025)


def test_text_fred_source_exact_same_series():
    text='TITLE: FEDFUNDS\nDATE VALUE\n'+'\n'.join(f'2020-{m:02d}-01 1.25' for m in range(1,13))
    s=parse_fred_text(text,'FEDFUNDS')
    assert len(s)==12 and s.name=='FEDFUNDS'


def test_board_source_is_one_request(monkeypatch,tmp_path):
    from erp import data as d
    calls=[]
    def downloader(url, cache, **kw):
        calls.append(url)
        return board_bytes()
    monkeypatch.setattr(d,'_cached_download',downloader)
    z=d.load_kw_zero_curve(cache_dir=tmp_path,source='board')
    assert len(calls)==1 and 'feds200533.csv' in calls[0]
    assert z.shape[1]==10


def test_board_source_auto_fred_fallback(monkeypatch,tmp_path):
    from erp import data as d
    def downloader(url, cache, **kw):
        raise ConnectionError('temporary Board timeout')
    times=pd.bdate_range('2021-01-01',periods=300)
    def orig_fred(name, cache_dir, refresh=False):
        return pd.Series(2.0 + int(name.replace('THREEFY',''))*.05, index=times,name=name)
    monkeypatch.setattr(d,'_cached_download',downloader)
    monkeypatch.setattr(d,'fred_csv_series',orig_fred)
    with pytest.warns(RuntimeWarning):
        z=d.load_kw_zero_curve(cache_dir=tmp_path,source='auto')
    assert list(z.columns)==[f'{i}Y' for i in range(1,11)]
    assert len(z)>=12

def test_board_day_first_explicit():
    x=parse_kw_board_csv(board_bytes())
    assert x.loc[pd.Timestamp('2023-01-03'),'1Y']>0
