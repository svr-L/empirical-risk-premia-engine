import io
import numpy as np
import pandas as pd
import pytest
from erp import data as d
from erp.rates import RatesData, RatesPremiaEngine


def original_mirror(start='1954-07-01', n=865):
    months=pd.date_range(start,periods=n,freq='MS')
    rows=['date,fed_funds_rate']
    for i,x in enumerate(months):
        rows.append(f'{x:%Y-%m-%d},{0.80 if i==0 else 2.0:.2f}')
    return ('\n'.join(rows)+'\n').encode()


def h15_preview():
    months=pd.date_range('2025-04-01','2026-10-01',freq='MS')
    rows=['<html><table><tr><th>Unique ID</th><th>Time Period</th><th>Value</th></tr>']
    for date in months:
        v='2.00' if date<=pd.Timestamp('2026-07-01') else '3.88'
        rows.append(f'<tr><td>H15/H15/RIFSPFF_N.M</td><td>{date:%Y-%m}</td><td>{v}</td></tr>')
    rows.append('</table></html>')
    return ''.join(rows).encode()


def test_mirror_exact_monthly_series():
    s=d.parse_fedfunds_mirror(original_mirror())
    assert len(s)==865 and s.index.is_month_end.all()
    assert s.iloc[0]==pytest.approx(.80) and s.loc['2026-07-31']==pytest.approx(2)


def test_mirror_rejects_wrong_schema_and_unit():
    with pytest.raises(ValueError,match='schema'):
        d.parse_fedfunds_mirror(original_mirror().replace(b'fed_funds_rate',b'sofr'))
    with pytest.raises(ValueError,match='wrong percent scale|wrong range|incomplete'):
        d.parse_fedfunds_mirror(original_mirror().replace(b',0.80',b',90.80'))


def test_h15_board_preview_parser_and_validated_augment():
    s=d.parse_board_h15_preview(h15_preview())
    assert len(s)==19 and s.index.max()==pd.Timestamp('2026-10-31')
    merged=d.supplement_fedfunds_with_board(d.parse_fedfunds_mirror(original_mirror()),h15_preview())
    assert len(merged)==868 and merged.loc['2026-10-31']==pytest.approx(3.88)


def test_board_overlap_rejection():
    html=h15_preview().replace(b'<td>2.00</td>',b'<td>6.00</td>')
    with pytest.raises(ValueError,match='disagree'):
        d.supplement_fedfunds_with_board(d.parse_fedfunds_mirror(original_mirror()),html)


def test_mirror_provider_no_fred_or_dbnomics_calls(monkeypatch,tmp_path):
    calls=[]
    def downloader(url,cache,**kw):
        calls.append(url)
        if 'eco3min.fr' in url:return original_mirror()
        if 'federalreserve.gov' in url:return h15_preview()
        raise AssertionError(f'unexpected endpoint {url}')
    monkeypatch.setattr(d,'_cached_download',downloader)
    result=d.load_fedfunds(cache_dir=tmp_path,source='mirror')
    assert result.index.max()==pd.Timestamp('2026-10-31')
    assert len(calls)==2 and not any('db.nomics.world' in x or 'fred.stlouisfed.org' in x for x in calls)


def test_unavailable_board_supplement_warns_without_stale_rate_imputation(monkeypatch,tmp_path):
    def downloader(url,cache,**kw):
        if 'eco3min' in url:return original_mirror()
        raise ConnectionError('H15 preview not reachable')
    monkeypatch.setattr(d,'_cached_download',downloader)
    with pytest.warns(RuntimeWarning,match="Retaining verified"):
        s=d.load_fedfunds(cache_dir=tmp_path,source='mirror')
    assert s.index.max()==pd.Timestamp('2026-07-31')


def test_local_original_csv_stays_first_priority(monkeypatch,tmp_path):
    csv=tmp_path/'funding.csv'
    csv.write_text('observation_date,FEDFUNDS\n'+'\n'.join(f'{t:%Y-%m-%d},3.75' for t in pd.date_range('2000-01-01',periods=240,freq='MS')))
    monkeypatch.setattr(d,'_cached_download',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('network contacted')))
    s=d.load_fedfunds(cache_dir=tmp_path,local_path=csv,source='mirror')
    assert len(s)==240 and s.iloc[-1]==pytest.approx(3.75)


def test_mirror_staleness_truncates_without_forward_fill():
    months=pd.date_range('1993-01-31','2026-10-31',freq='ME')
    months_ff=pd.date_range('1993-01-31','2026-07-31',freq='ME')
    rng=np.random.default_rng(33)
    changes=rng.normal(0,0.001,(len(months),10))
    z=pd.DataFrame(0.03+np.cumsum(changes,axis=0),index=months,columns=[f'{i}Y' for i in range(1,11)])
    ff=pd.Series(.025,index=months_ff)
    engine=RatesPremiaEngine(RatesData(z,ff))
    with pytest.warns(RuntimeWarning,match='truncating'):
        r=engine.fit(as_of='2026-10-31')
    assert r.as_of<=pd.Timestamp('2026-07-31')
