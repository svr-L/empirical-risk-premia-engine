"""FRED macro/cross-asset panel for Equity risk-factor interpretation.

Current-revised public series (NOT point-in-time vintages). Level state predictors
and contemporaneous observed macro *changes* are separate objects. Changes are
NOT identified macroeconomic announcement surprises.
"""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import requests

FRED_IDS=('CPIAUCSL','INDPRO','DGS10','DGS2','BAA10Y','VIXCLS')
PUBLIC_GRAPH='https://fred.stlouisfed.org/graph/fredgraph.csv?id='

@dataclass
class EquityMacroPanel:
    raw_monthly:pd.DataFrame
    states:pd.DataFrame
    changes:pd.DataFrame
    provenance:dict

    def export(self,folder):
        p=Path(folder);p.mkdir(parents=True,exist_ok=True)
        self.raw_monthly.to_csv(p/'fred_raw_levels.csv',index_label='DATE')
        self.states.to_csv(p/'equity_macro_states.csv',index_label='DATE')
        self.changes.to_csv(p/'equity_economic_changes_not_shocks.csv',index_label='DATE')
        return p


def _parse_csv_bytes(content):
    if zipfile.is_zipfile(BytesIO(content)):
        with zipfile.ZipFile(BytesIO(content)) as z:
            tabs=[pd.read_csv(BytesIO(z.read(n))) for n in z.namelist() if n.lower().endswith('.csv')]
            if not tabs: raise ValueError('ZIP has no CSV')
            # Multiple CSV files joined on dates.
            frames=[_parse_frame(df) for df in tabs]
            out=frames[0]
            for f in frames[1:]:out=out.combine_first(f)
            return out
    return _parse_frame(pd.read_csv(BytesIO(content)))


def _parse_frame(df):
    frame=df.copy()
    frame.columns=[str(c).strip().lstrip('\ufeff') for c in frame.columns]
    dcol=next((c for c in frame.columns if c.lower() in ('date','observation_date','observation date')),None)
    if dcol is None:raise ValueError('FRED table needs DATE or observation_date')
    date=pd.to_datetime(frame.pop(dcol),errors='coerce')
    if date.isna().any():raise ValueError('Unparseable macro dates')
    frame.index=pd.DatetimeIndex(date)
    frame=frame[[c for c in frame.columns if c in FRED_IDS]]
    if not len(frame.columns):raise ValueError('No recognized FRED series')
    frame=frame.apply(pd.to_numeric,errors='coerce').sort_index()
    if frame.index.has_duplicates:frame=frame.groupby(level=0).last()
    return frame.resample('ME').last()


def _combine(frames):
    out=frames[0]
    for f in frames[1:]:out=out.combine_first(f)
    return out.sort_index()


def load_equity_fred_macro(*, cache_dir='erp_equity_macro_cache', path=None,
        as_of='2026-10-31', refresh=False, timeout=16, session=None):
    """Download six public FRED series, or load a previously exported FRED CSV/ZIP.

    A combined graph CSV is requested first; if that fails, each missing series
    is requested individually. Store the raw panel and allow later offline runs.
    Never forward-fill missing observations or invent proxy series.
    """
    dest=Path(cache_dir);dest.mkdir(parents=True,exist_ok=True)
    cache=dest/'fred_equity_macro_raw.csv'
    provenance={'provider':'FRED graph CSV','ids':list(FRED_IDS),
                'historical_revision':'latest available vintage, NOT point-in-time',
                'missing_policy':'no fill, no replacement', 'urls':[]}
    if path is not None:
        raw=_parse_csv_bytes(Path(path).read_bytes())
        provenance['source']='user supplied FRED CSV/ZIP'
    elif cache.exists() and not refresh:
        raw=_parse_csv_bytes(cache.read_bytes())
        provenance['source']='cached official FRED CSV'
    else:
        get=requests.get if session is None else session.get
        ids=','.join(FRED_IDS)
        url=PUBLIC_GRAPH+ids
        frames=[]
        errs=[]
        try:
            response=get(url,timeout=(5,timeout),headers={'User-Agent':'Mozilla/5.0 research macro downloader'})
            response.raise_for_status()
            frames.append(_parse_csv_bytes(response.content))
            provenance['urls'].append(url)
        except (requests.RequestException,ValueError,zipfile.BadZipFile) as e:
            errs.append(f'combined: {str(e)[:160]}')
        missing=set(FRED_IDS)-set(_combine(frames).columns if frames else [])
        for code in sorted(missing):
            oneurl=PUBLIC_GRAPH+code
            try:
                response=get(oneurl,timeout=(5,timeout),headers={'User-Agent':'Mozilla/5.0 research macro downloader'})
                response.raise_for_status()
                frames.append(_parse_csv_bytes(response.content))
                provenance['urls'].append(oneurl)
            except (requests.RequestException,ValueError,zipfile.BadZipFile) as e:
                errs.append(f'{code}: {str(e)[:160]}')
        if not frames:
            raise ConnectionError('FRED downloads unavailable. Open '
                +url+' in your browser, save the CSV/ZIP, and provide path=... '
                +'or use the cached raw FRED CSV. Details: '+'; '.join(errs))
        raw=_combine(frames)
        provenance['source']='live public FRED data'
        provenance['download_errors']=errs
    absent=set(FRED_IDS)-set(raw.columns)
    if absent:raise ValueError('Missing required FRED identifiers: '+', '.join(sorted(absent))+
        '. Upload one wide original FRED CSV/ZIP with all six series or restore the cache.')
    raw=raw[list(FRED_IDS)].sort_index()
    raw.to_csv(cache,index_label='DATE')
    if as_of is not None: raw=raw.loc[:pd.Timestamp(as_of)]
    if raw.empty:raise ValueError('Macro data empty after as_of cutoff')
    return make_equity_macro_panel(raw,provenance)


def make_equity_macro_panel(raw,provenance=None):
    """Deterministic state construction (same month no fill); changes NOT shocks."""
    if not all(x in raw.columns for x in FRED_IDS):raise ValueError('All six original series required')
    r=raw[list(FRED_IDS)].copy().astype(float).sort_index()
    if not isinstance(r.index,pd.DatetimeIndex):raise ValueError('datetime monthly index required')
    r.index=r.index+pd.offsets.MonthEnd(0)
    if r.index.has_duplicates:raise ValueError('duplicate macro month')
    if ((r['CPIAUCSL'].dropna()<=0).any() or (r['INDPRO'].dropna()<=0).any()
            or (r['VIXCLS'].dropna()<=0).any()):raise ValueError('CPI, IP and VIX must be positive')
    # Monthly calendar, not row shifts across missing calendar months.
    r=r.reindex(pd.date_range(r.index.min(),r.index.max(),freq='ME'))
    states=pd.DataFrame(index=r.index)
    states['CPI_INFLATION_YOY_PCT']=100*np.log(r['CPIAUCSL']/r['CPIAUCSL'].shift(12))
    states['INDPRO_GROWTH_YOY_PCT']=100*np.log(r['INDPRO']/r['INDPRO'].shift(12))
    states['TERM_SPREAD_10Y2Y_PP']=r['DGS10']-r['DGS2']
    states['BAA_TREASURY_SPREAD_PP']=r['BAA10Y']
    states['LOG_VIX']=np.log(r['VIXCLS'])
    changes=pd.DataFrame(index=r.index)
    changes['CHANGE_INFLATION_YOY_PP']=states['CPI_INFLATION_YOY_PCT'].diff()
    changes['CHANGE_IP_GROWTH_YOY_PP']=states['INDPRO_GROWTH_YOY_PCT'].diff()
    changes['CHANGE_TREASURY_10Y_PP']=r['DGS10'].diff()
    changes['CHANGE_BAA_SPREAD_PP']=r['BAA10Y'].diff()
    changes['CHANGE_LOG_VIX']=states['LOG_VIX'].diff()
    # Keep missing values as NA; downstream complete-case intersection is explicit.
    meta=dict(provenance or {})
    meta['state_units']='CPI/IP log YoY percentage points; spreads percentage points; log VIX'
    meta['changes_definition']='first monthly differences of states / 10Y yield'
    meta['changes_are_identified_shocks']=False
    meta['point_in_time_vintages']=False
    meta['publication_lag']='apply >=2 calendar months to state levels when forecasting'
    meta['as_of']=str(r.index.max().date())
    meta['first_complete_state_month']=str(states.dropna().index.min().date()) if len(states.dropna()) else None
    return EquityMacroPanel(r,states,changes,meta)
