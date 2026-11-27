"""Acquire exactly the three historical macro-level inputs used by Rates v0.8.

No alternative macro proxies, no imputation of missing observations, no forward fill.
The latest revised observations are NOT point-in-time historical release vintages.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile
import warnings
import pandas as pd
import requests

IDS = {'CPIAUCSL':'CPI', 'INDPRO':'INDPRO', 'T10YIE':'BREAKEVEN10Y'}
CANON = ['CPI','INDPRO','BREAKEVEN10Y']
FRED_MULTI_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL%2CINDPRO%2CT10YIE'


def _datecol(cols):
    return next((c for c in cols if str(c).strip().lower() in {'date','observation_date','observation date'}), None)


def _parse_macro_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse wide raw FRED series or canonical CSV, preserving native units.

    FRED CPIAUCSL/INDPRO: raw index level; T10YIE: percent per annum.
    T10YIE is daily and is aligned via monthly last observed value, as in v0.8.
    """
    d=df.copy()
    d.columns=[str(x).strip().lstrip('\ufeff') for x in d.columns]
    dc=_datecol(d.columns)
    if dc is None:
        raise ValueError('Missing Date/observation_date column; expected original FRED wide CSV.')
    d.index=pd.to_datetime(d.pop(dc), errors='coerce')
    if d.index.isna().any(): raise ValueError('Macro input contains unparseable dates')
    names={**IDS, **{v:v for v in IDS.values()}}
    d=d.rename(columns={k:v for k,v in names.items() if k in d.columns})
    d=d.loc[:,~d.columns.duplicated(keep='first')]
    found=[k for k in CANON if k in d.columns]
    if not found: raise ValueError('No CPIAUCSL/INDPRO/T10YIE columns found')
    d=d[found].apply(pd.to_numeric, errors='coerce').sort_index()
    if (d.index.duplicated()).any(): d=d.groupby(level=0).last()
    # Use daily/monthly last observation, not interpolation or forward filling.
    d=d.resample('ME').last()
    return d


def _from_bytes(content: bytes) -> pd.DataFrame:
    if not content: raise ValueError('Macro download was empty')
    frames=[]
    if zipfile.is_zipfile(BytesIO(content)):
        with zipfile.ZipFile(BytesIO(content)) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.endswith('/')]
            if not names: raise ValueError('FRED ZIP contains no CSV files')
            for name in names:
                f=pd.read_csv(BytesIO(z.read(name)), low_memory=False)
                try:
                    frames.append(_parse_macro_frame(f))
                except ValueError:
                    # FRED bundles metadata/readme datasets: skip only non-series tables.
                    continue
    else:
        frames=[_parse_macro_frame(pd.read_csv(BytesIO(content),low_memory=False))]
    if not frames: raise ValueError('Macro ZIP did not contain a usable FRED series table')
    frame=frames[0]
    for later in frames[1:]:
        for col in later.columns:
            if col not in frame.columns: frame=frame.join(later[[col]],how='outer')
    frame=frame.sort_index()
    return frame


def _normalize(panel: pd.DataFrame, *, require_all=True, as_of=None) -> pd.DataFrame:
    have=[c for c in CANON if c in panel.columns]
    missing=set(CANON)-set(have)
    if missing and require_all: raise ValueError(f'Missing original macro series: {sorted(missing)}')
    d=panel[have].copy().sort_index()
    if as_of is not None: d=d.loc[:pd.Timestamp(as_of)]
    if d.empty:raise ValueError('Macro data empty after as_of cutoff')
    if 'CPI' in d and (d['CPI'].dropna()<=0).any():raise ValueError('CPI level must be positive')
    if 'INDPRO' in d and (d['INDPRO'].dropna()<=0).any():raise ValueError('INDPRO level must be positive')
    if 'BREAKEVEN10Y' in d:
        # 0.01 is 0.01%, not 1%: protects against accidental decimal input.
        b=d['BREAKEVEN10Y'].dropna()
        if not b.empty and (b.abs()>25).any():raise ValueError('T10YIE expected percentage points (e.g. 2.3, not 230)')
    if require_all:
        overlap=d[CANON].dropna()
        if len(overlap)<36:
            raise ValueError(f'Only {len(overlap)} complete macro months; expected >= 36 for v0.8 STATE model')
        if overlap.index.max()<pd.Timestamp(as_of or overlap.index.max())-pd.DateOffset(months=6):
            warnings.warn(f'Macro panel ends at {overlap.index.max().date()}; later STATE forecasts unavailable. No macro values have been forward-filled.',RuntimeWarning)
    return d


def load_macro_panel(*, cache_dir='erp_data_cache', source='auto', path=None,
                     as_of=None, timeout=20, refresh=False, session=None) -> pd.DataFrame:
    """Load genuine CPIAUCSL/INDPRO/T10YIE from local CSV/ZIP or FRED.

    source: 'auto' (prefer cache; one remote attempt), 'fred' (remote), 'local'.
    FRED multi-series URL may return CSV or ZIP; both are parsed. No proxy fallback.
    """
    if source not in ('auto','fred','local'):raise ValueError('source must be auto, fred or local')
    dest=Path(cache_dir); dest.mkdir(parents=True, exist_ok=True)
    normalized=dest/'macro_CPIAUCSL_INDPRO_T10YIE.csv'
    if path is not None:
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(path)
        raw=_from_bytes(path.read_bytes())
        out=_normalize(raw,as_of=as_of)
        # Persist untruncated data so a later rerun can change as_of.
        _normalize(raw).to_csv(normalized,index_label='Date')
        print(f'Macro source: local {path.name}; CPIAUCSL, INDPRO, T10YIE in original units.')
        return out
    if source!='fred' and not refresh and normalized.is_file():
        raw=_from_bytes(normalized.read_bytes())
        print(f'Macro source: cached {normalized}')
        return _normalize(raw,as_of=as_of)
    if source=='local': raise FileNotFoundError(f'Macro CSV/ZIP not found; supply path, or place {normalized}')
    getter=requests.get if session is None else session.get
    try:
        resp=getter(FRED_MULTI_URL,timeout=(6,timeout),headers={'User-Agent':'EmpiricalRiskPremiaEngine/0.2.5 (research)'})
        resp.raise_for_status()
        raw=_from_bytes(resp.content)
        out=_normalize(raw,as_of=as_of)
    except (requests.RequestException,ValueError,zipfile.BadZipFile) as e:
        raise ConnectionError('CPIAUCSL/INDPRO/T10YIE could not be downloaded from FRED. '
           'Download the original CSV or ZIP using your browser and specify MACRO_CSV_PATH. '
           'No alternative macro variables or forward-filled values will be substituted. '
           f'Underlying error: {e}') from e
    _normalize(raw).to_csv(normalized,index_label='Date')
    print(f'Macro source: FRED original series; cached {normalized}.')
    return out


def load_macro_series_files(paths: dict, *, as_of=None) -> pd.DataFrame:
    """Load three original individual FRED exports by ID, one CSV per series."""
    out=pd.DataFrame()
    for raw_id,col in IDS.items():
        if raw_id not in paths:raise ValueError(f'Missing {raw_id} CSV in supplied paths')
        parsed=_from_bytes(Path(paths[raw_id]).read_bytes())
        if col not in parsed: raise ValueError(f'{paths[raw_id]} does not contain {raw_id}')
        out=parsed[[col]] if out.empty else out.join(parsed[[col]],how='outer')
    return _normalize(out,as_of=as_of)
