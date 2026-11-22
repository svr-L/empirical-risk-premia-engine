"""Data providers: published Fama-French monthly returns and published AQR BAB/QMJ monthly returns.
No ETF proxies; no silent replacement; no silent factor scaling.
"""
from __future__ import annotations
import io
import re
import zipfile
import csv
import time
import warnings
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import requests

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRENCH_FILES = {
    "FF3": "F-F_Research_Data_Factors_CSV.zip",
    "FF5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "MOM": "F-F_Momentum_Factor_CSV.zip",
    "SIZE_BM_25": "25_Portfolios_5x5_CSV.zip",
    "INDUSTRY_30": "30_Industry_Portfolios_CSV.zip",
    "SIZE_OP_25": "25_Portfolios_ME_OP_5x5_CSV.zip",
    "SIZE_INV_25": "25_Portfolios_ME_INV_5x5_CSV.zip",
    "SIZE_MOM_25": "25_Portfolios_ME_Prior_12_2_CSV.zip",
}
AQR_URLS = {
    "BAB": "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
    "QMJ": "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx",
}
AQR_SHEETS = {"BAB": "BAB Factors", "QMJ": "QMJ Factors"}
MONTHLY_ROW = re.compile(r"^\s*\d{6}\s*,")


def _cached_download(url: str, cache: Path, timeout: int = 25, refresh: bool = False, retries: int = 2, allow_html: bool = False) -> bytes:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return cache.read_bytes()
    last_exception = None
    for attempt in range(retries + 1):
        try:
            # Separate connection and read timeouts to avoid indefinite Colab stalls.
            r = requests.get(url, timeout=(8, timeout),
                headers={"User-Agent": "EmpiricalRiskPremiaEngine/0.2.3 (research; data download)"})
            r.raise_for_status()
            break
        except requests.RequestException as ex:
            last_exception = ex
            if attempt < retries:
                time.sleep(min(2 * (attempt + 1), 4))
    else:
        if cache.exists():
            warnings.warn(f"{url}: refresh failed, using previously cached {cache}", RuntimeWarning)
            return cache.read_bytes()
        raise ConnectionError(
            f"Data provider unavailable after {retries+1} attempts: {url}. "
            f"Download the original CSV into {cache} or use the official Board source. "
            f"Root error: {last_exception}"
        ) from last_exception
    content = r.content
    if len(content) < 100 or (content[:1] == b"<" and not allow_html):
        raise ValueError(f"Vendor returned unexpected content instead of factor dataset: {url}")
    cache.write_bytes(content)
    return content


def parse_french_monthly(text: str, anchor: str | None = None) -> pd.DataFrame:
    """Monthly value-weighted first block, percent to decimal, preserving missing markers.

    Extracts only the first contiguous YYYYMM block after optional section anchor.
    """
    lines = text.splitlines()
    start = 0
    if anchor:
        candidates = [i for i, line in enumerate(lines) if anchor.lower() in line.lower()]
        if not candidates:
            raise ValueError(f"French source missing expected section {anchor!r}")
        start = candidates[0] + 1
    first = next((i for i in range(start, len(lines)) if MONTHLY_ROW.match(lines[i])), None)
    if first is None:
        raise ValueError("French source contains no YYYYMM monthly data")
    h = next((i for i in range(first - 1, start - 2, -1) if ',' in lines[i] and lines[i].strip()), None)
    if h is None:
        raise ValueError("French CSV monthly header not found")
    rows = []
    i = first
    while i < len(lines) and MONTHLY_ROW.match(lines[i]):
        rows.append(lines[i]); i += 1
    df = pd.read_csv(io.StringIO("\n".join([lines[h], *rows])))
    df.columns = [str(c).strip().replace("Mkt-RF", "MKT").replace("Mkt_RF", "MKT") for c in df.columns]
    col = df.columns[0]
    raw_dates = df.pop(col).astype(str).str.strip()
    df.index = pd.to_datetime(raw_dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= -99.0) / 100.0
    if df.index.has_duplicates:
        raise ValueError("Duplicate monthly French observation")
    return df.sort_index()


def load_french_monthly(key: str, cache_dir: str | Path = "data/cache", refresh: bool = False) -> pd.DataFrame:
    filename = FRENCH_FILES[key]
    cache = Path(cache_dir) / filename
    content = _cached_download(f"{FRENCH_BASE}/{filename}", cache, refresh=refresh)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [x for x in z.namelist() if x.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"French archive {filename} contains no CSV")
        text = z.read(names[0]).decode("utf-8", errors="replace")
    anchor = "Average Value Weighted Returns -- Monthly" if "_25" in key or "INDUSTRY" in key else None
    return parse_french_monthly(text, anchor=anchor)


def parse_aqr_monthly_xlsx(source: bytes | str | Path, factor: str,
                           units: Literal["decimal", "percent"] = "decimal") -> pd.Series:
    """Read official monthly workbook's USA column (header at row 19).

    Published factor returns must be monthly long/short factor returns, not ETF proxies.
    `units` must be confirmed against AQR workbook notes. Default AQR is decimal.
    """
    import openpyxl  # Excel dependency loaded only here
    src = io.BytesIO(source) if isinstance(source, bytes) else source
    xls = pd.ExcelFile(src)
    sheet = next((x for x in xls.sheet_names if x.casefold() == AQR_SHEETS[factor].casefold()), None)
    if sheet is None:
        raise ValueError(f"AQR {factor}: expected sheet '{AQR_SHEETS[factor]}', available: {xls.sheet_names}")
    raw = pd.read_excel(xls, sheet_name=sheet, header=18)
    raw.columns = [str(c).strip() for c in raw.columns]
    dc = next((x for x in raw.columns if x.upper() == "DATE"), None)
    uc = next((x for x in raw.columns if x.upper() in {"USA", "US"}), None)
    if dc is None or uc is None:
        raise ValueError(f"AQR {factor}: cannot identify DATE and USA columns: {list(raw.columns)[:35]}")
    dates = pd.to_datetime(raw[dc], errors="coerce")
    val = pd.to_numeric(raw[uc], errors="coerce")
    valid = dates.notna() & val.notna()
    if not valid.any():
        raise ValueError(f"AQR {factor}: workbook has no valid observations")
    series = pd.Series(val.loc[valid].to_numpy(dtype=float), index=dates.loc[valid] + pd.offsets.MonthEnd(0), name=factor)
    if units == "percent":
        series = series / 100.0
    elif units != "decimal":
        raise ValueError("AQR units must be 'decimal' or 'percent'; never silently infer")
    if series.index.has_duplicates:
        raise ValueError(f"AQR {factor}: duplicate months")
    series = series.sort_index()
    if (series.abs() > 1.0).any():
        raise ValueError(f"AQR {factor}: returns exceed 100% monthly; check workbook units")
    return series


def load_aqr_monthly(factor: str, cache_dir: str | Path = "data/cache",
                     refresh: bool = False, units: str = "decimal",
                     local_path: str | Path | None = None) -> pd.Series:
    if factor not in AQR_URLS:
        raise ValueError(f"Unknown AQR factor: {factor}")
    if local_path is None:
        cache = Path(cache_dir) / AQR_URLS[factor].rsplit('/', 1)[-1]
        source = _cached_download(AQR_URLS[factor], cache, refresh=refresh)
    else:
        source = Path(local_path)
    return parse_aqr_monthly_xlsx(source, factor, units=units)


# Original Kim-Wright THREEFY yield source, published by the Federal Reserve Board.
# Same 3-factor estimated zero-coupon curve as FRED THREEFY1..THREEFY10.
# NOT Gurkaynak-Sack-Wright (GSW) feds200628; NOT Treasury par CMT data.
FED_KW_BULK_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200533.csv"


def parse_kw_board_csv(content: bytes | str) -> pd.DataFrame:
    """Parse official FEDS 2005-33 CSV; yields in PERCENT, daily observations.

    Accepts commented/preamble rows and 40-column source or a user-saved
    11-column subset. Uses strict day-month-year for historical Board files.
    """
    source = content.decode('utf-8-sig', errors='replace') if isinstance(content, bytes) else content
    lines = source.splitlines()
    key_cols = [f"THREEFY{i:02d}00.B" for i in range(1, 11)]
    header_ix = None
    for j, line in enumerate(lines):
        fields = [x.strip().strip('"') for x in next(csv.reader([line]))]
        if all(k in fields for k in key_cols) and any(x.lower() == 'date' for x in fields):
            header_ix = j
            break
    if header_ix is None:
        raise ValueError(
            "Kim-Wright Board CSV has no DATE + THREEFY0100.B...THREEFY1000.B "
            "header; source may be an HTML error page or a different yield dataset."
        )
    raw = pd.read_csv(io.StringIO('\n'.join(lines[header_ix:])), on_bad_lines='skip', low_memory=False)
    raw.columns = [str(x).strip().strip('"') for x in raw.columns]
    dc = next(c for c in raw if c.casefold() == 'date')
    raw_dates = raw[dc].astype(str).str.strip()
    # Official Board reports use DD-MM-YYYY (e.g. 10-03-2026 is March 10).
    formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d']
    parsings = [(pd.to_datetime(raw_dates, format=fmt, errors='coerce'), fmt) for fmt in formats]
    dt, fmt = max(parsings, key=lambda x: x[0].notna().sum())
    if dt.notna().sum() < 60:
        raise ValueError('Kim-Wright CSV date parsing failed or history too short')
    z = raw[key_cols].apply(pd.to_numeric, errors='coerce')
    z.index = pd.DatetimeIndex(dt, name='Date')
    z.columns = [f'{i}Y' for i in range(1, 11)]
    z = z.loc[~z.index.isna()].sort_index()
    z = z.loc[~z.index.duplicated(keep='last')]
    z = z.dropna(how='all')
    if not (len(z) >= 60 and (z.notna().sum() >= 60).all()):
        raise ValueError('Kim-Wright yield grid is empty or missing essential maturities')
    return z


def load_kw_zero_curve(cache_dir: str | Path = 'data/cache', refresh: bool = False,
                       source: str = 'auto', local_path: str | Path | None = None) -> pd.DataFrame:
    """Monthly fitted zero yields in PERCENT (division by 100 happens in RatesData).

    source='board' : single authoritative Board CSV request (preferred).
    source='fred'  : 10 FRED series as explicit fallback.
    source='auto'  : try Board first, then original FRED series if it fails.
    local_path    : raw official Board CSV manually downloaded by the user.
    No substitution with different yield definitions.
    """
    cache = Path(cache_dir)
    if local_path is not None:
        content = Path(local_path).read_bytes()
        grid = parse_kw_board_csv(content)
        return grid.resample('ME').last().dropna()
    if source not in {'auto', 'board', 'fred'}:
        raise ValueError("source must be 'auto', 'board' or 'fred'")
    errors = []
    if source in {'auto', 'board'}:
        try:
            print('Treasury curve: loading one official Kim-Wright Board CSV (10 tenors).')
            content = _cached_download(FED_KW_BULK_URL, cache/'feds200533.csv', refresh=refresh)
            grid = parse_kw_board_csv(content)
            return grid.resample('ME').last().dropna()
        except (ConnectionError, ValueError, OSError, pd.errors.ParserError) as e:
            errors.append(f'Board: {e}')
            if source == 'board':
                raise ConnectionError(
                    "Official Board Kim-Wright CSV unavailable. Open "
                    f"{FED_KW_BULK_URL} in your browser, save it locally, "
                    "and set BOARD_CSV_PATH in notebook configuration. " + str(e)) from e
            warnings.warn(f'Board source unavailable, falling back to FRED THREEFY series: {e}', RuntimeWarning)
    try:
        curves = {}
        for i in range(1, 11):
            name = f'THREEFY{i}'
            print(f'FRED curve fallback: {name} ({i}/10)')
            curves[f'{i}Y'] = fred_csv_series(name, cache_dir=cache, refresh=refresh)
        grid = pd.DataFrame(curves)
        return grid.resample('ME').last().dropna()
    except Exception as e:
        errors.append(f'FRED: {e}')
        raise ConnectionError(
            'Cannot obtain Kim-Wright zero curve from its original official Board file '
            'or FRED. Download https://www.federalreserve.gov/data/yield-curve-tables/feds200533.csv '
            'with your browser and provide BOARD_CSV_PATH. Details: ' + ' | '.join(errors)) from e


def parse_fred_csv(content: bytes | str, name: str) -> pd.Series:
    """Read exact FRED csv series; source units retained (usually percent)."""
    raw = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content),
                      na_values=['.'])
    raw.columns = [str(x).strip() for x in raw.columns]
    if name not in raw.columns:
        raise ValueError(f'FRED CSV missing {name}: {list(raw.columns)}')
    dt = pd.to_datetime(raw.iloc[:, 0], errors='coerce')
    ser = pd.Series(pd.to_numeric(raw[name], errors='coerce').to_numpy(dtype=float),
                    index=pd.DatetimeIndex(dt, name='Date'), name=name)
    return ser.loc[~ser.index.isna()].sort_index()


def parse_fred_text(content: bytes | str, name: str) -> pd.Series:
    """Parse FRED /data/NAME.txt public text fallback (same series)."""
    text = content.decode('utf-8-sig', errors='replace') if isinstance(content, bytes) else content
    pairs = []
    for line in text.splitlines():
        m = re.match(r'^\s*(\d{4}-\d{2}-\d{2})\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)|\.)\s*$', line)
        if m:
            pairs.append((m.group(1), float(m.group(2)) if m.group(2) != '.' else np.nan))
    if len(pairs) < 12:
        raise ValueError(f'FRED text source for {name} has insufficient dated observations')
    return pd.Series([v for _, v in pairs], index=pd.to_datetime([d for d,_ in pairs]), name=name).rename_axis('Date')


def fred_csv_series(name: str, cache_dir: str | Path = 'data/cache', refresh: bool = False,
                    local_path: str | Path | None = None) -> pd.Series:
    """Download a specific FRED series with cached CSV and same-series TXT fallback."""
    cache = Path(cache_dir)
    if local_path is not None:
        path = Path(local_path)
        if path.suffix.lower() == '.txt':
            return parse_fred_text(path.read_bytes(), name)
        return parse_fred_csv(path.read_bytes(), name)
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}'
    csv_path = cache/f'fred_{name}.csv'
    if csv_path.exists() and not refresh:
        return parse_fred_csv(csv_path.read_bytes(), name)
    try:
        return parse_fred_csv(_cached_download(url, csv_path, refresh=refresh, retries=1), name)
    except (ConnectionError, ValueError) as err:
        try:
            txt_path = cache/f'fred_{name}.txt'
            return parse_fred_text(_cached_download(f'https://fred.stlouisfed.org/data/{name}.txt',
                                                   txt_path, refresh=refresh, retries=1), name)
        except Exception as err2:
            raise ConnectionError(
                f'FRED {name} unavailable from CSV and original text source. '
                f'Save original CSV into {csv_path} (or supply local_fedfunds_path). '
                f'CSV error: {err}; TXT error: {err2}') from err2


# v0.2.3: a verified public CSV endpoint, with strict validation. DBnomics
# short URL FRED/FEDFUNDS (used in v0.2.2) returned HTTP 404 and is removed.
# The mirror explicitly republishes FRED's FEDFUNDS observations, not an ETF,
# target-rate, or a SOFR surrogate. Source is third-party, not FRED itself.
FEDFUNDS_MIRROR_URL = 'https://eco3min.fr/dataset/federal-funds-rate.csv'
# Board DDP H.15 monthly preview: recent official values, optional for updating
# the mirror when it lags. This preview may be retired in 2027.
FEDFUNDS_BOARD_PREVIEW = (
    'https://www.federalreserve.gov/datadownload/Preview.aspx'
    '?pi=400&preview=H15%2FH15%2FRIFSPFF_N.M&rel=H15'
)


def parse_fedfunds_mirror(content: bytes | str) -> pd.Series:
    """Strictly validate the monthly FRED/FEDFUNDS replication CSV.

    Column names are documented by the provider as date,fed_funds_rate.
    Values are percent per year; no interpolation or automatic conversions.
    """
    raw = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content))
    raw.columns = [str(x).strip().lstrip('\ufeff') for x in raw.columns]
    if not {'date','fed_funds_rate'}.issubset(raw.columns):
        raise ValueError(f'FEDFUNDS mirror unexpected schema: {list(raw.columns)}')
    dates = pd.to_datetime(raw['date'], errors='coerce')
    values = pd.to_numeric(raw['fed_funds_rate'], errors='coerce')
    if dates.isna().any() or values.isna().any():
        raise ValueError('FEDFUNDS mirror has missing/unparseable dates or observations')
    idx = pd.DatetimeIndex(dates + pd.offsets.MonthEnd(0), name='Date')
    ser = pd.Series(values.to_numpy(dtype=float), index=idx, name='FEDFUNDS').sort_index()
    if ser.index.has_duplicates or not (ser.index.is_month_end.all()):
        raise ValueError('FEDFUNDS mirror duplicated or misaligned observations')
    if not (ser.index[0] <= pd.Timestamp('1954-07-31') and ser.index[-1] >= pd.Timestamp('2024-01-31')
            and len(ser) > 800 and np.all(ser.between(0, 25))):
        raise ValueError('FEDFUNDS mirror incomplete, wrong range or wrong percent scale')
    # July 1954: FEDFUNDS=0.80% (authoritative FRED historical table).
    if abs(ser.loc['1954-07-31'] - 0.80) > .011:
        raise ValueError('FEDFUNDS mirror failed historical-source spot check')
    expected = pd.date_range(ser.index[0], ser.index[-1], freq='ME')
    if len(expected) != len(ser):
        raise ValueError('FEDFUNDS mirror has missing calendar months')
    return ser


def parse_board_h15_preview(content: bytes | str) -> pd.Series:
    """Extract ONLY official H.15 RIFSPFF_N.M monthly rows from HTML preview.

    This is not the full historical series: it supplements a validated full
    history from the CSV mirror. Unsupported page formats fail without guesswork.
    """
    from html.parser import HTMLParser
    class _Table(HTMLParser):
        def __init__(self):
            super().__init__(); self.in_tr = False; self.in_cell = False
            self.rows=[]; self.row=[]; self.cell=[]
        def handle_starttag(self, tag, attrs):
            if tag == 'tr': self.in_tr=True; self.row=[]
            if self.in_tr and tag in ('td','th'): self.in_cell=True; self.cell=[]
        def handle_data(self,data):
            if self.in_tr and self.in_cell:self.cell.append(data)
        def handle_endtag(self,tag):
            if tag in ('td','th') and self.in_cell:
                self.row.append(''.join(self.cell).strip()); self.in_cell=False
            if tag == 'tr' and self.in_tr:
                self.rows.append(self.row); self.in_tr=False
    html = content.decode('utf-8-sig', errors='replace') if isinstance(content, bytes) else content
    reader = _Table(); reader.feed(html)
    found = {}
    for row in reader.rows:
        raw=' '.join(row)
        if 'RIFSPFF_N.M' not in raw:
            continue
        m=re.search(r'\b(20\d{2}-(?:0[1-9]|1[0-2]))\b',raw)
        if not m:continue
        after=' '.join(row[row.index(next(x for x in row if m.group(1) in x))+1:]) if any(m.group(1) in x for x in row) else ''
        v=re.search(r'^\s*([+-]?\d{1,2}(?:\.\d{1,4})?)\s*$',after)
        if not v:continue
        found[pd.Timestamp(m.group(1)+'-01')+pd.offsets.MonthEnd(0)]=float(v.group(1))
    if len(found) < 5:
        raise ValueError('Board H.15 preview lacked at least five identifiable monthly effective-rate observations')
    s=pd.Series(found,name='FEDFUNDS').sort_index().rename_axis('Date')
    if not s.between(0,25).all():raise ValueError('Board H15 invalid units')
    return s


def supplement_fedfunds_with_board(base: pd.Series, content: bytes | str) -> pd.Series:
    """Append verified Board H15 months, never revise mirror's older values.

    Refuse to mix definitions if they disagree on overlapping months.
    """
    board = parse_board_h15_preview(content)
    common=board.index.intersection(base.index)
    if len(common)<3:
        raise ValueError('FEDFUNDS Board preview has insufficient overlap to validate source alignment')
    maxdiff=float((board.loc[common]-base.loc[common]).abs().max())
    if maxdiff > .021:
        raise ValueError(f'FEDFUNDS Board and mirror disagree by up to {maxdiff:.3f} percentage points')
    extra=board.loc[board.index>base.index.max()]
    if len(extra):
        needed=pd.date_range(base.index[-1]+pd.offsets.MonthEnd(1),extra.index[-1],freq='ME')
        if not needed.isin(extra.index).all():
            raise ValueError('Board supplemental dates have gaps; refusing to fill')
        return pd.concat([base,extra]).rename('FEDFUNDS')
    return base


def load_fedfunds(cache_dir: str | Path = 'data/cache', refresh: bool = False,
                  local_path: str | Path | None = None, source: str = 'mirror',
                  update_with_board: bool = True) -> pd.Series:
    """Return verified monthly FEDFUNDS values in PERCENT (not decimal).

    source='mirror' : Eco3min CSV republishing monthly FEDFUNDS; no FRED contact.
    source='fred'   : original blocked FRED endpoints; explicit opt-in only.
    source='auto'   : mirror first, then FRED if mirror fails.
    source='local'  : requires the original FEDFUNDS CSV/TXT via local_path.
    Board H15 preview can append recently published observations, only after
    confirming overlap with the mirror; errors here do NOT discard valid base.
    """
    cache=Path(cache_dir)
    if local_path is not None:
        path=Path(local_path)
        if not path.is_file():raise FileNotFoundError(f'FEDFUNDS local path does not exist: {path}')
        raw=path.read_bytes()
        if path.suffix.lower()=='.txt': return parse_fred_text(raw,'FEDFUNDS')
        # Allow manual download of either original FRED or the documented mirror CSV.
        try:return parse_fred_csv(raw,'FEDFUNDS')
        except ValueError:return parse_fedfunds_mirror(raw)
    if source not in ('mirror','fred','auto','local'):
        raise ValueError('funding source must be mirror, fred, auto or local (not removed DBnomics)')
    localcsv=cache/'fred_FEDFUNDS.csv'
    if localcsv.exists() and not refresh:return parse_fred_csv(localcsv.read_bytes(),'FEDFUNDS')
    if source=='local':raise FileNotFoundError('source=local requires FEDFUNDS_CSV_PATH or cached fred_FEDFUNDS.csv')
    if source=='fred':return fred_csv_series('FEDFUNDS',cache_dir=cache,refresh=refresh)
    mirror_error=None
    try:
        payload=_cached_download(FEDFUNDS_MIRROR_URL, cache/'mirror_FEDFUNDS.csv',
                                 refresh=refresh, timeout=12, retries=1)
        series=parse_fedfunds_mirror(payload)
        print(f'Funding: verified FEDFUNDS mirror ({len(series)} months through {series.index.max().date()}).')
        if update_with_board:
            try:
                ext=_cached_download(FEDFUNDS_BOARD_PREVIEW,cache/'board_H15_FEDFUNDS_preview.html',
                                     refresh=refresh,timeout=10,retries=0,allow_html=True)
                series_new=supplement_fedfunds_with_board(series,ext)
                if len(series_new)>len(series):
                    print(f'Funding: {len(series_new)-len(series)} later months appended from official Board H.15 preview.')
                series=series_new
            except (ConnectionError,ValueError,KeyError,IndexError,OSError) as ex:
                warnings.warn(f'Board recent-month supplement unavailable: {ex}. '
                              f'Retaining verified FEDFUNDS values through {series.index.max().date()} only.',RuntimeWarning)
        return series
    except (ConnectionError,ValueError,KeyError,OSError) as ex:
        mirror_error=ex
    if source=='auto':
        try:return fred_csv_series('FEDFUNDS',cache_dir=cache,refresh=refresh)
        except (ConnectionError,ValueError) as ex:
            raise ConnectionError(f'FEDFUNDS mirror and FRED unavailable: mirror={mirror_error}; FRED={ex}') from ex
    raise ConnectionError('FEDFUNDS CSV mirror unavailable or failed strict validation. '
                          'Download the original series at https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS '
                          'OR https://eco3min.fr/dataset/federal-funds-rate.csv '
                          'and upload it to Colab, then set FEDFUNDS_CSV_PATH. '
                          f'Details: {mirror_error}') from mirror_error
