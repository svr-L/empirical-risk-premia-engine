"""Cross-asset traded-return proxies; no conversion of yield changes into returns."""
from pathlib import Path
import pandas as pd
import numpy as np
from .equity_shock_followup import _monthly
ETF_TICKERS=('IEF','HYG','UUP','DBC','VIXY')

def etf_factor_returns(prices,rf,*,as_of='2026-10-31'):
    """Adjusted USD close prices, including distributions/splits; monthly decimals.
    Credit proxy is a simple HYG-IEF spread, not duration-neutral. VIXY is a
    futures ETF, not spot VIX or pure variance premium. All long legs funded RF.
    """
    p=prices.copy();p.index=pd.to_datetime(p.index);p=p.sort_index().loc[:as_of]
    if p.index.has_duplicates:raise ValueError('Duplicate adjusted-price dates')
    if not set(ETF_TICKERS)<=set(p):raise ValueError('Missing ETF adjusted-price columns')
    p=p[list(ETF_TICKERS)].apply(pd.to_numeric,errors='coerce');p=p.where(p>0)
    cutoff=pd.Timestamp(as_of);last_complete=cutoff if cutoff.is_month_end else cutoff+pd.offsets.MonthEnd(-1)
    # Require end-of-month observation, not an arbitrary stale price early in month.
    if not p.empty:
        obs=pd.DataFrame({c:pd.Series(p.index,index=p.index).where(p[c].notna()) for c in p})
        last=obs.resample('ME').last(); m=p.resample('ME').last(); maxgap=pd.DataFrame({c:(last.index.to_series()-last[c]).dt.days for c in p})
        m=m.where(maxgap<=7).loc[:last_complete]
    else:raise ValueError('No ETF prices')
    r=m.pct_change(fill_method=None); rf=pd.Series(rf).reindex(r.index)
    return pd.DataFrame({'XA_DURATION':r.IEF-rf,'XA_CREDIT':r.HYG-r.IEF,'XA_USD':r.UUP-rf,'XA_COMMODITY':r.DBC-rf,'XA_VOL_FUTURES':r.VIXY-rf}).dropna(how='all')

def load_etf_adjusted_prices(*,cache_dir='erp_cache/cross_asset',path=None,as_of='2026-10-31'):
    """Explicit local adjusted-close CSV or optional yfinance source. No fake fallback."""
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True); dest=cache/f'etf_adjusted_prices_{as_of}.csv'
    if path is not None or dest.exists():
        src=Path(path) if path is not None else dest
        p=pd.read_csv(src,index_col=0,parse_dates=True);source=str(src)
    else:
        try:import yfinance as yf
        except ImportError as e:raise ConnectionError('Install yfinance or supply adjusted-price CSV') from e
        raw=yf.download(list(ETF_TICKERS),start='2000-01-01',end=(pd.Timestamp(as_of)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=False,progress=False,threads=False,timeout=12)
        if raw.empty or 'Adj Close' not in raw:raise ConnectionError('No adjusted closes from Yahoo Finance')
        p=raw['Adj Close'];source='Yahoo Finance via yfinance, adjusted close (revised data)'
        if not set(ETF_TICKERS)<=set(p) or p[list(ETF_TICKERS)].notna().sum().min()<2:raise ConnectionError('Incomplete ETF download; supply a local adjusted-price CSV')
        p.to_csv(dest,index_label='Date')
    return p,source

def load_rates_factor_returns(path):
    p=pd.read_csv(path,index_col=0,parse_dates=True)
    cols=['Level','Slope','Curvature']
    if not set(cols)<=set(p):raise ValueError('Expected ERP Rates factor_returns.csv with Level/Slope/Curvature')
    return _monthly(p[cols].astype(float)).add_prefix('XA_RATES_')

def managed_factor_returns(ff,states,*,lag=2,window=60,min_obs=36,clip=2.):
    """Predeclared state-scaled MKT/HML, using lagged and historical standardized states.
    Lag is measured in calendar months. No future or contemporaneous macro observation.
    Latest-vintage macro data still not point-in-time.
    """
    ff,states=map(_monthly,(ff,states)); names=['CPI_INFLATION_YOY_PCT','TERM_SPREAD_10Y2Y_PP'];fac=['MKT','HML']
    if not set(names)<=set(states) or not set(fac)<=set(ff):raise ValueError('Missing managed factor/state columns')
    s=states[names].reindex(pd.date_range(states.index.min(),max(states.index.max(),ff.index.max()),freq='ME')).shift(lag)
    history=s.shift(1).rolling(window,min_periods=min_obs); z=((s-history.mean())/history.std()).clip(-clip,clip).reindex(ff.index)
    return pd.DataFrame({f'MANAGED_{f}_{v}':ff[f]*z[v] for f in fac for v in names})
