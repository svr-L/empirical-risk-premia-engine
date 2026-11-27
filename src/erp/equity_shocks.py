"""Economic shocks with event-based identification, not differenced FRED levels.

Primary public input: Jarocinski--Karadi updated Fed monetary-policy (MP) and
central-bank-information (CBI) announcement surprises, monthly aggregation.
Separate optional inflation RELEASE surprises require actual + pre-release
consensus, and cannot be generated from revised CPI levels alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import zipfile
from io import BytesIO
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

JK_URL='https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/main/shocks_fed_jk_m.csv'
JK_REQUIRED=('MP_median','CBI_median','MP_pm','CBI_pm','pc1_hf')

@dataclass
class EconomicShockPanel:
    shocks:pd.DataFrame
    provenance:dict
    def export(self,folder):
        p=Path(folder);p.mkdir(parents=True,exist_ok=True)
        self.shocks.to_csv(p/'identified_economic_shocks_monthly.csv',index_label='DATE')
        (p/'shock_provenance.json').write_text(json.dumps(self.provenance,indent=2,default=str))
        return p


def _jk_parse(raw):
    """Parse official monthly JK data without imputing unidentified shocks.

    The author CSV occasionally has MP_median + CBI_median != pc1_hf,
    including months with unavailable high-frequency stock-market data.
    These months are retained in the calendar index but set to NaN, NEVER
    silently treated as zero-shock months or used in regression estimates.
    Valid author-defined zero-event months remain actual zeros.
    """
    import warnings
    r=pd.read_csv(BytesIO(raw))
    r.columns=r.columns.str.strip()
    if not {'year','month',*JK_REQUIRED}.issubset(r.columns):
        raise ValueError('JK source requires year,month,MP_median,CBI_median,MP_pm,CBI_pm,pc1_hf')
    d=pd.to_datetime({'year':r.year.astype(int),'month':r.month.astype(int),'day':1})+pd.offsets.MonthEnd(0)
    a=r[list(JK_REQUIRED)].apply(pd.to_numeric,errors='coerce')
    a.index=pd.DatetimeIndex(d)
    if a.index.has_duplicates or not a.index.is_monotonic_increasing:
        raise ValueError('Duplicate/unsorted JK months')
    if not a.index.equals(pd.date_range(a.index.min(),a.index.max(),freq='ME')):
        raise ValueError('Missing calendar months in JK: do not silently impute zero')
    miss=~np.isfinite(a[['MP_median','CBI_median','pc1_hf']].to_numpy()).all(axis=1)
    diff=(a.MP_median+a.CBI_median-a.pc1_hf).abs()
    mismatch=~np.isclose(a.MP_median+a.CBI_median,a.pc1_hf,atol=2e-5,rtol=1e-4)
    bad=pd.Series(miss|mismatch,index=a.index)
    if bad.sum()>max(12,int(.05*len(a))):
        raise ValueError(f'JK data have {int(bad.sum())} inconsistent months (>5% or 12); validate the source before using shocks')
    excluded=[str(t.date()) for t in a.index[bad]]
    if excluded:
        warnings.warn('JK source contains '+str(len(excluded))+' months without a valid median decomposition: '+', '.join(excluded)+'. These are set to NaN (NOT zero); they are excluded from regressions.',RuntimeWarning,stacklevel=2)
        a.loc[bad,list(JK_REQUIRED)]=np.nan
    a=a[['MP_median','CBI_median','MP_pm','CBI_pm','pc1_hf']]
    a.attrs['excluded_invalid_decomposition_months']=excluded
    return a

def load_jk_monetary_shocks(*,cache_dir='erp_cache/economic_shocks',path=None,
          refresh=False,as_of='2026-10-31',timeout=15,session=None):
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    saved=cache/'jk_monthly_original.csv'
    if path is not None:
        content=Path(path).read_bytes();source='local original JK CSV'
    elif saved.exists() and not refresh:
        content=saved.read_bytes();source='cached original JK CSV'
    else:
        try:
            get=requests.get if session is None else session.get
            response=get(JK_URL,timeout=(5,timeout))
            response.raise_for_status();content=response.content
            source='live author-maintained JK GitHub CSV'
        except requests.RequestException as exc:
            raise ConnectionError('JK download unavailable: supply path to shocks_fed_jk_m.csv from '+JK_URL+
                                  '; network detail: '+str(exc)[:140]) from exc
    panel=_jk_parse(content)
    excluded=panel.attrs.get('excluded_invalid_decomposition_months',[])
    if as_of is not None:panel=panel.loc[:pd.Timestamp(as_of)]
    if panel.empty:raise ValueError('No shocks at as_of')
    if panel[['MP_median','CBI_median']].dropna().empty:raise ValueError('No valid JK median shock months at as_of')
    saved.write_bytes(content)
    prov={'source':source,'original_url':JK_URL,'reference':'Jarocinski & Karadi (2020), AEJ: Macroeconomics, updated author replication',
          'data_units':'policy rate shock in percentage points per year',
          'median_identification':'event-window sign-restriction decomposition',
          'zero_event_months':'explicit zero per source; not imputed',
          'max_date':str(panel.index.max().date()),'latest_vintage':True,
          'not_structural_causal_proof':'sign restrictions and high-frequency identification assumptions apply',
          'unknown_after_end':'months after source last date are missing (not zeros)',
          'excluded_invalid_decomposition_months':[d for d in excluded if as_of is None or pd.Timestamp(d)<=pd.Timestamp(as_of)],
          'invalid_month_treatment':'NaN, excluded from regressions; never imputed as zero'}
    return EconomicShockPanel(panel,prov)


def load_inflation_release_surprises(path,*,as_of='2026-10-31'):
    """Release-aligned actual minus forecast: user-provided pre-announcement consensus.

    Required: release_date, actual, consensus, consensus_source. Optional unit.
    Date refers to RELEASE month, not reference CPI measurement month. Multiple
    releases in a month are summed, with no artificial zero for absent months.
    """
    b=Path(path).read_bytes()
    if zipfile.is_zipfile(BytesIO(b)):
        with zipfile.ZipFile(BytesIO(b)) as z:
            files=[name for name in z.namelist() if name.lower().endswith('.csv')]
            if len(files)!=1:raise ValueError('Inflation ZIP requires exactly one release-level CSV')
            raw=pd.read_csv(BytesIO(z.read(files[0])))
    else:raw=pd.read_csv(BytesIO(b))
    required={'release_date','actual','consensus','consensus_source'}
    if not required.issubset(raw.columns):raise ValueError('Inflation releases require '+', '.join(sorted(required)))
    if raw.consensus_source.isna().any() or raw.consensus_source.astype(str).str.strip().eq('').any():
        raise ValueError('Each release needs provenance for pre-release consensus')
    dt=pd.to_datetime(raw.release_date,errors='raise')
    if dt.isna().any() or dt.duplicated().any():raise ValueError('Invalid/duplicate release dates')
    for col in ['actual','consensus']:
        raw[col]=pd.to_numeric(raw[col],errors='raise')
        if not np.isfinite(raw[col]).all():raise ValueError('Nonfinite inflation release surprise')
    raw['month']=dt+pd.offsets.MonthEnd(0)
    if as_of is not None:raw=raw.loc[dt<=pd.Timestamp(as_of)]
    if raw.empty:raise ValueError('No CPI release surprises before as_of')
    surprises=(raw['actual']-raw['consensus']).groupby(raw['month']).sum().to_frame('CPI_RELEASE_SURPRISE')
    return EconomicShockPanel(surprises,{'source':'user-supplied, release-level actual - pre-release consensus',
        'measurement':'inflation news at CPI release, NOT contemporaneous CPI monthly change',
        'source_coverage':'only months containing supplied releases, no zero filling',
        'consensus_sources':sorted(map(str,raw.consensus_source.unique())),
        'units':'as supplied: actual and consensus must share same units; no automatic standardization',
        'point_in_time_requirement':'consensus must be timestamped before actual announcement; parser cannot verify provider timestamp'})


def _one_shock_mapping(y,shocks,*,train_end,lag=0):
    """Formation regression, OOS holdout same months; lag=1 forecasts next month.
    Fit y_t=a+b shocks_(t-lag); lag=0 is *descriptive contemporary*
    association, never an investable EOM-before-announcement forecast.
    """
    # shift calendar labels, not rows: missing announcement months stay missing.
    shifted=shocks.shift(lag,freq='ME') if lag else shocks
    aligned=pd.concat([y.rename('return'),shifted],axis=1,join='inner').dropna()
    tr=aligned.index<=pd.Timestamp(train_end)
    if tr.sum()<60 or (~tr).sum()<24:return None
    cols=list(shocks.columns)
    x=aligned[cols];response=aligned['return']
    if np.linalg.matrix_rank(np.column_stack([np.ones(int(tr.sum())),x.loc[tr].to_numpy()])) < 1+len(cols):
        return {'model':'insufficient_identification_rank','n_train':int(tr.sum()),'n_oos':int((~tr).sum())}
    ztr=sm.add_constant(x.loc[tr],has_constant='add')
    fit=sm.OLS(response.loc[tr],ztr).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    pred=fit.predict(sm.add_constant(x.loc[~tr],has_constant='add'))
    test=response.loc[~tr]
    null=float(response.loc[tr].mean())
    mse=float(np.mean((test-pred)**2));bas=float(np.mean((test-null)**2))
    out={'train_r2':float(fit.rsquared),'oos_r2_vs_train_mean':float(1-mse/bas) if bas>1e-15 else np.nan,
          'n_train':int(tr.sum()),'n_oos':int((~tr).sum()),'oos_start':str(test.index.min().date()),
          'oos_end':str(test.index.max().date()),'lag_months':lag,
          'interpretation':'contemporaneous association' if lag==0 else 'forecast using prior month event shocks'}
    for col in cols:
        out[f'exposure_{col}']=float(fit.params[col])
        out[f't_hac_{col}']=float(fit.tvalues[col])
    return out


def analyze_equity_shock_mapping(ff:pd.DataFrame,pc_returns:pd.DataFrame,
         shocks:EconomicShockPanel,*,train_end='2005-12-31',min_oos=24):
    """Evaluate FF/AQR and RP factor returns on published event shocks.

    Full calendar union is NOT filled: dates beyond dataset end not assumed 0.
    Controls: no factor from S&P500 high-frequency outcome in JK input.
    """
    src=shocks.shocks
    if {'MP_median','CBI_median'}.issubset(src.columns):
        chosen=src[['MP_median','CBI_median']].copy()
        family='JK_MP_CBI'
    elif 'CPI_RELEASE_SURPRISE' in src.columns:
        chosen=src[['CPI_RELEASE_SURPRISE']].copy()
        family='CPI_actual_minus_prerelease_consensus'
    else:
        raise ValueError('Unknown verified shock family')
    cols=pd.concat([ff,pc_returns],axis=1)
    rows=[]
    for response in cols.columns:
        for lag in (0,1):
            res=_one_shock_mapping(cols[response],chosen,train_end=train_end,lag=lag)
            if res is not None:
                rows.append({'factor':response,'shock_group':family,'lag':lag,**res})
    out=pd.DataFrame(rows)
    if len(out) and 'n_oos' in out and out.n_oos.notna().any():
        out=out.loc[out.n_oos>=min_oos].copy()
    return out


def estimate_jk_shock_prices(heldout_excess:pd.DataFrame,ff:pd.DataFrame,
        shocks:EconomicShockPanel,*,train_end='2005-12-31',hac_lags=6):
    """Exploratory conditional cross-sectional price of JK policy/info risk.

    Using heldout portfolio excess returns, estimate FF+MP+CBI beta exposures
    in formation, infer monthly cross-sectional lambda (with intercept), and
    validate conditional vs FF-only mean pricing on identical subsequent months.
    Nontraded shocks: lambda is price *per exposure unit*, NOT mean shock return.
    This does not establish causality or correct cross-sectional overlap, weak
    beta errors or selection bias. None of the OOS outcomes fit the betas/lambda.
    """
    from .equity_v012 import _fmb_train_oos
    if not {'MP_median','CBI_median'}.issubset(shocks.shocks):
        raise ValueError('Requires two JK median high-frequency event shocks')
    panel=pd.concat([heldout_excess,ff,shocks.shocks[['MP_median','CBI_median']]],axis=1,join='inner').dropna()
    tr=panel.index<=pd.Timestamp(train_end)
    if int(tr.sum())<96 or int((~tr).sum())<36:raise ValueError('Too few common months for shock-price validation')
    ys=panel[heldout_excess.columns];base=panel[ff.columns];new=panel[['MP_median','CBI_median']]
    stats=[];lambdas=[]
    for name,x in [('FF_AQR',base),('FF_AQR_PLUS_JK',pd.concat([base,new],axis=1))]:
        pred,coef,info=_fmb_train_oos(ys.loc[tr].to_numpy(),ys.loc[~tr].to_numpy(),
                                      x.loc[tr].to_numpy(),x.loc[~tr],hac_lags=hac_lags)
        stats.append({'model':name,**info,'formation_months':int(tr.sum()),
                      'oos_months':int((~tr).sum()),'oos_start':str(ys.loc[~tr].index[0].date()),
                      'oos_end':str(ys.loc[~tr].index[-1].date())})
        coef['model']=name;lambdas.append(coef)
    summary=pd.DataFrame(stats)
    prices=pd.concat(lambdas,ignore_index=True)
    return summary,prices
