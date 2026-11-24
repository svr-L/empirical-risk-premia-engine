# Run this cell ONCE in your existing v0.12 Colab session, BEFORE rerunning section 10 (JK shocks).
# No need to rerun the full FF/PCA/pricing analysis.
import erp.equity_shocks as _jkmod
from io import BytesIO
import numpy as np, pandas as pd, warnings
JK_REQUIRED=_jkmod.JK_REQUIRED
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

_jkmod._jk_parse=_jk_parse
print('Jarocinski-Karadi safe parser installed. Re-run section 10, then the audit/export cells.')
