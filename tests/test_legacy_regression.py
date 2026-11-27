"""Numerical regression tests against functions extracted from the user's original notebook sources.

No network or notebook execution; compares algorithms on controlled inputs.
"""
from pathlib import Path
import ast
from dataclasses import dataclass
import nbformat
import numpy as np
import pandas as pd
import statsmodels.api as sm
from erp.equity import _alpha_metrics
from erp.rates import exact_funded_factor_weights, zc_return_decomposition

ROOT=Path(__file__).resolve().parents[1]


def legacy_functions(file_name,names,extra):
    nb=nbformat.read(ROOT/'legacy'/file_name,as_version=4)
    namespace={'np':np,'pd':pd,'sm':sm,'dataclass':dataclass,**extra}
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        # AST isolates def/class nodes only, no expensive notebook loading.
        tree=ast.parse(cell.source)
        for item in tree.body:
            if isinstance(item,(ast.FunctionDef,ast.ClassDef)) and item.name in names:
                exec(compile(ast.Module(body=[item],type_ignores=[]),'legacy', 'exec'),namespace)
    assert all(name in namespace for name in names)
    return namespace


def test_equity_v01_ts_same_exposures_and_residuals():
    rng=np.random.default_rng(16)
    ix=pd.date_range('2000-01-31',periods=140,freq='ME')
    f=pd.DataFrame(rng.normal(0,.03,(140,5)),index=ix,columns=['MKT','SMB','HML','RMW','CMA'])
    yf=pd.DataFrame(f.to_numpy()@rng.normal(size=(5,25))+.004+rng.normal(0,.02,(140,25)),index=ix,columns=[f'P{i}' for i in range(25)])
    old=legacy_functions('equity_v0_1_reference.ipynb',{'TimeSeriesModelFit','fit_time_series_model'},
               {'HAC_LAGS':6,'FACTORS_BY_MODEL':{'FF5':f},'MODEL_REGISTRY':{'FF5':{'factors':list(f.columns)}}})
    ref=old['fit_time_series_model'](yf,'FF5')
    new_stats,b,a,res,yr,ff=_alpha_metrics(yf,f)
    pd.testing.assert_frame_equal(ref.betas,b)
    pd.testing.assert_frame_equal(ref.residuals,res,check_dtype=False)
    assert np.allclose(ref.asset_stats['alpha_monthly'],new_stats['alpha_monthly'])
    assert np.allclose(ref.asset_stats['alpha_t_hac'],new_stats['alpha_t_hac'])


def test_rates_v08_funded_solver_same_weights_exact():
    rng=np.random.default_rng(17)
    ix=[f'ZC{m}Y' for m in range(2,11)]
    names=['Level','Slope','Curvature']
    B=pd.DataFrame(rng.normal(0,.2,(9,3)),index=ix,columns=names)
    u=rng.normal(0,1,(9,9));H=pd.DataFrame(u.T@u+np.eye(9),index=ix,columns=ix)
    old=legacy_functions('rates_v0_8_reference.ipynb',{'rms_standardize_columns','exact_funded_factor_weights'},
                         {'FACTOR_NAMES':names})
    for name in names:
        ref_w, achieved, ref_diag, scales=old['exact_funded_factor_weights'](B,H,name)
        current_w,current_diag=exact_funded_factor_weights(B,H,name)
        np.testing.assert_allclose(ref_w.to_numpy(),current_w.to_numpy(),rtol=1e-12,atol=1e-12)
        assert current_diag['cash_weight']==ref_diag['cash_funding_weight']
        assert current_diag['purity']==ref_diag['purity_normalized_beta']


def test_rates_v08_zc_returns_same_pricing_convention():
    rng=np.random.default_rng(18)
    dates=pd.date_range('1995-01-31',periods=120,freq='ME')
    z=pd.DataFrame(.03+np.linspace(0,.004,10)+np.cumsum(rng.normal(0,.001,(120,10)),axis=0),
                   index=dates,columns=[f'{i}Y' for i in range(1,11)])
    old=legacy_functions('rates_v0_8_reference.ipynb',{'interp_curve_yield','zero_coupon_price_cc','build_zero_coupon_return_decomposition'}, {})
    ref_r,ref_c,ref_cur=old['build_zero_coupon_return_decomposition'](z,list(range(2,11)))
    r,c=zc_return_decomposition(z)
    pd.testing.assert_frame_equal(ref_r,r,check_dtype=False)
    pd.testing.assert_frame_equal(ref_c,c,check_dtype=False)
