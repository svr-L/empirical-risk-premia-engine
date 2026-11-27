"""Original notebook source parity + deterministic live calculation parity and as_of checks."""
import ast
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from erp.rates import RatesData, RatesPremiaEngine
from erp import rates_research as rr

HERE=Path(__file__).resolve().parents[1]


def toy(months=216):
    rng=np.random.default_rng(133)
    idx=pd.date_range('1991-01-31',periods=months,freq='ME')
    level=.035+np.cumsum(rng.normal(0,.0007,months))
    slope=.008+np.cumsum(rng.normal(0,.0003,months))
    curve=pd.DataFrame({f'{n}Y':level+.001*n+slope*(n-5)/10 for n in range(1,11)},index=idx)
    funding=pd.Series(.02+.004*np.sin(np.arange(months)/13),index=idx)
    macro=pd.DataFrame({'CPI':150*np.exp(.002*np.arange(months))*(1+.001*np.sin(np.arange(months)/7)),
        'INDPRO':80*np.exp(.001*np.arange(months))*(1+.005*np.sin(np.arange(months)/8)),
        'BREAKEVEN10Y':2+.15*np.sin(np.arange(months)/9)},index=idx)
    return RatesData(curve,funding),macro


def fitted():
    d,macro=toy()
    rates=RatesPremiaEngine(d)
    snapshot=rates.fit(as_of='2008-12-31',pca_train_end='2002-12-31')
    return rates,snapshot,macro


def test_source_functions_exactly_original():
    orig=json.loads((HERE/'legacy/rates_v0_8_reference.ipynb').read_text())
    original={}
    for cell in orig['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef):
                original[node.name]=ast.get_source_segment(source,node).strip()
    source=(HERE/'src/erp/rates_research.py').read_text()
    exported={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in original:
            exported[node.name]=ast.get_source_segment(source,node).strip()
    names={'signed_factor_signal','trailing_factor_vol','capped_signed_normalize',
      'build_factor_allocations','combine_factor_and_maturity_weights','apply_overlay_next_month',
      'max_drawdown','performance_metrics','build_forward_rate_panel',
      'build_horizon_zero_coupon_data','build_horizon_factor_targets',
      'fama_bliss_univariate_diagnostics','fit_standardized_ridge_predictor',
      'fit_cp_first_stage','realization_cutoff_date','model_predictor_training_block',
      'walk_forward_empirical_premia_model','build_horizon_unconditional_forecast',
      'build_horizon_carry_only_forecast','predictive_diagnostics','interp_curve_yield',
      'zero_coupon_price_cc'}
    assert all(exported[n]==original[n] for n in names)


def test_cash_curve_factor_identity_and_horizon_timing():
    rates,snapshot,_=fitted()
    assert list(rates.mimic_weights_panel)==['Level','Slope','Curvature']
    for f in rr.FACTOR_NAMES:
        formation=rates.mimic_weights_panel[f]
        assert formation.index.max() < rates.zero_curve_sample.index.max()
    z=rates.zero_curve_sample
    c=rates.cash_returns.reindex(z.index)
    m=rates.data.fedfunds.reindex(z.index)/12
    h3=rr.build_horizon_zero_coupon_data(z,list(range(2,11)),c,m,3)
    t=rr.build_horizon_factor_targets(rates.mimic_weights_panel,h3)
    assert np.allclose((t['carry']+t['curve']-t['total']).dropna().to_numpy(),0,atol=1e-12)
    dt=t['curve'].dropna().index[25]
    assert rr.realization_cutoff_date(dt,3,z.index)==z.index[z.index.get_loc(dt)-3]
    assert h3['excess'].index.max()==z.index[-4]
    assert snapshot.as_of==pd.Timestamp('2008-12-31')


def test_full_five_family_grid_and_exports(tmp_path):
    r,ss,macro=fitted()
    res=rr.RatesEmpiricalBondPremiaEngine(r).fit(ss,macro_raw=macro,horizons=(3,),min_train=12,cost_sensitivity=(5,))
    assert set(rr.MODEL_FAMILIES)==set(x[1] for x in res.forecasts)
    assert (3,'UNCONDITIONAL') in res.backtests
    assert (3,'CARRY_ONLY') in res.backtests
    assert np.isfinite(res.strategy_results['CAGR']).all()
    assert set(res.conditional_value_add.columns)>={'spread_t_stat','spread_t_hac'}
    out=res.export(tmp_path)
    assert (out/'rates_v08_manifest.json').is_file()
    assert (out/'rates_v08_cost_sensitivity.csv').is_file()
    assert (out/'rates_v08_forecast_03m_FB.csv').is_file()
    assert res.metadata['models_skipped_for_missing_macro']==[]


def test_missing_macro_does_not_fabricate_state():
    r,ss,_=fitted()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res=rr.RatesEmpiricalBondPremiaEngine(r).fit(ss,horizons=(3,),min_train=12,cost_sensitivity=())
    assert set(x[1] for x in res.forecasts)=={'FB','CP'}
    assert set(res.metadata['models_skipped_for_missing_macro'])=={'BASE_STATE','FB_STATE','CP_STATE'}
    with pytest.raises(ValueError,match='macro_raw'):
        rr.RatesEmpiricalBondPremiaEngine(r).fit(ss,horizons=(3,),min_train=12,require_full_grid=True)


def test_asof_cannot_see_future_curve():
    d,macro=toy()
    r1=RatesPremiaEngine(d);s1=r1.fit(as_of='2008-12-31',pca_train_end='2002-12-31')
    d2=RatesData(d.zero_curve.copy(),d.fedfunds.copy())
    mask=d2.zero_curve.index>pd.Timestamp('2008-12-31')
    d2.zero_curve.loc[mask]*=3
    d2.fedfunds.loc[mask]*=2
    r2=RatesPremiaEngine(d2);s2=r2.fit(as_of='2008-12-31',pca_train_end='2002-12-31')
    pd.testing.assert_frame_equal(s1.factors,s2.factors)
    pd.testing.assert_frame_equal(r1.mimic_weights_panel['Level'],r2.mimic_weights_panel['Level'])
