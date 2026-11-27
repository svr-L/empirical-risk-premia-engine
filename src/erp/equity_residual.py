"""Equity residual-factor research: fixed-training PCA, factor-neutral funded portfolios,
held-out pricing and contemporaneous risk attribution.

Avoids treating a large residual eigenvalue as evidence of a priced risk. All
asset returns are monthly decimals. Loadings and weights are frozen before the
first evaluation observation; no candidate is selected using evaluation returns.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .equity import EquityData, MODELS


def _coef(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _hac_alpha(y: pd.Series, factors: pd.DataFrame | None = None, lags: int = 6) -> dict:
    """Test traded portfolio alpha; regress against existing factors or just cash."""
    if factors is None:
        val=y.dropna()
        x=np.ones((len(val),1))
    else:
        frame=factors.join(y.rename('_portfolio'),how='inner').dropna()
        val=frame['_portfolio']
        x=np.column_stack([np.ones(len(frame)),frame[factors.columns].to_numpy()])
    if len(val) <= x.shape[1]+lags+3:
        return {'alpha_monthly':np.nan,'t_hac':np.nan,'p_hac':np.nan,'nobs':len(val)}
    f=sm.OLS(val.to_numpy(),x).fit(cov_type='HAC',cov_kwds={'maxlags':min(lags,len(val)-1)})
    return {'alpha_monthly':float(f.params[0]),'t_hac':float(f.tvalues[0]),
            'p_hac':float(f.pvalues[0]),'nobs':int(f.nobs), 'r2':float(f.rsquared)}


def _holm(pvalues: pd.Series) -> pd.Series:
    """Holm-adjusted two-sided HAC p-values for predeclared PC tests."""
    ok=pvalues.dropna().astype(float).sort_values()
    result=pd.Series(np.nan,index=pvalues.index,dtype=float)
    n=len(ok); previous=0.
    for j,(key,p) in enumerate(ok.items()):
        previous=max(previous,min(1.,(n-j)*float(p)))
        result.loc[key]=previous
    return result


def _standardized_condition(b: np.ndarray) -> float:
    norms=np.maximum(np.linalg.norm(b,axis=0),1e-14)
    return float(np.linalg.cond(b/norms))


def _build_panel(data: EquityData, keys: tuple[str,...], rf: pd.Series) -> pd.DataFrame:
    arrays=[]
    for key in keys:
        if key not in data.universes:
            raise ValueError(f'Unknown universe {key}; available: {list(data.universes)}')
        arrays.append(data.universes[key].add_prefix(f'{key}__'))
    raw=pd.concat(arrays,axis=1)
    merged=raw.join(rf.rename('_RF'),how='inner')
    return merged[raw.columns].sub(merged['_RF'],axis=0)


def _metrics(y: np.ndarray, predicted: np.ndarray, baseline: np.ndarray | None = None):
    mse=float(np.mean((y-predicted)**2))
    if baseline is None:
        ss=float(np.sum((y-y.mean(axis=0))**2))
        r2=float(1-np.sum((y-predicted)**2)/ss) if ss>1e-20 else np.nan
    else:
        ss=float(np.sum((y-baseline)**2))
        r2=float(1-np.sum((y-predicted)**2)/ss) if ss>1e-20 else np.nan
    return {'rmse_monthly':float(np.sqrt(mse)),'oos_r2':r2}


@dataclass
class EquityResidualResult:
    model: str
    discovery_universes: tuple[str,...]
    validation_universes: tuple[str,...]
    train_end: pd.Timestamp
    as_of: pd.Timestamp
    factors: pd.DataFrame
    residual_eigensystem: pd.DataFrame
    residual_eigenvalues: pd.DataFrame
    weights: pd.DataFrame
    fmp_returns: pd.DataFrame
    fmp_premia_tests: pd.DataFrame
    neutrality_tests: pd.DataFrame
    holdout_metrics: pd.DataFrame
    holdout_pricing: pd.DataFrame
    sdf_gmm_comparison: pd.DataFrame
    sdf_gmm_coefficients: pd.DataFrame
    state_forecasts: pd.DataFrame
    state_scores: pd.DataFrame
    macro_exposures: pd.DataFrame
    macro_diagnostics: pd.DataFrame
    macro_sdf_pricing: pd.DataFrame
    macro_sdf_coefficients: pd.DataFrame
    metadata: dict

    def export(self, directory: str|Path) -> Path:
        p=Path(directory);p.mkdir(parents=True,exist_ok=True)
        frames={
            'residual_eigenvectors':self.residual_eigensystem,
            'residual_eigenvalues':self.residual_eigenvalues,
            'factor_neutral_weights':self.weights,
            'realized_fmp_returns':self.fmp_returns,
            'incremental_premia_tests':self.fmp_premia_tests,
            'factor_neutrality_tests':self.neutrality_tests,
            'holdout_oos_attribution':self.holdout_metrics,
            'holdout_oos_pricing':self.holdout_pricing,
            'sdf_gmm_comparison':self.sdf_gmm_comparison,
            'sdf_gmm_coefficients':self.sdf_gmm_coefficients,
            'lagged_state_forecasts':self.state_forecasts,
            'lagged_state_scores':self.state_scores,
            'macro_exposures':self.macro_exposures,
            'macro_diagnostics':self.macro_diagnostics,
            'macro_sdf_pricing':self.macro_sdf_pricing,
            'macro_sdf_coefficients':self.macro_sdf_coefficients,
        }
        for name,frame in frames.items():
            frame.to_csv(p/(name+'.csv'),index_label='Date' if isinstance(frame.index,pd.DatetimeIndex) else None)
        info=dict(self.metadata)
        info.update({'model':self.model,'train_end':str(self.train_end.date()),
                     'as_of':str(self.as_of.date()),'discovery_universes':list(self.discovery_universes),
                     'validation_universes':list(self.validation_universes),
                     'files':[name+'.csv' for name in frames],
                     'definitions':{'PC':'training OLS residual covariance PCA; not an economic premium',
                       'FMP':'self-financing portfolio of characteristic-sorted test portfolios, gross 1, benchmark beta neutral IN TRAIN ONLY',
                       'pricing':'OOS factor portfolio alpha vs FF/AQR and held-out mean-pricing error; not a test of priced SDF',
                       'attribution':'OOS realized factor return attribution with pretest loadings, NOT conditional return forecasting',
                       'macro':'contemporaneous descriptive exposure only, NOT macro risk prices or causal identification',
                       'sdf_gmm':'one-step equal-weight GMM linear SDF m=1-b_(F-Fmean) estimated on training holdouts; OOS mean error, not efficient two-step GMM',
                       'lagged_states':'expanding OOS forecast with lagged observable state and train-only ridge; not point-in-time historical macro vintages',
                       'macro_sdf':'OPTIONAL one-step linear SDF/GMM with nontraded shocks, estimated on training test asset cross-section, evaluated OOS; identification fragile without independent valid macro shocks'}})
        (p/'manifest.json').write_text(json.dumps(info,indent=2,default=str),encoding='utf-8')
        return p


def analyze_equity_residuals(
    data: EquityData,
    *,model: str='FF6_BAB_QMJ',
    discovery_universes: tuple[str,...]=('SIZE_BM_25','INDUSTRY_30','SIZE_MOM_25'),
    validation_universes: tuple[str,...]=('SIZE_OP_25','SIZE_INV_25'),
    train_end: str='2005-12-31',as_of: str|None='2026-10-31',
    n_pcs:int=5,hac_lags:int=6,
    macro_shocks:pd.DataFrame|None=None,
    sdf_ridge:float=1e-4,
    state_predictors:pd.DataFrame|None=None,
    state_lag_months:int=2,
    state_ridge:float=10.,
) -> EquityResidualResult:
    """Fixed formation-period discovery, then temporally held-out and universe-held-out tests.

    ``macro_shocks`` is optional, user-supplied contemporaneous economic *shocks*
    (not lagged CPI levels). Its mapping is diagnostic and never infers risk prices.

    Training residuals are demeaned by OLS intercept. A PC score's in-sample
    mean is therefore ZERO by construction. All premium tests instead use
    self-financing raw-return mimicking portfolios on the strictly future sample.
    """
    if model not in MODELS: raise ValueError(f'Unknown model {model}')
    if sdf_ridge < 0:raise ValueError('sdf_ridge must be nonnegative and predeclared')
    if state_lag_months<1 or state_ridge<0:
        raise ValueError('State predictors must use a >=1 month lag and nonnegative predeclared ridge')
    disc=tuple(discovery_universes); valid=tuple(validation_universes)
    if not disc or not valid or (set(disc)&set(valid)):
        raise ValueError('Discovery and validation universes must be non-empty and disjoint')
    if not isinstance(n_pcs,int) or not 1<=n_pcs<=20:
        raise ValueError('n_pcs must be 1..20 and chosen before viewing OOS results')
    train_end=pd.Timestamp(train_end)
    f,rf=data.factors(model)
    last=min(f.index.max(),rf.index.max())
    cutoff=pd.Timestamp(as_of) if as_of is not None else last
    cutoff=min(cutoff,last)
    if train_end>=cutoff:raise ValueError('Training must end strictly before as_of')
    f=f.loc[:cutoff]
    discovery=_build_panel(data,disc,rf).loc[:cutoff]
    holdout=_build_panel(data,valid,rf).loc[:cutoff]
    all_data=pd.concat([f,discovery,holdout],axis=1,join='inner').dropna()
    fs=f.columns.to_list(); dc=discovery.columns.to_list(); hc=holdout.columns.to_list()
    if len(set(all_data.columns)) != all_data.shape[1]:raise ValueError('Duplicated column names')
    ft=all_data[fs]; d=all_data[dc]; h=all_data[hc]
    is_train=ft.index<=train_end
    if int(is_train.sum())<max(96,5*len(fs)) or int((~is_train).sum())<36:
        raise ValueError('Need >=96 common training months (>=5 x factor count) and >=36 OOS months')
    if d.shape[1]<=len(fs)+n_pcs+1:
        raise ValueError('Insufficient discovery assets to fit neutral residual factors')
    Ftrain=ft.loc[is_train].to_numpy(); Dtrain=d.loc[is_train].to_numpy()
    X=np.column_stack([np.ones(len(Ftrain)),Ftrain])
    b=_coef(X,Dtrain)
    E=Dtrain-X@b
    if not np.allclose(X.T@E,0,atol=1e-7):
        raise RuntimeError('Training residual OLS orthogonality failed')
    # Eigenvectors of asset x asset residual covariance via thin SVD. 
    # PC direction sign is deterministic on FORMATION ONLY (largest abs loading positive).
    _,singular,Vt=np.linalg.svd(E,full_matrices=False)
    vec=Vt[:n_pcs].T.copy()
    eigen=singular**2/max(1,len(E)-1)
    var_frac=eigen/np.sum(eigen)
    for k in range(n_pcs):
        v=vec[:,k]; idx=np.argmax(np.abs(v))
        if v[idx]<0:vec[:,k]*=-1
    eigvec=pd.DataFrame(vec,index=dc,columns=[f'RPC{k+1}' for k in range(n_pcs)])
    eigenvalues=pd.DataFrame({'eigenvalue':eigen[:n_pcs],
                              'residual_variance_share':var_frac[:n_pcs],
                              'cumulative_share':np.cumsum(var_frac[:n_pcs])},index=eigvec.columns)
    # Dollar neutrality, model-factor neutrality of RAW traded portfolio returns
    # using train-only asset beta estimates. Projection onto null of [1,B].
    constraints=np.column_stack([np.ones(len(dc)),b[1:,:].T])
    if np.linalg.matrix_rank(constraints) != constraints.shape[1]:
        raise ValueError('Training factor beta matrix rank deficient: FMP isolation not identified')
    proj=np.eye(len(dc))-constraints@np.linalg.pinv(constraints)
    W=[]
    for col in eigvec:
        w=proj@eigvec[col].to_numpy()
        if np.sum(np.abs(w))<1e-8:
            raise ValueError(f'{col} lies too close to benchmark factor span')
        w=w/np.sum(np.abs(w))
        # The sign follows the residual PCA direction: do not orient to OOS mean.
        W.append(w)
    W=np.column_stack(W)
    weights=pd.DataFrame(W,index=dc,columns=eigvec.columns)
    overhang=constraints.T@W
    neutrality=pd.DataFrame(overhang,index=['net_dollar_weight',*fs],columns=eigvec.columns).T
    neutrality['gross_exposure']=weights.abs().sum(axis=0).to_numpy()
    neutrality['max_abs_factor_beta_train']=neutrality[fs].abs().max(axis=1)
    # Frozen factor portfolio returns; FMPs ARE raw-return investments, not OLS residual scores.
    full_fmp=pd.DataFrame(d.to_numpy()@W,index=d.index,columns=eigvec.columns)
    Ftrain_df=ft.loc[is_train]; Ftest_df=ft.loc[~is_train]
    pr_rows=[]
    for pc in eigvec:
        tr=_hac_alpha(full_fmp.loc[is_train,pc],Ftrain_df,hac_lags)
        ts=_hac_alpha(full_fmp.loc[~is_train,pc],Ftest_df,hac_lags)
        mean=_hac_alpha(full_fmp.loc[~is_train,pc],None,hac_lags)
        test_betas=_coef(np.column_stack([np.ones(len(Ftest_df)),Ftest_df.to_numpy()]),
                         full_fmp.loc[~is_train,pc].to_numpy())[1:]
        pr_rows.append({'factor':pc,'train_spanning_alpha_ann':tr['alpha_monthly']*12,
                        'oos_spanning_alpha_ann':ts['alpha_monthly']*12,
                        'oos_spanning_t_hac':ts['t_hac'],'oos_spanning_p_hac':ts['p_hac'],
                        'oos_spanning_r2':ts['r2'],
                        'oos_mean_ann':mean['alpha_monthly']*12,'oos_mean_t_hac':mean['t_hac'],
                        'oos_mean_p_hac':mean['p_hac'],
                        'oos_empirical_factor_beta_max':float(np.max(np.abs(test_betas))),
                        'n_oos_months':ts['nobs'],
                        'eigenvalue_share_train':float(eigenvalues.loc[pc,'residual_variance_share'])})
    premia=pd.DataFrame(pr_rows).set_index('factor')
    premia['oos_spanning_p_holm']=_holm(premia['oos_spanning_p_hac'])
    premia['oos_mean_p_holm']=_holm(premia['oos_mean_p_hac'])

    # External universe validation. Never re-estimate any beta on evaluation period.
    hold_metrics=[]; hold_prices=[]; sdf_rows=[]; sdf_coef_rows=[]
    used_h= h.loc[is_train].to_numpy(); future_h=h.loc[~is_train].to_numpy()
    all_pred_train=pd.concat([Ftrain_df,full_fmp.loc[is_train]],axis=1)
    all_pred_test=pd.concat([Ftest_df,full_fmp.loc[~is_train]],axis=1)
    x0tr=np.column_stack([np.ones(int(is_train.sum())),Ftrain_df.to_numpy()])
    x0te=np.column_stack([np.ones(int((~is_train).sum())),Ftest_df.to_numpy()])
    b0=_coef(x0tr,used_h)
    y0=x0te@b0
    mu0_noalpha=Ftrain_df.mean().to_numpy()@b0[1:,:]
    mu0_realized=Ftest_df.mean().to_numpy()@b0[1:,:]
    for k in range(n_pcs+1):
        cols=fs+list(eigvec.columns[:k])
        atr=all_pred_train[cols]; ate=all_pred_test[cols]
        xtr=np.column_stack([np.ones(len(atr)),atr.to_numpy()])
        xte=np.column_stack([np.ones(len(ate)),ate.to_numpy()])
        beta=_coef(xtr,used_h)
        pred=xte@beta
        stat=_metrics(future_h,pred,baseline=y0)
        # A separate risk attribution statistic, using realized contemporaneous factor moves.
        ss=float(((future_h-future_h.mean(axis=0))**2).sum())
        explained=1-float(((future_h-pred)**2).sum())/ss if ss>1e-20 else np.nan
        # Non-alpha pricing forecasts: no fitted alpha, no future factor means.
        train_factor_mean=atr.mean().to_numpy()
        risk_mean=(train_factor_mean@beta[1:,:])
        realized_risk_mean=(ate.mean().to_numpy()@beta[1:,:])
        actual_mean=future_h.mean(axis=0)
        hold_metrics.append({'n_new_pcs':k,'model':'FF_ONLY' if k==0 else f'FF_PLUS_{k}_RPC',
                             'n_heldout_assets':len(hc),'n_test_months':len(ate),
                             'oos_rmse_monthly':stat['rmse_monthly'],
                             'oos_r2_vs_ff_baseline':stat['oos_r2'],
                             'oos_realized_risk_attribution_r2':explained,
                             'pricing_rmse_train_factor_means_ann':float(12*np.sqrt(np.mean((actual_mean-risk_mean)**2))),
                             'pricing_rmse_realized_factor_means_ann':float(12*np.sqrt(np.mean((actual_mean-realized_risk_mean)**2)))})
        # One-step linear SDF / identity-weighted GMM over held-out test assets.
        # m = 1 - b'(f-E_train f), so E_train[m R_i] = mu_i - Cov(R_i,f)b.
        # Fit b only in formation period; score predictions vs FUTURE realized
        # held-out mean. This is not efficient two-step GMM nor proof of pricing.
        factor_train=atr.to_numpy()
        mu_asset=used_h.mean(axis=0)
        demean_asset=used_h-mu_asset
        demean_fac=factor_train-factor_train.mean(axis=0)
        cov_asset_fac=(demean_asset.T@demean_fac)/max(1,len(used_h)-1)
        norms=np.maximum(np.linalg.norm(cov_asset_fac,axis=0),1e-14)
        c_std=cov_asset_fac/norms
        b_std=np.linalg.solve(c_std.T@c_std+sdf_ridge*np.eye(len(cols)),c_std.T@mu_asset)
        b_sdf=b_std/norms
        price_pred=cov_asset_fac@b_sdf
        risk_prices=np.cov(factor_train,rowvar=False,ddof=1)@b_sdf
        sdf_rows.append({'n_new_pcs':k,'n_moments':len(hc),'n_sdf_factors':len(cols),
                         'pricing_rmse_train_ann':float(12*np.sqrt(np.mean((mu_asset-price_pred)**2))),
                         'pricing_rmse_oos_ann':float(12*np.sqrt(np.mean((actual_mean-price_pred)**2))),
                         'pricing_mae_oos_ann':float(12*np.abs(actual_mean-price_pred).mean()),
                         'covariance_condition_scaled':_standardized_condition(cov_asset_fac),
                         'ridge_relative':float(sdf_ridge)})
        for j,facname in enumerate(cols):
            sdf_coef_rows.append({'n_new_pcs':k,'factor':facname,
                                  'sdf_price_coefficient':float(b_sdf[j]),
                                  'implied_lambda_monthly':float(risk_prices[j]),
                                  'training_factor_mean_monthly':float(train_factor_mean[j])})
        for j,asset in enumerate(hc):
            hold_prices.append({'n_new_pcs':k,'asset':asset,'mean_oos_realized_ann':float(actual_mean[j]*12),
                                'expected_using_training_means_ann':float(risk_mean[j]*12),
                                'pricing_error_training_means_ann':float((actual_mean[j]-risk_mean[j])*12),
                                'pricing_error_realized_factor_means_ann':float((actual_mean[j]-realized_risk_mean[j])*12)})
    hold_metric_frame=pd.DataFrame(hold_metrics).set_index('n_new_pcs')
    hold_pricing_frame=pd.DataFrame(hold_prices).set_index(['n_new_pcs','asset'])
    sdf_comparison=pd.DataFrame(sdf_rows).set_index('n_new_pcs')
    sdf_coefficients=pd.DataFrame(sdf_coef_rows).set_index(['n_new_pcs','factor'])

    # Optional conditional expected return research. All state values are lagged
    # by a user-declared number of calendar MONTHLY observations. Historic revised
    # data are not a publication-vintage reconstruction.
    state_fc=pd.DataFrame();state_scores=pd.DataFrame()
    if state_predictors is not None:
        if not isinstance(state_predictors,pd.DataFrame) or state_predictors.empty:
            raise ValueError('state_predictors must be a nonempty numeric monthly DataFrame')
        z=state_predictors.copy()
        z.index=pd.to_datetime(z.index)+pd.offsets.MonthEnd(0)
        z=z.sort_index()
        if z.index.has_duplicates or z.columns.has_duplicates:
            raise ValueError('state predictors require unique dates and column names')
        if not all(pd.api.types.is_numeric_dtype(z[c]) for c in z.columns):
            raise ValueError('state_predictors must contain numeric series')
        full_months=pd.date_range(ft.index.min(),ft.index.max(),freq="ME")
        z=z.reindex(full_months).shift(state_lag_months).reindex(ft.index)
        target_panel=pd.concat([ft,full_fmp],axis=1)
        conditional=pd.concat([z.add_prefix('STATE__'),target_panel],axis=1)
        predictor_cols=['STATE__'+col for col in z.columns]
        target_cols=list(target_panel.columns)
        forecast_rows=[]
        for date in ft.loc[~is_train].index:
            prior=conditional.loc[conditional.index<date].dropna()
            obs=conditional.loc[date]
            if len(prior)<max(72,5*len(predictor_cols)) or obs.isna().any():
                continue
            xx=prior[predictor_cols].to_numpy()
            yy=prior[target_cols].to_numpy()
            mean=xx.mean(axis=0)
            scale=np.maximum(xx.std(axis=0,ddof=1),1e-12)
            xnorm=(xx-mean)/scale
            xp=np.column_stack([np.ones(len(xx)),xnorm])
            pen=np.eye(xp.shape[1])*state_ridge
            pen[0,0]=0
            coef=np.linalg.solve(xp.T@xp+pen,xp.T@yy)
            current=(obs[predictor_cols].to_numpy(dtype=float)-mean)/scale
            pred=np.r_[1,current]@coef
            average=yy.mean(axis=0)
            actual=obs[target_cols].to_numpy(dtype=float)
            for j,fname in enumerate(target_cols):
                forecast_rows.append({'Date':date,'factor':fname,
                                      'actual_excess_return':float(actual[j]),
                                      'conditional_forecast':float(pred[j]),
                                      'expanding_unconditional_benchmark':float(average[j]),
                                      'last_state_observation_month':str((date-pd.DateOffset(months=state_lag_months)).to_period('M'))})
        if forecast_rows:
            state_fc=pd.DataFrame(forecast_rows).set_index(['Date','factor']).sort_index()
            sr=[]
            for fac,grp in state_fc.reset_index().groupby('factor',sort=False):
                target=grp['actual_excess_return'].to_numpy()
                cond=grp['conditional_forecast'].to_numpy()
                bench=grp['expanding_unconditional_benchmark'].to_numpy()
                mse_b=float(np.mean((target-bench)**2))
                gain=(target-bench)**2-(target-cond)**2
                test=_hac_alpha(pd.Series(gain),None,hac_lags)
                sr.append({'factor':fac,'n_oos':len(grp),
                           'oos_r2_vs_expanding_mean':float(1-np.mean((target-cond)**2)/mse_b) if mse_b>1e-20 else np.nan,
                           'loss_gain_monthly_sq':float(np.mean(gain)),
                           'loss_gain_hac_t':test['t_hac'],
                           'loss_gain_hac_p':test['p_hac'],
                           'state_lag_months':state_lag_months,
                           'state_ridge':state_ridge})
            state_scores=pd.DataFrame(sr).set_index('factor')

    # Economic interpretation is diagnostic: do not treat macro OLS coefficients
    # as prices of nontraded risk; no macro factors are fabricated.
    macro_exp=pd.DataFrame();macro_diag=pd.DataFrame()
    macro_sdf_price=pd.DataFrame();macro_sdf_coef=pd.DataFrame()
    if macro_shocks is not None:
        if not isinstance(macro_shocks,pd.DataFrame) or macro_shocks.empty:
            raise ValueError('macro_shocks must be a nonempty DataFrame with economic innovations')
        macro=macro_shocks.copy()
        macro.index=pd.to_datetime(macro.index)+pd.offsets.MonthEnd(0)
        macro=macro.sort_index().loc[:cutoff]
        if macro.columns.has_duplicates or len(macro.columns)==0:
            raise ValueError('Invalid or repeated macro factor names')
        if not all(pd.api.types.is_numeric_dtype(macro[c]) for c in macro.columns):
            raise ValueError('macro_shocks should contain numeric innovations, not raw text')
        econ=pd.concat([ft,full_fmp],axis=1)
        c=econ.join(macro,how='inner').dropna()
        mcols=list(macro.columns)
        ct=c.index<=train_end
        if int(ct.sum())<max(48,4*len(mcols)) or int((~ct).sum())<24:
            raise ValueError('Insufficient contemporaneous macro data in both training and OOS')
        xm=np.column_stack([np.ones(int(ct.sum())),c.loc[ct,mcols].to_numpy()])
        xx=np.column_stack([np.ones(int((~ct).sum())),c.loc[~ct,mcols].to_numpy()])
        ys=c.loc[ct,econ.columns].to_numpy();yt=c.loc[~ct,econ.columns].to_numpy()
        coefs=_coef(xm,ys)
        fitted=xx@coefs
        baseline=np.repeat(ys.mean(axis=0)[None,:],len(xx),axis=0)
        for j,name in enumerate(econ.columns):
            macro_diag.loc[name,'oos_macro_r2_vs_train_mean']=_metrics(yt[:,j:j+1],fitted[:,j:j+1],baseline[:,j:j+1])['oos_r2']
            macro_diag.loc[name,'oos_macro_residual_mean_ann']=float(12*np.mean(yt[:,j]-fitted[:,j]))
            macro_diag.loc[name,'n_macro_train_months']=int(ct.sum())
            macro_diag.loc[name,'n_macro_oos_months']=int((~ct).sum())
        macro_exp=pd.DataFrame(coefs[1:,:].T,index=econ.columns,columns=mcols)
        macro_exp.index.name='equity_factor_or_rpc'
        # Optional nontraded macro factor *price-of-risk* estimate, distinct from
        # the OLS exposure/interpretation above: m=1-b'(g-Etrain[g]).
        # Estimate b from formation cross-sectional excess returns of VALIDATION
        # portfolios, not from FMP returns, then evaluate held-out months.
        # A significant OOS fit would still need weak-ID/measurement/vintage checks.
        stacked=c.join(h.add_prefix('TEST__'),how='inner').dropna()
        mt=stacked.index<=train_end
        hg=stacked.loc[mt,['TEST__'+x for x in hc]].to_numpy()
        hgo=stacked.loc[~mt,['TEST__'+x for x in hc]].to_numpy()
        ztrain=stacked.loc[mt,mcols].to_numpy()
        gmean=ztrain.mean(axis=0)
        d_h=hg-hg.mean(axis=0)
        d_g=ztrain-gmean
        C=(d_h.T@d_g)/max(1,len(hg)-1)
        colnorm=np.maximum(np.linalg.norm(C,axis=0),1e-14)
        Cscaled=C/colnorm
        bscaled=np.linalg.solve(Cscaled.T@Cscaled+sdf_ridge*np.eye(len(mcols)),
                                 Cscaled.T@hg.mean(axis=0))
        bmacro=bscaled/colnorm
        lambda_macro=np.cov(ztrain,rowvar=False,ddof=1)@bmacro if len(mcols)>1 else np.asarray([float(np.var(ztrain[:,0],ddof=1)*bmacro[0])])
        facname=list(econ.columns)
        factor_train=stacked.loc[mt,facname].to_numpy()
        factor_test=stacked.loc[~mt,facname].to_numpy()
        Cf=((factor_train-factor_train.mean(axis=0)).T@d_g)/max(1,len(hg)-1)
        predicted=Cf@bmacro
        macro_sdf_price=pd.DataFrame({'macro_implied_premium_ann':12*predicted,
             'realized_factor_mean_ann_oos':12*factor_test.mean(axis=0),
             'macro_unexplained_mean_ann_oos':12*(factor_test.mean(axis=0)-predicted)},index=facname)
        macro_sdf_price.index.name='equity_factor_or_rpc'
        macro_sdf_price.attrs.clear()
        macro_sdf_price['validation_pricing_rmse_train_ann']=float(12*np.sqrt(np.mean((hg.mean(axis=0)-C@bmacro)**2)))
        macro_sdf_price['validation_pricing_rmse_oos_ann']=float(12*np.sqrt(np.mean((hgo.mean(axis=0)-C@bmacro)**2)))
        macro_sdf_price['macro_cs_condition_scaled']=_standardized_condition(C)
        macro_sdf_coef=pd.DataFrame({'sdf_price_coefficient':bmacro,
                                   'macro_risk_price_per_unit_monthly':lambda_macro},index=mcols)
        macro_sdf_coef.index.name='macro_shock'


    return EquityResidualResult(model=model,discovery_universes=disc,validation_universes=valid,
           train_end=train_end,as_of=ft.index.max(),factors=ft,residual_eigensystem=eigvec,
           residual_eigenvalues=eigenvalues,weights=weights,fmp_returns=full_fmp,
           fmp_premia_tests=premia,neutrality_tests=neutrality,
           holdout_metrics=hold_metric_frame,holdout_pricing=hold_pricing_frame,
           sdf_gmm_comparison=sdf_comparison,sdf_gmm_coefficients=sdf_coefficients,
           state_forecasts=state_fc,state_scores=state_scores,
           macro_exposures=macro_exp,macro_diagnostics=macro_diag,
           macro_sdf_pricing=macro_sdf_price,macro_sdf_coefficients=macro_sdf_coef,
           metadata={'version':'0.10.0','formation_months':int(is_train.sum()),
              'oos_months':int((~is_train).sum()),
              'train_start':str(ft.index.min().date()),'oos_start':str(ft.loc[~is_train].index.min().date()),
              'training_residual_share_of_total_centered_variance':float(np.sum(E**2)/max(np.sum((Dtrain-Dtrain.mean(axis=0))**2),1e-20)),
              'source':'French factor/portfolio returns + published AQR factors',
              'universe_overlap_warning':'Sorted/industry portfolios can share underlying stocks despite separate portfolio definitions',
              'pca_formation_only':True,'weights_reestimated_oos':False,
              'factor_model':'OLS residualization with intercept on formation window; PCA of residual covariance',
              'direction_selection':'training eigenvector largest magnitude loading positive; never OOS premium optimized',
              'fmp_weights':'gross=1, net=0, train benchmark betas=0; cost/short-borrow not modeled',
              'multiple_testing':'Holm-adjusted p for standalone OOS PC mean and alpha tests; exploratory',
              'macro':'optional economic exposures and tentative nontraded macro SDF/GMM prices; never causal identification or validated market risk prices',
              'sdf_method':'one-step identity-weighted linear SDF/GMM trained on holdout assets pre-OOS with predeclared relative ridge',
              'state_forecasts':'optional expanding conditional forecasts on lagged state predictors; revised levels not PIT; no macro vintages reconstruction',
              'warning':'vendor history revised; not point-in-time vintages; no claim that OOS realization contemporaneous risk attribution is forecasting'})
