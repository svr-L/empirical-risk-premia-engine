# Equity Risk Premia — v0.12 research specification and audit

## Scientific question
Do residual directions not spanned by FF6+BAB+QMJ produce **incremental, stable and priced** systematic risks? Can the risk-price and residual-factor framework be linked to **identified announcement surprises**, rather than to monthly CPI, rates or volatility changes alone?

## Input and design (unchanged from v0.11)
- Model: FF6+BAB+QMJ (8 traded factors). Test assets: 130 characteristic/industry portfolios, of which **80 discovery** and **50 heldout-by-definition** (not guaranteed independent underlying stocks). Monthly excess returns, in decimal units.
- Original train through 2005-12, test Jan 2006-Aug 2026 in the prior real data snapshot. In this new code the dates are computed from the actual input data; no current OOS results are asserted.
- Factors: Kenneth French, AQR BAB & QMJ. No synthetic substitutes, original v0.11 loader + `FRED_LOCAL_PATH` convenience retained.

## 1 — Rolling FF-orthogonal, pricing-weighted residual factors
- Each formation date t-1 uses a rolling 120-month window, long/short characteristic-sorted discovery portfolios. Holding starts t and changes annually (configurable). At formation we regress discovery returns on FF/AQR with intercept, obtaining coefficient matrix `(alpha,beta)`, OLS innovations `E`, and a FF-neutral projection of portfolio weights.
- Augmented mean-pricing objective uses `M_gamma = E'E/(T-1) + gamma * alpha alpha'`. The eigensystem is equivalent to SVD of `[E/sqrt(T-1); sqrt(gamma)*alpha']`, so it **explicitly penalizes expected-return pricing error**. `gamma=0` is exactly ordinary residual PCA. **This is a partialled-FF conditional RP-PCA challenger, not the complete published Lettau–Pelger estimator or its asymptotic theory.**
- Fixed sign is chosen from the previous **known formation weights**. The observation at t can influence weights formed at t+1, never the current holding interval. All weights have zero net capital and zero training FF betas, gross one. The test returns are computed from original monthly asset excess returns, never training residual means (which equal zero).
- Diagnostic outputs include rolling subspace overlap (mean squared principal-angle cosine), factor direction cosine, rebalancing L1 turnover, FF-neutrality at formation, HAC OOS factor premia and factor-spanning alpha. Rolling returns are **not** inserted into a fixed-formation asset-pricing model's historical regressors, because that would create a methodological mismatch.

## 2 — Matched pricing comparisons: FMB vs 2-step GMM/SDF
- On **exactly the same** 50 heldout portfolios and train/OOS dates, compare: FF/AQR; FF/AQR+3 static RP factors gamma=0; FF/AQR+3 static RP factors gamma=12. Factor weights estimated in formation once then frozen for the matched horse race.
- **FMB:** time-series betas estimated only during formation. Each formation month regresses the holdout asset returns cross-sectionally on these betas plus intercept; average lambda and HAC t are reported. Forecast mean prices for heldout portfolios use formation-only `beta × lambda`. This is a conventional two-pass FMB diagnostic, with no Shanken adjustment, EIV bias correction, rank-robust inference or guarantee of traded-factor mean equivalence.
- **GMM/SDF:** `m_t=1-b'(F_t-E[F])` with factor returns standardized using formation standard deviations, `g_T=E[R_t^e m_t]`. Initial ridge identity-GMM fit, HAC/Newey–West moment covariance calculated at first-stage residuals, weighted two-step GLS with 50% shrinkage toward the diagonal. **Do not use naive chi-square J inference** with this regularization and overlapping test assets. Only formation estimates determine OOS mean prices.
- **OOS:** annualized RMSE between model-predicted mean excess returns and the *observed OOS average* of the same 50 portfolios. Separately compare realized monthly portfolio squared forecast-loss differences vs the identical FF benchmark with a paired 12-month circular block bootstrap (399 draws). These are distinct performance metrics.
- Stronger tests for future: stock-level nonoverlapping portfolios, weak-beta inference, prevalidated hyperparameter and PC rank selection, joint conditional model uncertainty and cross-market price tests.

## 3 — Better-identified macro news, not monthly proxies mislabeled as shocks
- `load_jk_monetary_shocks` imports the [Jarociński–Karadi update](https://github.com/marekjarocinski/jkshocks_update_fed) directly from the author-maintained raw CSV. Event-window interest-rate surprise (`pc1_hf`) is decomposed into `MP_median` and `CBI_median`, with a parser-level identity check. The author defines zero for no announcement in a month. **After the final author-data month, values are *missing*, not zero.** Source displayed in Nov 2026 includes through Mar 2026; the downloader checks the exact actual file dates at runtime.
- FOMC **same-month** exposure regression of original FF/AQR and rolling RPC factors on `MP_median,CBI_median`: contemporaneous economic association, not a tradeable return forecast and not automatic causal attribution. **Lag 1** applies calendar-month shifts to announcement shocks before predicting returns at t, so shocks at month-end t-1 are known at the next decision. All coefficients fit on pre-2006 common calendar, scored OOS within actual shock coverage.
- `estimate_jk_shock_prices` is a **two-pass descriptive cross-sectional** risk-price diagnostic, adding these two nontraded shocks to FF/AQR controls, fixed formation betas and common OOS heldout assets. Lambda is per unit *exposure*, NOT the mean shock. Weak beta dispersion, measurement-error bias and identification assumptions preclude treating lambda t-stat alone as proof of a structural policy risk premium.
- Optional CPI **actual−pre-release consensus** loader requires `release_date,actual,consensus,consensus_source` and consistent units. Observations indexed by **announcement month**; absent months remain absent. The parser cannot audit the original provider's announcement timestamp. No CPI-release surprises are generated from historic revised `CPIAUCSL` (not the same thing).
- The v0.11 FRED proxies still provide descriptive historical macro/cross-asset changes; they **remain separate** from this identified-shock pipeline.

## Runnable artifacts and exported files
`notebooks/Equity_Factor_Discovery_and_Pricing_v0_12_PORTABLE.ipynb` is a 26-cell portable Colab notebook, with all ERP Python source embedded. A fresh runtime needs internet for original French/AQR data (or previously cached original source files); FRED and JK have explicit local-source fallbacks. It saves:
- `equity_v012_rolling_and_pricing/rolling_weights.csv`, `rolling_returns.csv`, `rolling_diagnostics.csv`, `rolling_premia.csv`
- `equity_v012_rolling_and_pricing/pricing_comparison.csv`, `fmb_lambdas.csv`, `bootstrap_comparison.csv`, `manifest.json`
- `jk_announcement_shocks/identified_economic_shocks_monthly.csv`, optional CPI release files
- `jk_ff_rpc_exposure_and_predictability.csv`, `jk_common_sample_mean_pricing.csv`, `jk_shock_fmb_lambda_hac.csv` when JK is available
- All original baseline v0.11 reports, separately.

## Source literature
- Lettau & Pelger (2020), *Factors That Fit the Time Series and Cross-Section of Stock Returns*, **RFS**, doi:10.1093/rfs/hhaa020. This v0.12 residual adaptation is **not** claimed to reproduce their full estimator or theory.
- Giglio & Xiu (2021), *Asset Pricing with Omitted Factors*, **JPE**. Three-pass diagnostics carried forward in v0.11, no weak-factor-robust asymptotic inference.
- Jarociński & Karadi (2020), *Deconstructing Monetary Policy Surprises*, **AEJ: Macro**; monthly dataset maintained at `github.com/marekjarocinski/jkshocks_update_fed`, CC-BY 4.0 data.
- FRBSF (2026 update), *U.S. Monetary Policy Event-Study Database* for alternative high-frequency robustness checks; full USMPD integration not included in this release.

No new true-data statistical significance or P&L is claimed until this notebook runs against the original published data.
