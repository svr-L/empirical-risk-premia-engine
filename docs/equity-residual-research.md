> Historical methodology note. Version-specific notebook filenames refer to earlier development releases; the consolidated executable is `notebooks/01_equity_factor_discovery_and_pricing.ipynb`.

# Equity — Residual-factor research v0.10 (implemented)

## Research questions

1. Does residual common equity risk remain after FF6+BAB+QMJ?
2. Does a predeclared subset of residual PCs correspond to incremental traded factor premia **out of sample**, rather than merely principal variance directions?
3. Does augmenting standard FF/AQR with these factors improve time-separated factor attribution or cross-sectional asset pricing on distinct test-portfolio families?
4. Can contemporaneous economic shocks **interpret** both existing and residual factor returns? Do lagged, genuinely observable economic states forecast subsequent factor returns?

These are **independent questions**. Explained covariance, factor alpha, cross-sectional pricing and conditional expected-return forecasting are not equivalent.

## Version and implementation

Built upon **ERP v0.9.0** without changing Equity's original `equity.py`, Rates `rates_v09.py`, any public factor definitions or benchmark notebook estimators. New source: `src/erp/equity_residual.py`; new notebook `Equity_Residual_Factor_Research_v0_10_PORTABLE.ipynb` (fully embedded, no installation), plus `Equity_Premia_v0_10_PORTABLE.ipynb` to retain the old model comparison notebook on the same source snapshot.

Default samples: **formation** observations through 2005-12; **evaluation** from 2006-01 through min(2026-10, last official factor/test-portfolio month). The *exact* data spans are reported from the run and may be shorter. Defaults: 25 Size×B/M + 30 Industries + 25 Size×Momentum for discovery (80 portfolio definitions); 25 Size×Profitability + 25 Size×Investment for validation (50 definitions). These sorted portfolio universes are **not independent at constituent-stock level**; stock-level confirmation later.

### 1. Discovery

For training excess-return matrix `R_d` (T×N) and FF/AQR factors `F` (T×K), estimate `R_d=1 a'+F B'+E` using OLS with intercept. Train PCA on `Cov(E)` only. `E` is zero-mean in training by construction. `v_k` are residual-PC asset-space eigenvectors; their selection and signs depend only on formation data. No OOS period influences PCA or the count `K`.

### 2. Trading return portfoliios with neutrality

To avoid using zero-mean residual *scores* as if traded factor premia, construct weights on the **raw constituent test-portfolio returns**. Let `C=[1,B]` and `P=I-C(C'C)^+ C'`. For `k`, `w_k=P v_k / ||P v_k||_1`.

- `sum(w)=0` (self-financing), `||w||_1=1` (gross normalization), `B' w = 0` to training precision.
- **Unconditionally timed:** hold `w` fixed throughout the evaluation period; no reinvestment, transaction costs, borrow fees, gross leverage constraints beyond nominal gross normalization or market-impact assumptions. These are **synthetic factor portfolios of published sorted test portfolios**, not guaranteed to be directly tradable.
- Report **actual post-training beta drift**: train beta neutrality does **not** guarantee OOS neutrality.
- Factor sign is chosen by largest-magnitude PC training loading, not by profit / t-stat.

### 3. Premia tests (OOS only)

Calculate each factor portfolio return `r^RPC_t=w'R_{d,t}` in strictly future months. Estimate both:

- unconditional monthly mean and HAC significance;
- time-series alpha over FF/AQR and HAC significance (incremental traded factor spanning).

Holm adjustment over the **predeclared** `K` factor tests. A significant factor portfolio alpha alone does **not** prove a priced systematic risk in the asset cross-section; data mining, factor identification and economic interpretation must be addressed separately.

### 4. Held-out asset pricing and risk attribution

Fit baseline `FF/AQR` vs `FF/AQR + first K RPC` exposure regressions on the formation period of *separate test portfolios*. Freeze exposures, score realized return attribution on future observations; this is **ex post factor attribution conditional on realized factors**, not forecasting expected returns. Also report time-separated mean-pricing errors using `train beta × train factor mean` (true predictive expected-mean baseline), and `train beta × OOS realized factor mean` (retrospective pricing attribution). Never optimize K on these scores.

### 5. Linear SDF/GMM challenger

In each model take `f` (traded FF/AQR and/or RPCs), form centered **training** factor return series and **training** test-asset moment matrix `C=Cov(R_test,f)`. Estimate one-step SDF `m=1-b'(f-E_train f)` by weighted moment matching `min_b ||E_train R_test - C b||²` plus **predeclared relative ridge** (default 1e-4). Factor normalizations avoid raw scale penalty dominance. Reconstruct **formation** expected returns `C b` and compare against *future* actual mean returns of validation assets, with no recalibration on future months. Report train and OOS cross-sectional RMSE, condition number, SDF coefficients and `Σ_f b` implied prices.

This is **one-step identity-weighted GMM**, not efficient two-step heteroskedasticity/autocorrelation-robust GMM, not Hansen–Jagannathan and not the Giglio–Xiu three-pass. SDF test only shows a first pricing-oriented challenge, not a completed risk-premium identification analysis.

### 6. Optional economic mapping (not automatic)

When supplied with genuine monthly economic innovations, regress BOTH FF/AQR factors and raw RPC FMP returns on contemporaneous economic shocks, fit coefficients on formation months and report OOS explanatory R² and mean residual. **Separate from that OLS mapping**, fit a tentative one-step SDF/GMM model to the observed economic shocks using the separate validation-asset training cross-section, report macro prices of risk, implied FF/RPC factor premia and train/future pricing error. This is a pricing-based *challenger*, **not** a validated or uniquely identified price of nontraded macro risk; no efficient GMM, shock causal identification, point-in-time vintages or weak-factor robust inference. Do not feed raw CPI levels into the 'shock' slot.

When supplied with observed monthly state series (e.g. inflation-growth YoY, valuations, real-rate levels, credit spreads), build strictly lagged signals (default **two whole calendar months**) and re-estimate Ridge on expanding samples before **each OOS target month**. Score conditional FMP expected returns versus expanding unconditional means on the same future months, with HAC on squared-loss gains. Publication calendars and historical revisions are **not** reconstructed by lagging current-vintage FRED data. This functionality is optional and never fills missing states or invents economic series.

## Audit / exports

- `residual_eigenvectors.csv`, `residual_eigenvalues.csv`
- `factor_neutral_weights.csv`, `factor_neutrality_tests.csv`, `realized_fmp_returns.csv`
- `incremental_premia_tests.csv`: HAC mean and alpha t/p, Holm p, OOS factor beta drift
- `holdout_oos_attribution.csv`: all 0..K augmented models, fixed coefficients, future factor returns
- `holdout_oos_pricing.csv`: per-asset future means and pricing errors
- `sdf_gmm_comparison.csv`, `sdf_gmm_coefficients.csv`
- `macro_exposures.csv`, `macro_diagnostics.csv`, `macro_sdf_pricing.csv`, `macro_sdf_coefficients.csv` (optional macro shock and one-step SDF diagnostics)
- `lagged_state_forecasts.csv`, `lagged_state_scores.csv` (optional)
- `manifest.json`: methodology and limitations.

## Next steps, **NOT IMPLEMENTED**

- Empirical study on original real market data (requires Colab/public data or provided snapshot); independent stock-level validation.
- PC selection inference including multiple comparisons across K/hypotheses and stability/rolling windows, robustness to stock overlap, nonlinearity/conditional dependence of residuals.
- RP-PCA/SPCA challenger, Giglio–Xiu three-pass, robust two-step GMM / Hansen–Jagannathan, economic prices of nontraded macro shocks, cross-asset identification.
- Tradability, cost models and investable TAA factor rotation.

Do **not** claim positive real-data premia, new academic findings or TAA alpha from passing synthetic tests.
