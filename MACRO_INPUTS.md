# Optional economic inputs — Equity Residual Factor Research

The Equity notebook runs **without any macro data**. Optional inputs serve two independent research purposes. No sample data are automatically fabricated.

## Economic factor innovations (contemporaneous)

CSV `DATE,INFLATION_SURPRISE,GROWTH_SURPRISE,MONETARY_SHOCK,...` with one calendar-month-end row per month, all numeric innovations. These should ideally be *unexpected innovations* with documented timestamp, release calendar and units — **not** CPI or IP raw levels, not an arbitrary lagged variable. Regressions give descriptive exposures of FF/BAB/QMJ and residual FMP returns. A tentative linear one-step SDF/GMM uses these nontraded innovations to estimate candidate macro prices of risk on historical held-out portfolios. This price requires further weak-identification, alternative-asset and PIT checks and should not be treated as validated.

## Observable state variables (lagged)

CSV `DATE,CPI_YOY,IP_YOY,CREDIT_SPREAD,...`, or other numeric monthly state variables. Internal `state_lag_months=2` uses only the observation from at least 2 entire calendar months earlier for a monthly return target. Model is re-estimated only from preceding observations before forecasting. **Revised current-vintage historic series are not point-in-time news**; publication-lagging does not solve this. Calculations of YoY from levels, if needed, must not impute missing observations or use full-sample normalization. Some data may require longer publication lags.

## Required file and indexing conventions

Dates should be ISO 8601 YYYY-MM-DD and correspond to calendar months; each source must have one row per month. Values may be missing for months with no observation (not forward-filled). Do not use extraneous text columns. For meaningful asset-pricing tests, use enough months in both training (through 2005-12 by default) and OOS (2006-01 onward), and choose the risk-shock definitions *before* observing results.
