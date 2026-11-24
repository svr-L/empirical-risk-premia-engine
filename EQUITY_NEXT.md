# Next Equity research after ERP v0.12

**Current:** Rolling FF-neutral residual risk portfolio factors, FMB vs shrinkage GMM on common heldout assets, and real FOMC announcement shock-based macro mapping / cross-sectional lambda diagnostics are available. No new historical run has yet been performed with v0.12.

**Next gates:** (1) run real notebook; (2) assess subspace persistence and gross-neutral exposure drift; (3) verify incremental OOS pricing bootstrap confidence intervals and alpha Holm, not only in-sample mean; (4) stock-level nonoverlapping validation and weak factor inference; (5) CPI release-time survey consensus, then model monetary/inflation factor risk premia with careful structural identification; (6) only after credible forecast, conditional factor rotation in TAA.

---

# Equity active research queue after ERP v0.10

**Implemented now:** frozen-sample residual PCA after FF6+BAB+QMJ; long-short FMPs neutral to benchmark factor betas in training; HAC + Holm OOS factor premia tests; separate holdout portfolio families/time; common OOS risk attribution, mean-pricing tests and one-step linear SDF/GMM; optional contemporaneous macro interpretation, exploratory macro SDF premia and lagged state forecast. Full detail: `EQUITY_RESIDUAL_RESEARCH.md`.

**Immediate real-data task:** run `notebooks/Equity_Residual_Factor_Research_v0_10_PORTABLE.ipynb` using official French/AQR data, inspect the tables rather than presuming that a high-variance residual PC is priced. Compare mean/alpha significance and holdout pricing gains in common future months. The notebook's macros are explicitly optional; revised macro levels cannot identify nontraded macro risk prices by themselves.

**Next econometric stage:** RP-PCA/SPCA plus stronger SDF/GMM/HJ and Giglio–Xiu three-pass, focusing on identification and meaning of incremental factor risk prices. Then independent stock-level/outside-portfolio validation, economic shock interpretation and genuine conditional-premia forecasting subject to historical publication vintage. Only after that consider investable multi-asset rotation TAA.

**Rates:** Rates v0.9 remains unchanged and active in parallel; the Equity release does not replace the validated v0.8 core or previous Rates common-horizon forecasts.
