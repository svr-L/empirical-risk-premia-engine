> Historical methodology note. Version-specific notebook filenames refer to earlier development releases; the consolidated executable is `notebooks/01_equity_factor_discovery_and_pricing.ipynb`.

# Equity v0.11 — Residual dependence, priced risk, economic decomposition

## Scientific question

A factor model typically satisfies the **statistical** return decomposition
\(R_{i,t}^e=\alpha_i+\beta_i'F_t+\varepsilon_{i,t}\). It does **not** follow
from a nonzero OOS alpha or unexplained variance that there exists a new
priced systematic factor. Omitted priced risks, misestimated exposures, weak
factor identification, nonlinear exposures, sampling noise, anomalies/mispricing
and conditional premia can all generate discrepancies. The v0.11 design tests
these alternatives incrementally; it does not optimize the number of discovered
PCs using OOS performance.

## 1 — Residual commonality significance

- Baseline FF6+BAB+QMJ on formation-only 80 discovery sorted portfolios, 50
  validation portfolios held out by **portfolio definition** (stocks can overlap).
- Compute OLS residual covariance on the formation window only.
- Simulate 199 independent circular time shifts of each asset's residual path,
  compare top five eigenvalues with rankwise empirical null and the null maximum
  across ranks (conservative FWER test of synchronous common dependence). This
  holds individual temporal dependence and destroys synchronous cross-asset
  dependence, but does not test economic pricing nor account for fundamental
  overlap of stocks in sorted portfolios.
- Split the formation window in halves and report mean squared cosines of
  principal angles between top-K residual eigenspaces as a stability diagnostic.

## 2 — Pricing-oriented residual factor challenger

Pure PCA maximizes residual variance but need not capture priced risk. A
**pricing-weighted residual PCA** challenger extracts eigenvectors from

\[M_\gamma = \widehat\Sigma_\varepsilon + \gamma\widehat\alpha\widehat\alpha'\]

where the **formation-time-series alpha vector** is estimated in the regression
including FF/AQR and an intercept. Crucially, the in-sample OLS *residual means*
are exactly zero, so using those means in an RP-PCA penalty would be invalid.
\(\gamma\in\{0,12,60\}\) is fixed *before* OOS examination; gamma=0 exactly
reproduces the existing residual PCA direction.

- Each eigenvector is projected in formation onto the null of net-dollar
  exposure and FF/AQR beta exposures. Normalize gross exposure to 1; sign is
  determined by its largest absolute formation loading. Original asset returns,
  not residual PC score means, form the fixed-weight long-short FMP.
- Report OOS means and factor-spanning intercepts with HAC, and Holm-adjusted
  p-values across **all** predeclared gamma×PC hypotheses; no p-hacking.
- Fit one-step ridge-regularized *linear SDF moment estimator* on the distinct
  held-out validation portfolios in formation and score predicted average
  returns against the OOS realized validation means. No efficient two-stage GMM
  or formal test of nested asset-pricing models is claimed.

**Literature status:** This is a deliberately transparent RP-PCA-*inspired*
**residual** covariance-plus-alpha implementation, **not an exact published
Lettau–Pelger RP-PCA estimator** and not yet weak-factor robust inference.
References: Lettau & Pelger (2020), *Review of Financial Studies*,
https://academic.oup.com/rfs/article-abstract/33/5/2274/5756219 .

### GX-style diagnostic

Three-point-estimate stages on formation only: PCA of *raw discovery test-asset
returns* (not OLS residual scores), cross-sectional pricing of latent exposures
in validation test assets, and projection of observed FF/AQR returns onto latent
factors. An OOS pricing error is displayed for audit. It is **not** the original
Giglio–Xiu estimator's full weak-factor-robust inference nor proof that observed
traded FF factor means equal GX prices.
Reference: Giglio & Xiu (2021), *Journal of Political Economy*,
https://www.journals.uchicago.edu/doi/full/10.1086/714090 .

## 3 — Economic decomposition (automatic public observations)

`src/erp/equity_macro.py` downloads and caches **six official FRED series**:

| Series ID | Source/meaning | Feature |
|---|---|---|
| `CPIAUCSL` | CPI-U price index, monthly | log YoY inflation (%) |
| `INDPRO` | industrial production index, monthly | log YoY production growth (%) |
| `DGS10` | 10Y Treasury constant maturity yield (daily) | monthly end yield (%) |
| `DGS2` | 2Y Treasury constant maturity yield (daily) | 10Y–2Y slope (percentage points) |
| `BAA10Y` | Moody's Baa yield less 10Y UST (daily) | credit spread (percentage points) |
| `VIXCLS` | CBOE VIX (daily) | monthly end log VIX |

Automatic download uses FRED graph CSV/ZIP with a combined call followed by
per-series retries and a local cache. Exact original combined URL:
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL%2CINDPRO%2CDGS10%2CDGS2%2CBAA10Y%2CVIXCLS`.
On FRED/Colab blocking, the original CSV or ZIP can be loaded with
`FRED_LOCAL_PATH`. **No missing value filling or synthetic substitution**.
There is no reliable internet access during this release's container tests;
*real FRED download success is not claimed*.

- **Economic factor mapping:** regress known FF/AQR and discovered RPC **returns**
  contemporaneously on differenced inflation YoY, differenced industrial growth
  YoY, changes in 10Y yield, changes in credit spread and changes in log VIX.
  Training estimates and OOS out-of-sample explanatory scores are separate.
  Market moves and macro series changes are **not structurally identified
  news shocks**. Their loading cannot be interpreted automatically as a price
  of macroeconomic risk.
- **Conditional state forecasts:** latest-revised levels, strictly lagged by
  **two complete calendar months** (conservative, not a true release-calendar
  reconstruction). Expanding Ridge forecast of both standard and RPC traded
  factor returns is compared with an expanding historical mean OOS.
- **Optional one-step linear macro SDF** in existing v0.10 layer is output as a
  *proxy risk-price moment diagnostic*, **not** a validated premium to a
  nontraded macro surprise. Need event-based surprises / a verified innovation
  model with measurement error and point-in-time vintages for proper claims.

## Reproducibility, tests and outstanding challenges

- New result files: residual eigenvalue null, principal-angle stability, priced
  FMP weights and factor premia, SDF comparison, GX-style point estimates, and
  full economic mapping / state forecasts when public FRED observations load.
- All discovered directions, regression loadings, neutral weights, and SDF
  parameters are frozen before OOS evaluation; RP gamma/grid and PC count are
  predeclared. Shared constituent-stock overlap remains.
- `pytest tests/test_equity_extensions.py` executes deterministic synthetic
  tests for null detection, gamma=0 replication, FF neutrality, absence of OOS
  weight lookahead, FRED parsing/caching, and no missing-data interpolation.
- **Not completed:** stock-level independent validation, exact Lettau–Pelger
  RP-PCA / Giglio–Xiu weak-factor robust inference, priced conditional SDF,
  identified macroeconomic announcement surprises / vintage availability,
  transaction costs and practical tradability of sorted-portfolio FMPs.
- **No empirical discovery is claimed from passing synthetic code tests.**
