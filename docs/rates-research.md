# Treasury rates: funded factor portfolios and conditional bond premia

## Risk representation and tradable returns

The adapter uses fitted continuously compounded Kim–Wright US zero-coupon yields (1–10 years) and FEDFUNDS as a cash-funding proxy. PCA is applied to yield changes, scaled by training volatilities, with fixed formation ending December 2002. The three principal directions are labelled Level, Slope and Curvature. Portfolio construction imposes funded factor exposure targets, cross-factor neutrality and cash accounting. The PCA variance shares measure statistical yield risk, not proof of priced risk. Level orientation is a positive yield shock (predominantly short duration).

Zero-coupon holding returns are repriced on the aged curve. Excess return is decomposed into funded carry/roll and curve repricing. Factor portfolios use weights known before the return. See `src/erp/rates.py`, `rates_research.py` and `rates_v09.py` for exact conventions.

## Predictive experiment

The retained v0.8 research layer implements Fama–Bliss forward spreads and a Cochrane–Piazzesi-style common forward-curve predictor, trained walk-forward. Five families: BASE_STATE, FB, FB_STATE, CP, CP_STATE. Forecast horizons 3, 6 and 12 months; the strategy is rebalanced monthly, not at the forecast horizon. Data inputs include CPIAUCSL, INDPRO and T10YIE. The decomposed forecast combines carry and expected curve repricing; the diagnostics reported from v0.2.5 use the **repricing** target. Fama–Bliss and CP-style are adapted to funded factor portfolios rather than presented as exact replications of the original articles.

The subsequent v0.9 common-sample layer scores **total funded excess return** on intersected origin dates across all models, and realized P&L on separately intersected trading dates. DIRECT_FB and DIRECT_CP predict total excess return rather than the repricing residual. It reports overlapping-horizon HAC, paired block bootstrap, CP first-stage conditioning and fixed ridge-sensitivity diagnostics, and rotation comparisons against carry-only and unconditional. These extensions are implemented and have synthetic/regression test coverage, but no completed real-data v0.9 notebook was supplied for this release.

## Historical evidence

The included [executed v0.2.5 notebook](../results/evidence/rates_v025_executed.ipynb) reached all five families, three horizons and both benchmarks. See [forecast scores](../results/tables/rates_forecast_scores.csv) and [rotation scores](../results/tables/rates_rotation_scores.csv). The mean factor-level FB OOS R² against the unconditional repricing forecast was 7.66%, 11.38%, 13.48% at 3/6/12 months (222/216/204 forecast dates per factor). The original model families have different availability windows; these are not common-sample v0.9 results. Conditional rotation did not beat the simpler benchmarks in reported Sharpe at 5 bp. No deployable timing alpha is claimed.

## Reproduction and dependencies

Use `notebooks/02_rates_bond_premia.ipynb`. For the historical five-family protocol, `scripts/run_rates_v025.py` calls the retained estimator; note that historical vendor snapshots may differ and newer source fixes can affect results. Local inputs can replace network calls: Kim–Wright Board CSV, FEDFUNDS CSV, and original combined FRED macro CSV/ZIP. No point-in-time release vintages are provided. A completed v0.9 real-data run is a next research gate.

References: [Fama & Bliss (1987), The Information in Long-Maturity Forward Rates](https://www.jstor.org/stable/1814539); [Cochrane & Piazzesi (2005), Bond Risk Premia](https://www.aeaweb.org/articles?id=10.1257/0002828053828581); [Federal Reserve yield curve models](https://www.federalreserve.gov/data/yield-curve-models.htm).
