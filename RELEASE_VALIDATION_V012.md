# ERP v0.12 — verification report

## Directly verified

- The ERP v0.12 repository starts from the exact ERP v0.11 ZIP previously delivered. Existing Equity FF/AQR, Rates v0.9 and v0.11 residual modules remain unchanged.
- **64/64 tests passed** for the core, Equity v0.12, macro/shock loaders and ordinary Rates regressions. Command:
  `PYTHONPATH=src:tests pytest -q --ignore=tests/test_rates_v08_port.py --ignore=tests/test_rates_full_grid_short.py`.
- **6/6 additional heavy Rates tests passed**, run in a separate process:
  `PYTHONPATH=src:tests pytest -q tests/test_rates_v08_port.py tests/test_rates_full_grid_short.py`.
- **70/70 tests passed across the two groups.** An attempted all-in-one `pytest` invocation exceeded the container timeout and is **not claimed to have completed**; grouping was used to validate the same tests.
- New v0.12 tests cover: rolling FMP weights and train-only neutrality; pre-OOS no-lookahead; augmented-mean gamma=0 recovering residual PCA; paired time-block bootstrap; FMB/GMM same-sample pricing; author JK high-frequency shock identity, cache and date cutoff; no shifting sparse release observations across nonexistent months; actual-minus-consensus CPI surprises indexed by release date; and initial shock-price identification diagnostic.
- The standalone 26-cell notebook was successfully *bootstrapped in a clean temporary directory*, unpacking **ERP 0.12.0** and importing the new modules without local installation. All Colab notebooks were generated from the same current source tree.

## Not verified / not claimed

- End-to-end Colab real data download/execution (Ken French, AQR, JK, FRED) has **not** been performed in this container. Cache and original-file fallbacks remain available.
- **No new real-market premia estimates, p-values, pricing gains or macroeconomic risk-pricing findings** are claimed.
- Augmented-mean factor method is a conditional residual adaptation; no claim to the complete Lettau–Pelger RP-PCA estimator, all asymptotic results or three-pass GX weak-factor-robust inference.
- Two-pass FMB is not corrected for beta estimation error or overlapping constituent stocks; GMM SDF uses HAC weight shrinkage, so no off-the-shelf J-test distribution is asserted. JK identifying assumptions remain relevant.
- Full event-level CPI surprise data cannot be reconstructed from monthly FRED CPI alone; the optional CPI release file requires real pre-announcement consensus observations.
